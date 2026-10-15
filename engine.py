#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pilotage du MOTEUR d'inference local (Strata) — v1.6.5.

Pourquoi ce module existe
-------------------------
Le contexte (64k/128k/256k), le MTP et les autres leviers de vitesse ne sont PAS des
parametres de requete : ce sont des FLAGS DE LANCEMENT. `--max-context`, `--mtp DIR`,
`--spec T`, `--kv`, `--vision` sont lus par le binaire `strata` au demarrage et graves
dans le KV/la VRAM. On ne peut donc pas « choisir 128k » dans un corps de requete : il
faut relancer le moteur avec une autre configuration.

Ce module :
  * DECOUVRE le moteur en cours (process, fichier de config, args reels) ;
  * LIT son etat vivant (`/metrics` de serve/server.py : ctx, kv, spec, mtp, VRAM libre) ;
  * CALCULE une variante d'arguments (ctx, MTP on/off, KV, vision, reserve VRAM) ;
  * ECRIT un fichier de config dedie et RELANCE le moteur, en attendant qu'il reponde.

Garde-fou : `apply()` refuse de toucher au moteur si `busy()` dit qu'un duel ou un bench
tourne — relancer le moteur au milieu d'un run perdrait le run en cours.

Stdlib uniquement, comme le reste de prompt-duel.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

STRATA_DIR = Path(os.environ.get("STRATA_DIR", "/home/gab/llm/strata"))
CONFIG_DIR = Path(os.environ.get("STRATA_CONFIG_DIR", str(STRATA_DIR / "configs")))
SERVER_PY = STRATA_DIR / "serve" / "server.py"
VENV_PYTHON = STRATA_DIR / ".venv" / "bin" / "python"
DEFAULT_PORT = int(os.environ.get("LLM_PORT", "8080"))

# valeurs proposees a l'UI (le binaire strata n'accepte que celles-la)
CTX_CHOICES = (65536, 131072, 262144)
KV_CHOICES = ("int8", "fp16", "q4_0")
KV_RESIDENT_MIN = 20480
# --spec T est obligatoire pour --mtp ; sans --mtp il ne reste que la speculation par
# relecture de prompt (--suffix-draft). Les trois modes proposes :
SPEC_MODES = {
    "mtp": "MTP + speculation (le plus rapide, mais le draft peut corrompre un score)",
    "lookup": "speculation par relecture de prompt, SANS MTP (bon compromis)",
    "off": "aucune speculation — REFUSE sur un pack IQ natif (il exige --spec T, T>=2)",
}
# flags qui n'ont de sens qu'avec la speculation
_SPEC_FLAGS = ("--spec", "--spec-min-p", "--mtp", "--suffix-draft", "--mtp-max-t",
               "--spec-split", "--no-spec-split", "--mtp-window")

START_TIMEOUT_S = float(os.environ.get("ENGINE_START_TIMEOUT_S", "1800"))


# ------------------------------------------------------------------ lecture bas niveau
def _ps_args():
    try:
        out = subprocess.run(["ps", "-eo", "pid,args="], capture_output=True, text=True,
                             timeout=15).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, _, args = line.partition(" ")
        rows.append((pid.strip(), args.strip()))
    return rows


# Un shell qui MENTIONNE la commande (« bash -c cd … && python serve/server.py … »)
# n'est PAS le moteur : le tuer ne libere ni la VRAM ni le port. Le vrai process a un
# interpreteur Python en premier argument.
_PYTHON_HEADS = ("python", "python3", "python3.12", "python3.11", "python3.10")


def find_process(port=DEFAULT_PORT):
    """Le process du serveur strata pour ce port, ou None."""
    for pid, args in _ps_args():
        if "serve/server.py" not in args:
            continue
        head = args.split()[0] if args.split() else ""
        if os.path.basename(head) not in _PYTHON_HEADS:
            continue
        if f"--port {port}" in args or f"--port={port}" in args:
            cfg = None
            m = re.search(r"--config\s+(\S+)", args)
            if m:
                cfg = m.group(1)
                # Le lanceur peut passer un chemin RELATIF (« --config strata-x.json »
                # depuis /home/gab/llm/strata) : sans resolution contre le cwd du process,
                # l'app le cherche dans SON dossier et croit la configuration illisible.
                if cfg and not os.path.isabs(cfg):
                    try:
                        cfg = os.path.join(os.readlink(f"/proc/{pid}/cwd"), cfg)
                    except OSError:
                        cfg = os.path.abspath(cfg)
            return {"pid": int(pid), "args": args, "config": cfg, "port": port}
    return None


def read_config(path):
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _flag(args, name):
    """(index, valeur) du flag dans la liste d'args, ou (None, None)."""
    for i, a in enumerate(args):
        if a == name:
            return i, (args[i + 1] if i + 1 < len(args) else None)
    for i, a in enumerate(args):
        if a.startswith(name + "="):
            return i, a.split("=", 1)[1]
    return None, None


def _set_flag(args, name, value):
    """Pose/remplace un flag a valeur (value None => retire le flag et sa valeur)."""
    args = list(args)
    i, _ = _flag(args, name)
    if value is None:
        if i is not None:
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                del args[i:i + 2]
            else:
                del args[i:i + 1]
        return args
    if i is not None:
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            args[i + 1] = str(value)
        else:
            args.insert(i + 1, str(value))
        return args
    args += [name, str(value)]
    return args


def _set_switch(args, name, on):
    args = list(args)
    i, _ = _flag(args, name)
    if on and i is None:
        args.append(name)
    elif not on and i is not None:
        del args[i]
    return args


def describe(args):
    """Ce que la ligne de commande dit du moteur (independant du process)."""
    def val(name):
        return _flag(args, name)[1]
    mtp_dir = val("--mtp")
    spec = val("--spec")
    return {
        "max_context": _to_int(val("--max-context")),
        "spec": _to_int(spec),
        "spec_min_p": _to_float(val("--spec-min-p")),
        "mtp": mtp_dir,
        "suffix_draft": _to_int(val("--suffix-draft")),
        "kv": val("--kv") or "fp16",
        "kv_resident": _to_int(val("--kv-resident")),
        "vision": _flag(args, "--vision")[0] is not None,
        "vram_reserve_mib": _to_int(val("--vram-reserve-mib")),
        "expert_cache": val("--expert-cache"),
        "prefill": val("--prefill"),
        "native": val("--native"),
        "ple_gguf": val("--ple-gguf"),
        "gpu": None,
        "spec_mode": "mtp" if mtp_dir else ("lookup" if spec else "off"),
    }


def _to_int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _to_float(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _metrics(base):
    """Etat vivant publie par serve/server.py (ctx servi, kv, spec, VRAM libre)."""
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=6) as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _props(base):
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/props", timeout=6) as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def mtp_lock(args):
    """(verrouille, raison) — le MTP est-il imposé par le moteur ?

    Mesure du 30/09/2026 sur le pack IQ natif du .224 : le binaire refuse de démarrer
    sans `--spec T, --mtp DIR and --prefill CHUNK` (les DEUX). Autrement dit, sur ce
    couple moteur/pack, « MTP on/off » n'existe pas : seuls ses réglages se pilotent.
    On le dit à l'UI au lieu de laisser croire à un interrupteur.
    """
    native = "--native" in args
    if native:
        return True, ("pack IQ natif : le moteur exige --spec ET --mtp au démarrage — "
                      "le MTP ne peut pas être coupé, seuls sa profondeur (--spec), son "
                      "seuil (--spec-min-p) et la relecture de prompt (--suffix-draft) "
                      "se règlent.")
    return False, None


def state(base=None, port=DEFAULT_PORT):
    """Tout ce que l'UI doit savoir : process, config, args, et etat vivant."""
    base = base or f"http://127.0.0.1:{port}"
    proc = find_process(port)
    cfg_path = (proc or {}).get("config")
    cfg = read_config(cfg_path) if cfg_path else {}
    args = list(cfg.get("args") or [])
    info = describe(args)
    info["gpu"] = cfg.get("gpu")
    metrics = _metrics(base)
    props = _props(base)
    eng = metrics.get("engine") or {}
    live = metrics.get("live") or {}
    locked, locked_why = mtp_lock(args)
    return {
        "available": bool(SERVER_PY.exists()),
        "running": proc is not None,
        "pid": (proc or {}).get("pid"),
        "config_path": cfg_path,
        "model_name": cfg.get("model_name"),
        "port": port,
        "cmdline": (proc or {}).get("args"),
        "args": args,
        "current": info,
        "live": {
            "state": live.get("state"),
            "phase": live.get("phase"),
            "generated": live.get("generated"),
            "max_tokens": live.get("max_tokens"),
            "elapsed_s": live.get("elapsed_s"),
            "tok_s": live.get("tok_s"),
        },
        "engine": {
            "context": eng.get("context") or props.get("default_generation_settings", {}).get("n_ctx"),
            "max_context": eng.get("max_context"),
            "kv": eng.get("kv"),
            "kv_resident": eng.get("kv_resident"),
            "spec": eng.get("spec"),
            "mtp_max": eng.get("mtp_max"),
            "lookup": eng.get("lookup"),
            "images": eng.get("images"),
            "vram_free_mib": eng.get("vram_free_mib"),
            "arena_mib": eng.get("arena_mib"),
            "expert_slots": eng.get("expert_slots"),
            "version": eng.get("version"),
        },
        "mtp_locked": locked,
        "mtp_locked_reason": locked_why,
        "choices": {
            "ctx": list(CTX_CHOICES),
            "kv": list(KV_CHOICES),
            "spec_mode": sorted(SPEC_MODES),
            "kv_resident_min": KV_RESIDENT_MIN,
        },
        "spec_modes": SPEC_MODES,
    }


# ------------------------------------------------------------------ plan de relance
def plan(args, opts):
    """(nouveaux args, liste des changements lisibles, resume du resultat).

    `opts` : ctx, spec_mode, spec, spec_min_p, kv, kv_resident, vision, vram_reserve_mib,
    expert_cache, prefill. Toute cle absente n'est PAS touchee.
    """
    args = list(args)
    before = describe(args)
    changes = []

    def setv(name, value, label):
        nonlocal args
        cur = _flag(args, name)[1]
        if value is None:
            return
        if str(cur) != str(value):
            changes.append(f"{label} : {cur if cur is not None else '—'} → {value}")
        args = _set_flag(args, name, value)

    def setsw(name, on, label):
        nonlocal args
        cur = _flag(args, name)[0] is not None
        if bool(on) != cur:
            changes.append(f"{label} : {'activé' if cur else 'désactivé'} → "
                           f"{'activé' if on else 'désactivé'}")
        args = _set_switch(args, name, bool(on))

    ctx = opts.get("ctx")
    if ctx is not None:
        setv("--max-context", int(ctx), "contexte")

    mode = opts.get("spec_mode")
    if mode is not None:
        if mode not in SPEC_MODES:
            raise ValueError(f"mode de speculation inconnu : {mode}")
        mtp_dir = before.get("mtp")
        locked, why = mtp_lock(args)
        if locked and mode != "mtp":
            raise ValueError(
                "ce moteur charge un pack IQ natif : le MTP est IMPOSE par le binaire au "
                "demarrage (« needs --spec T, --mtp DIR and --prefill CHUNK »). Choisis "
                "« MTP + speculation » et regle plutot --spec, --spec-min-p ou "
                "--suffix-draft : " + (why or ""))
        if mode == "off" and "--native" in args:
            # Mesure du 30/09/2026 : le moteur refuse de demarrer sans --spec sur un pack
            # IQ natif (« this is a native (IQ) pack: it needs --native SHARD1, --spec T
            # (T >= 2) and --prefill CHUNK »). Mieux vaut le dire AVANT de tuer le moteur
            # en cours que de le laisser mourir : le seul levier reel est MTP oui/non.
            raise ValueError(
                "ce moteur charge un pack IQ natif : la speculation ne peut PAS etre coupee "
                "(le binaire exige --spec T, T>=2). Choisis « speculation sans MTP » pour "
                "desactiver le MTP, ou « MTP + speculation » pour l'activer.")
        if mode == "off":
            for f in _SPEC_FLAGS:
                if _flag(args, f)[0] is not None:
                    changes.append(f"{f} retiré (spéculation coupée)")
                args = _set_flag(args, f, None)
        else:
            spec = opts.get("spec")
            spec = int(spec) if spec else (before.get("spec") or 4)
            setv("--spec", spec, "profondeur de speculation")
            setv("--spec-min-p", opts.get("spec_min_p") or before.get("spec_min_p") or 0.5,
                 "seuil de speculation")
            if mode == "mtp":
                if not mtp_dir:
                    raise ValueError("MTP demandé mais le dossier --mtp est introuvable dans "
                                     "la configuration de départ")
                setv("--mtp", mtp_dir, "dossier MTP")
                setv("--suffix-draft", opts.get("suffix_draft") or 3, "drafts par relecture")
            else:                                   # lookup : speculation sans MTP
                if _flag(args, "--mtp")[0] is not None:
                    changes.append("--mtp retiré (MTP désactivé)")
                args = _set_flag(args, "--mtp", None)
                args = _set_flag(args, "--mtp-max-t", None)
                setv("--suffix-draft", opts.get("suffix_draft") or 8, "drafts par relecture")

    # Réglages FINS de la spéculation, applicables même sans changer de mode — c'est le
    # seul levier réel quand le MTP est imposé par le moteur (pack IQ natif).
    if opts.get("spec") is not None:
        setv("--spec", int(opts["spec"]), "profondeur de speculation")
    if opts.get("spec_min_p") is not None:
        setv("--spec-min-p", opts["spec_min_p"], "seuil de speculation")
    if opts.get("suffix_draft") is not None:
        setv("--suffix-draft", int(opts["suffix_draft"]), "drafts par relecture")

    kv = opts.get("kv")
    if kv is not None and kv != before.get("kv"):
        changes.append(f"cache KV : {before.get('kv')} → {kv}")
        args = _set_flag(args, "--kv", kv)
        if kv == "k8v4" and _flag(args, "--kv-resident")[0] is not None:
            changes.append("--kv-resident retiré (incompatible avec k8v4)")
            args = _set_flag(args, "--kv-resident", None)

    if opts.get("kv_resident") is not None:
        val = int(opts["kv_resident"])
        setv("--kv-resident", val if val > 0 else None, "KV résident")

    if opts.get("vision") is not None:
        setsw("--vision", opts["vision"], "vision")

    if opts.get("vram_reserve_mib") is not None:
        val = int(opts["vram_reserve_mib"])
        setv("--vram-reserve-mib", val if val >= 0 else None, "VRAM réservée (MiB)")

    setv("--expert-cache", opts.get("expert_cache"), "cache d'experts")
    setv("--prefill", opts.get("prefill"), "prefill")

    after = describe(args)
    return args, changes, {"before": before, "after": after}


def variant_name(cfg, after, opts):
    """Nom de fichier de config lisible : modele-ctx<k>-<spec_mode>[-kv].json"""
    slug = re.sub(r"[^a-z0-9]+", "-", str(cfg.get("model_name") or "moteur").lower()).strip("-")
    k = int((after.get("max_context") or 0) // 1024)
    bits = [slug or "moteur", f"ctx{k or 0}k", after.get("spec_mode") or "spec"]
    if opts.get("kv"):
        bits.append(str(opts["kv"]))
    return "-".join(bits) + ".json"


def write_variant(cfg, args, name):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / name
    new = dict(cfg)
    new["args"] = list(args)
    new["log"] = str(path.with_suffix(".log"))
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(new, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path


def _health(base, timeout=5):
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=timeout) as fh:
            return fh.status == 200
    except Exception:
        return False


def _stop(proc, wait_s=25.0):
    """SIGTERM puis SIGKILL sur le moteur — jamais sur un autre process."""
    pid = proc.get("pid")
    if not pid:
        return "aucun process à arrêter"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "process déjà absent"
    except PermissionError:
        return f"permission refusée pour arrêter le pid {pid}"
    t0 = time.time()
    while time.time() - t0 < wait_s:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return f"pid {pid} arrêté proprement en {time.time() - t0:.1f} s"
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
        return f"pid {pid} tué (SIGKILL) après {wait_s:.0f} s"
    except Exception as exc:
        return f"arrêt impossible : {exc}"


def engine_unit(port=DEFAULT_PORT):
    return f"prompt-duel-engine-{port}"


def _spawn(config_path, port):
    """Démarre le moteur HORS du cgroup de l'app.

    Pourquoi systemd-run : un `Popen` même avec `start_new_session` reste dans le cgroup
    de `prompt-duel.service`. Un `systemctl --user restart prompt-duel` tuait donc le
    moteur au passage (constaté le 30/09/2026 : le moteur disparaît au redémarrage de
    l'app, et il faut recharger 40 Go). Une unité transitoire possède le process :
    l'app peut redémarrer sans rien casser.

    Renvoie (popen|None, commande, chemin du journal, unité|None).
    """
    python = str(VENV_PYTHON) if VENV_PYTHON.exists() else "python3"
    cmd = [python, str(SERVER_PY), "--engine", "strata",
           "--config", str(config_path), "--port", str(port)]
    log_path = Path(str(config_path)).with_suffix(".log")
    unit = engine_unit(port)
    if shutil.which("systemd-run") and os.environ.get("XDG_RUNTIME_DIR"):
        subprocess.run(["systemctl", "--user", "reset-failed", unit],
                       capture_output=True, timeout=20)
        srun = ["systemd-run", "--user", "--unit", unit, "--collect", "--quiet",
                f"--property=WorkingDirectory={STRATA_DIR}",
                f"--property=StandardOutput=append:{log_path}",
                f"--property=StandardError=append:{log_path}",
                "--"] + cmd
        try:
            subprocess.run(srun, check=True, capture_output=True, timeout=60)
            return None, srun, str(log_path), unit
        except Exception:
            pass                                   # pas de systemd : repli direct
    log = open(log_path, "a", encoding="utf-8", buffering=1)   # noqa: SIM115 — volontairement ouvert
    proc = subprocess.Popen(cmd, cwd=str(STRATA_DIR), stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    return proc, cmd, str(log_path), None


def unit_alive(unit):
    """True si l'unité transitoire tourne encore."""
    try:
        out = subprocess.run(["systemctl", "--user", "is-active", unit],
                             capture_output=True, text=True, timeout=15).stdout
        return out.strip() in ("active", "activating", "reloading")
    except Exception:
        return False


def apply(opts, base=None, port=DEFAULT_PORT, busy=None, on_step=None):
    """Relance le moteur avec la variante demandee. Renvoie (ok, message, etat).

    `busy()` : appelable qui dit si un run/bench tourne — on refuse alors de relancer.
    `on_step(texte)` : journalisation pas a pas (le moteur met plusieurs minutes a charger).
    """
    def step(msg):
        if on_step:
            on_step(msg)

    base = base or f"http://127.0.0.1:{port}"
    if not SERVER_PY.exists():
        return False, f"serveur strata introuvable ({SERVER_PY})", None
    if busy and busy():
        return False, ("un duel ou un bench est en cours : relancer le moteur le tuerait. "
                       "Arrête-le d'abord."), None
    proc = find_process(port)
    if not proc or not proc.get("config"):
        return False, ("aucun moteur strata en cours sur ce port : je ne devine pas la "
                       "configuration de départ (lance-le une fois à la main)."), None
    cfg = read_config(proc["config"])
    if not cfg.get("args"):
        return False, f"configuration illisible : {proc['config']}", None
    try:
        args, changes, diff = plan(cfg["args"], opts or {})
    except ValueError as exc:
        return False, str(exc), None
    if not changes:
        return True, "aucun changement demandé — le moteur est déjà dans cet état", state(base, port)

    name = variant_name(cfg, diff["after"], opts or {})
    path = write_variant(cfg, args, name)
    step(f"configuration écrite : {path}")
    step("changements : " + " | ".join(changes))

    step("arrêt du moteur en cours…")
    stopped = _stop(proc)
    step(stopped)
    for _ in range(20):                       # le port doit être rendu avant de relancer
        if not _health(base, timeout=2):
            break
        time.sleep(0.5)

    log = ""
    previous = proc.get("config")
    try:
        child, cmd, log, unit = _spawn(path, port)
        step("moteur relancé : " + " ".join(cmd))
        step(f"journal : {log}" + (f" · unité systemd {unit}" if unit else ""))
    except Exception as exc:
        return False, f"relance impossible : {exc.__class__.__name__} {exc}", state(base, port)

    def rollback(raison):
        """Remet la configuration PRECEDENTE : une variante qui ne demarre pas ne doit
        jamais laisser la machine sans moteur (le modele met plusieurs minutes a charger,
        on ne veut pas le decouvrir a la main)."""
        if not previous or not Path(previous).exists():
            return "aucune configuration précédente à restaurer"
        step(f"ECHEC de la variante ({raison}) — retour a {previous}")
        try:
            _spawn(previous, port)
        except Exception as exc:
            return f"rollback impossible : {exc}"
        t0 = time.time()
        while time.time() - t0 < START_TIMEOUT_S:
            if _health(base, timeout=3):
                return f"moteur precedent restaure en {time.time() - t0:.0f} s"
            time.sleep(3)
        return "le moteur precedent n'a pas repondu non plus"

    t0 = time.time()
    last = 0.0
    while time.time() - t0 < START_TIMEOUT_S:
        mort = (child.poll() is not None) if child is not None else (
            (time.time() - t0) > 15 and not unit_alive(unit))
        if mort:
            tail = ""
            try:
                with open(log, encoding="utf-8", errors="replace") as fh:
                    tail = "".join(fh.readlines()[-6:]).strip()[-600:]
            except OSError:
                pass
            undo = rollback("le process a quitte avant d'etre pret")
            code = child.returncode if child is not None else "unité non active"
            return False, (f"la variante n'a pas démarré ({code}) : {tail} "
                           f"· {undo}"), state(base, port)
        if _health(base, timeout=3):
            st = state(base, port)
            ctx = st["engine"].get("context") or st["engine"].get("max_context")
            wanted = diff["after"].get("max_context")
            if wanted and ctx and int(ctx) != int(wanted):
                step(f"le moteur répond mais annonce ctx {ctx} au lieu de {wanted} — attente")
                time.sleep(5)
                continue
            return True, (f"moteur relancé en {time.time() - t0:.0f} s · "
                          f"ctx {ctx} · speculation {diff['after'].get('spec_mode')} · "
                          f"kv {diff['after'].get('kv')}"), st
        if time.time() - last > 20:
            last = time.time()
            step(f"chargement en cours… {time.time() - t0:.0f} s")
        time.sleep(2)
    undo = rollback(f"pas de reponse en {START_TIMEOUT_S:.0f} s")
    return False, (f"le moteur n'a pas répondu en {START_TIMEOUT_S:.0f} s — voir {log} · "
                   f"{undo}"), state(base, port)


def stop(port=DEFAULT_PORT, busy=None):
    """Arrête le moteur (utile pour libérer la VRAM)."""
    if busy and busy():
        return False, "un run est en cours — arrêt refusé"
    proc = find_process(port)
    if not proc:
        return False, "aucun moteur en cours"
    return True, _stop(proc)
