#!/usr/bin/env python3
"""bench_index.py — construit l'INDEX UNIFIE des benchmarks LLM locaux.

Une seule source de verite : le vault Obsidian (documents/llm-benchmarks/index.json)
+ une copie locale a cote de app.py pour que la page /benchmarks marche meme si le
NFS/sshfs du NAS est absent.

Ce que l'index agrege :
  * duel   : runs de prompt-duel (runs/<id>/run.json) — 32+ prompts HTML par modele
  * vitesse: logs llama-server (bench 128 tokens) -> TG, PP, VRAM, chargement
  * context: remplissages reels (fill-*.out) -> tokens ingeres, needle, prefill, pic VRAM
  * battery: batteries de 13 petites taches FR (results_*.json) -> score, latences
  * humaneval: sous-ensembles HumanEval (humaneval-summary*.json) -> score, latence

Usage (sur le .224, dans ~/apps/prompt-duel) :
    python3 bench_index.py              # scanne + copie les artefacts dans le vault + ecrit l'index
    python3 bench_index.py --no-mirror  # lit seulement ce qui est deja dans le vault
    python3 bench_index.py --print      # affiche l'index sur stdout sans ecrire
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import statistics
import sys
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.environ.get("RUNS_DIR") or os.path.join(APP_DIR, "runs")
VAULT_ROOT = os.environ.get("BENCH_VAULT_ROOT") or "/home/gab/NAS/AgentsMirror/vaults/personnel"
VAULT_DOCS = os.environ.get("BENCH_DOCS") or os.path.join(VAULT_ROOT, "documents/llm-benchmarks")
VAULT_WIKI = os.environ.get("BENCH_WIKI") or os.path.join(
    VAULT_ROOT, "wiki/casquettes/developpeur/llm-benchmarks")
LOG_SRC = os.environ.get("BENCH_LOG_SRC") or "/home/gab/llm"
MACHINE = os.environ.get("BENCH_MACHINE") or "224"
CAMPAIGN = os.environ.get("BENCH_CAMPAIGN") or f"{MACHINE}-{datetime.now():%Y-%m-%d}"
LOCAL_INDEX = os.path.join(APP_DIR, "bench_index.json")

# Libelles -> nom de modele lisible. Le plus specifique d'abord.
MODEL_PRETTY = [
    (r"^bonsai2[_-]?ptq1", "Ternary Bonsai 2 27B"),
    (r"^bonsai2[_-]?pq2", "Ternary Bonsai 2 27B"),
    (r"^bonsai2b", "Bonsai 2 1-bit"),
    (r"^bonsai1", "Bonsai 1 (1-bit)"),
    (r"^k2h37", "K2 Horizon 3.7B"),
    (r"^35b[_-]iq4nl", "Qwen3.6-35B-A3B UD-IQ4_NL"),
    (r"^35b[_-]q4kxl", "Qwen3.6-35B-A3B UD-Q4_K_XL"),
    (r"^swift15", "Swift-1.5-27B IQ4_XS"),
    (r"^27b[_-]q4kxl", "Qwen3.8-27B UD-Q4_K_XL"),
    (r"^27b[_-]q6k", "Qwen3.8-27B Q6_K"),
    (r"qwen3\.6-35b", "Qwen3.6-35B-A3B UD-IQ4_NL"),
    (r"swift-1\.5", "Swift-1.5-27B IQ4_XS"),
    (r"qwen3\.8-27b", "Qwen3.8-27B"),
    (r"flash-?next", "FlashNext 177B"),
]
CTX_SUFFIX = {"128k": 131072, "256k": 262144, "262k": 262144, "32k": 32768, "64k": 65536,
              "8k": 8192, "16k": 16384, "48k": 49152, "96k": 98304, "192k": 196608,
              "224k": 229376, "512k": 524288}

# --- conventions de nommage des cas VRAM (bonsai_case.py) ---------------------
# <famille>-<quant>[-b8][-<ctx>k][-t<n>][-mmproj]   ex. bonsai2-ptq1-32k, bonsai2-pq2-b8-16k-t2
VRAM_FAMILY = {
    "bonsai2": "Ternary Bonsai 2 27B",
    "bonsai2b": "Bonsai 2 (1-bit)",
    "bonsai1": "Bonsai 1 (1-bit)",
    "k2h37": "K2 Horizon 3.7B",
}
# taille du modele dans le label (le 8B et le 27B de la famille Bonsai 1 ne doivent
# pas se confondre dans l'index)
VRAM_SIZE = {"27b": "27B", "8b": "8B", "4b": "4B", "1.7b": "1.7B", "3.7b": "3.7B"}
VRAM_QUANT = {
    "ptq1": "PTQ1_0", "pq2": "PQ2_0", "q1": "Q1_0", "q2": "Q2_0", "q2g64": "Q2_0_g64",
    "q4km": "Q4_K_M", "q4ks": "Q4_K_S", "q4kxl": "Q4_K_XL", "q5km": "Q5_K_M",
    "q6k": "Q6_K", "q8": "Q8_0", "iq4nl": "IQ4_NL", "iq4xs": "IQ4_XS", "f16": "F16",
}


def vram_label_spec(label):
    """Decoupe un label de cas VRAM -> (famille, quant, ctx, budget_go, note)."""
    parts = [p for p in re.split(r"[-_]", label or "") if p]
    fam = None
    quant = None
    ctx = None
    size = None
    budget = 16
    note = []
    for i, p in enumerate(parts):
        low = p.lower()
        if fam is None and low in VRAM_FAMILY:
            fam = VRAM_FAMILY[low]
            continue
        if quant is None and low in VRAM_QUANT:
            quant = VRAM_QUANT[low]
            continue
        if low in VRAM_SIZE:
            size = VRAM_SIZE[low]
            continue
        m = re.fullmatch(r"(\d+)k", low)
        if m:
            ctx = int(m.group(1)) * 1024
            continue
        if low in ("b8", "8go", "8gb"):
            budget = 8
        elif low in ("b16", "16go", "16gb"):
            budget = 16
        elif low == "mmproj":
            note.append("mmproj")
        elif low.startswith("t") and low[1:].isdigit():
            note.append(f"essai {low[1:]}")
        elif low == "nokvoff":
            note.append("KV en RAM")
        else:
            note.append(p)
    if fam and size:
        fam = f"{fam} {size}"
    return fam, quant, ctx, budget, ", ".join(note) or None


def pretty_model(label):
    for rx, name in MODEL_PRETTY:
        if re.search(rx, label or "", re.I):
            return name
    return (label or "?").replace("_", " ")


def label_ctx(label, default=None):
    m = re.search(r"_(\d+k)$", label or "")
    return CTX_SUFFIX.get(m.group(1), default) if m else default


def label_mode(label, default=None):
    if re.search(r"think", label or "", re.I) and not re.search(r"nothink", label or "", re.I):
        return "think"
    if re.search(r"nothink", label or "", re.I):
        return "nothink"
    return default


def read_text(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except OSError:
        return ""


def num(v, dec=1, suffix=""):
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:,.{dec}f}".replace(",", " ") + suffix


# ---------------------------------------------------------------- duels
def scan_duels():
    out = []
    seen = {}
    for run_json in sorted(glob.glob(os.path.join(RUNS_DIR, "*", "run.json"))):
        # runs/latest est un lien vers un run déjà listé : on l'ignore
        if os.path.basename(os.path.dirname(run_json)) == "latest":
            continue
        try:
            with open(run_json, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        res = d.get("results") or []
        if not res:
            continue
        run_id = d.get("run_id") or os.path.basename(os.path.dirname(run_json))
        if run_id in seen:
            continue
        seen[run_id] = True
        params = d.get("params") or {}
        duree = sum((r.get("duree_s") or 0) for r in res)
        toks = sum((r.get("completion_tokens") or 0) for r in res)
        slug = d.get("model_slug") or ""
        out.append({
            "kind": "duel",
            "date": d.get("started_at"),
            "run_id": run_id,
            "model": pretty_model(slug),
            "model_slug": slug,
            "model_path": d.get("model_path"),
            "ctx": d.get("n_ctx"),
            "temperature": params.get("temperature"),
            "max_tokens": params.get("max_tokens"),
            "thinking": bool(params.get("enable_thinking")),
            "prompts_total": len(res),
            "prompts_ok": sum(1 for r in res if r.get("status") == "ok"),
            "duration_s": round(duree, 1),
            "tokens_out": toks,
            "tok_s": round(toks / duree, 1) if duree else None,
            "run_url": d.get("run_url") or f"/runs/{os.path.basename(os.path.dirname(run_json))}/",
            "source": os.path.relpath(run_json, os.path.dirname(RUNS_DIR)),
        })
    return out


# ---------------------------------------------------------------- vitesse
def parse_case_log(path):
    txt = read_text(path)
    label = os.path.basename(path)
    label = re.sub(r"^case-", "", label)
    label = re.sub(r"\.(full\.)?log$", "", label)
    # bench = 128 tokens de sortie, 489 tokens de prompt (bench-llama-server.py)
    tg = [float(x) for x in re.findall(
        r"eval time *=.*?/ *128 tokens.*?([0-9.]+) tokens per second", txt)]
    pp = [float(x) for x in re.findall(
        r"prompt eval time *=.*?/ *489 tokens.*?([0-9.]+) tokens per second", txt)]
    method = "bench 128 tok (mediane eval time)"
    if not tg:  # logs de case-full.sh : lignes synthetiques
        tg = [float(x) for x in re.findall(r"TG ([0-9.]+) t/s", txt)]
        pp = pp or [float(x) for x in re.findall(r"PP ([0-9.]+) t/s", txt)]
        method = "bench 128 tok (lignes run N)"
    if not tg:
        return None
    # Un vrai bench de vitesse envoie le meme prompt (489 tokens) 3 fois : sans
    # mesure de prefill, c'est une generation opportuniste (HumanEval, remplissage)
    # et le chiffre n'est pas comparable.
    if not pp or len(tg) < 2:
        return None
    vram = re.search(
        r"VRAM avant : 0, [^,]+, (\d+) MiB, (\d+) MiB \| 1, [^,]+, (\d+) MiB, (\d+) MiB", txt)
    if vram:
        vram_txt = (f"{int(vram.group(1))/1024:.1f} / {int(vram.group(3))/1024:.1f} Go")
    else:
        couple = re.findall(r"^([01]), (\d+) MiB, (\d+) MiB", txt, re.M)
        vram_txt = (" / ".join(f"{int(c[1])/1024:.1f} Go" for c in couple) + " Go") if couple else None
    pret = re.search(r"PRET en (\d+)s", txt)
    ctx = re.search(r"n_ctx_slot = (\d+)", txt)
    entry = {
        "kind": "speed",
        "date": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
        "label": label,
        "model": pretty_model(label),
        "mode": label_mode(label),
        "note": "n_max 4" if re.search(r"n4\b", label) else ("split 0,65/0,35" if "split65" in label else None),
        "ctx": int(ctx.group(1)) if ctx else label_ctx(label),
        "tg": round(statistics.mean(tg), 1),
        "tg_samples": len(tg),
        "pp": round(statistics.mean(pp), 0) if pp else None,
        "pp_samples": len(pp),
        "vram": vram_txt,
        "load_s": int(pret.group(1)) if pret else None,
        "method": method,
        "source": os.path.relpath(path, os.path.dirname(VAULT_DOCS)),
    }
    return entry


# ---------------------------------------------------------------- contexte long
def parse_fill_out(path):
    txt = read_text(path)
    label = os.path.basename(path)
    label = re.sub(r"^fill-", "", label)
    label = re.sub(r"\.out$", "", label)
    head = re.search(r"=== DEBUT fill (\S+) ([0-9: -]+) +ctx=(\d+) kv=(\S+)", txt)
    done = re.search(r'\{"stage": "done".*\}', txt)
    err = re.search(r'\{"stage": "error".*\}', txt)
    pic = re.findall(r"^GPU \d+ pic (\d+)", txt, re.M)
    entry = {
        "kind": "context",
        "date": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
        "label": label,
        "model": pretty_model(label),
        "ctx": int(head.group(3)) if head else label_ctx(label),
        "kv": head.group(4) if head else None,
        "status": "ok",
        "tokens": None,
        "needle": None,
        "prefill_tps": None,
        "peak_vram": (" / ".join(f"{int(p)/1024:.1f} Go" for p in pic)) if pic else None,
        "note": None,
        "source": os.path.relpath(path, os.path.dirname(VAULT_DOCS)),
    }
    if done:
        try:
            d = json.loads(done.group(0))
            entry["tokens"] = d.get("prompt_tokens")
            entry["needle"] = bool(d.get("needle_ok"))
            entry["prefill_tps"] = d.get("pp_tps")
            entry["elapsed_s"] = d.get("elapsed_s")
        except ValueError:
            pass
    if err:
        entry["status"] = "error"
        entry["needle"] = False
        try:
            e = json.loads(err.group(0))
            note = (e.get("error") or "").split("(")[0].strip()
            entry["note"] = f"coupure a {e.get('elapsed_s')}s : {note}"
        except ValueError:
            entry["note"] = "echec"
    return entry


# ---------------------------------------------------------------- batteries
def parse_battery_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    s = d.get("summary") or {}
    res = d.get("results") or []
    if not s or not res:
        return None
    label = s.get("label") or os.path.basename(path)
    return {
        "kind": "battery",
        "date": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
        "label": label,
        "model": pretty_model(label),
        "mode": label_mode(label) or "nothink",
        "tasks": s.get("tasks") or len(res),
        "score": s.get("score_full"),
        "score_avg": s.get("score_avg"),
        "lat_median": s.get("lat_median"),
        "lat_max": s.get("lat_max"),
        "total_wall_s": s.get("total_wall_s"),
        "fails": [r.get("id") for r in res if not r.get("score")],
        "source": os.path.relpath(path, os.path.dirname(VAULT_DOCS)),
    }


# ---------------------------------------------------------------- HumanEval
def parse_humaneval(path):
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return []
    out = []
    for key, m in (d.get("machines") or {}).items():
        disp = m.get("display_name") or m.get("model") or key
        parts = [p.strip() for p in disp.split("|")]
        out.append({
            "kind": "humaneval",
            "date": d.get("timestamp"),
            "machine": key,
            "model": parts[0] if parts else disp,
            "config": " | ".join(parts[1:]),
            "passed": m.get("passed"),
            "total": m.get("total"),
            "score_pct": m.get("score_pct"),
            "avg_latency_s": m.get("avg_latency_s"),
            "wall_s": m.get("wall_time_s") or d.get("wall_time_s"),
            "avg_tokens_per_sec": m.get("avg_tokens_per_sec"),
            "source": os.path.relpath(path, os.path.dirname(VAULT_DOCS)),
        })
    return out


# ---------------------------------------------------------------- VRAM (sampler)
def _median(vals):
    vals = sorted(vals)
    if not vals:
        return None
    n = len(vals)
    return round(vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2, 1)


def _norm_row(parts, want):
    """Realigne une ligne CSV quand awk a ecrit la decimale a la francaise
    (ancien sampler sans LC_ALL=C) : l'epoch « 1790711851,408 » occupe deux
    champs. La largeur du CSV est une CONTRAINTE : jamais de devinette muette."""
    if len(parts) == want + 1:
        parts = [parts[0], parts[1] + "." + parts[2]] + parts[3:]
    return parts if len(parts) == want else None


def parse_vram_csv(path):
    """Entree de type `vram` depuis un CSV produit par vram-sample.sh.

    Le CSV est la PREUVE : les chiffres (idle, charge, pics gen/duel/prefill) sont
    recalcules ici a partir des echantillons et des jalons de phase
    (`vram-<label>.phases`), jamais recopies d'un resume. Le pic retenu est le
    maximum sur la carte reellement chargee (celle dont l'occupation max est la
    plus forte), pas la somme des deux cartes.
    """
    base = os.path.basename(path)
    if ".rss." in base or not base.endswith(".csv"):
        return None
    label = base[len("vram-"):-len(".csv")]
    if label.startswith("probe-"):
        # cas « probe » de la recherche de contexte budget 8 Go : servent de
        # preuve pour la recherche, mais ne sont PAS publies dans l'index.
        return None
    txt = read_text(path)
    samples = []          # (epoch, gpu, used)
    for line in txt.splitlines()[1:]:
        p = _norm_row(line.split(","), 8)
        if p is None:
            continue
        try:
            samples.append((float(p[1]), int(p[2]), float(p[3]), float(p[4])))
        except ValueError:
            continue
    if not samples:
        return None
    gpus = {}
    for ep, g, used, total in samples:
        cur = gpus.setdefault(g, {"ep": [], "used": [], "total": total})
        cur["ep"].append(ep)
        cur["used"].append(used)
    # details du cas (commande exacte, extra) ecrits par bonsai_case.py
    case_path = os.path.join(LOG_SRC, f"case-{label}.json")
    try:
        with open(case_path, encoding="utf-8") as fh:
            case = json.load(fh)
    except (OSError, ValueError):
        case = {}

    # quelle carte sert ? le cas le dit (gpu_index) ; sinon, la carte dont
    # l'occupation AUGMENTE le plus (delta max-min) — jamais celle du bureau.
    want = case.get("gpu_index")
    if want in gpus:
        serving = want
    else:
        serving = max(gpus, key=lambda g: max(gpus[g]["used"]) - min(gpus[g]["used"]))
    d = gpus[serving]
    total_mib = d["total"]
    series = sorted(zip(d["ep"], d["used"]))

    phases = {}
    pm = re.search(r"\.csv$", path)
    phases_path = path[:pm.start()] + ".phases"
    for line in read_text(phases_path).splitlines():
        if "," in line:
            t, name = line.split(",", 1)
            try:
                phases.setdefault(name.strip(), float(t))
            except ValueError:
                pass

    def window(a, b):
        t0 = phases.get(a) if a else (series[0][0] if series else None)
        t1 = phases.get(b)
        if t0 is None:
            return []
        return [u for ep, u in series if ep >= t0 and (t1 is None or ep < t1)]

    after = [u for ep, u in series if "ready" in phases and ep >= phases["ready"]]

    def peak(seg):
        return round(max(seg), 1) if seg else None

    idle = window(None, "load")
    loaded = window("ready", "ready_end")
    gen = window("speed", "battery_end") or window("speed", "speed_end")
    duel_w = window("duel", "duel_end")
    fill_w = window("fill", "fill_end")
    peak_all = peak(after)
    idle_med = _median(idle)

    # RSS pic du process llama-server
    rss = None
    rss_path = path[:pm.start()] + ".rss.csv"
    if os.path.exists(rss_path):
        vals = []
        for line in read_text(rss_path).splitlines()[1:]:
            p = _norm_row(line.split(","), 5)
            if p is None:
                continue
            try:
                vals.append(int(float(p[3])))
            except ValueError:
                pass
        rss = round(max(vals) / 1024, 1) if vals else None

    cmd = case.get("cmd") or ""
    ngl = None
    m = re.search(r"-ngl\s+(\d+)", cmd)
    if m:
        ngl = int(m.group(1))
    fam, quant, ctx_label, budget, note_extra = vram_label_spec(label)
    ctx = case.get("n_ctx") or ctx_label
    gpu_name = case.get("gpu") or f"nvidia-smi #{serving} ({total_mib/1024:.0f} Go)"
    if case.get("gpu_index") is not None and serving != case.get("gpu_index"):
        gpu_name += f" [mesure sur GPU {serving}]"
    extra = (case.get("extra") or "") + " " + cmd
    notes = [n for n in [note_extra] if n]
    if "--no-kv-offload" in extra:
        notes.append("KV en RAM")
    if "--mmproj" in extra:
        notes.append("mmproj")
    if case.get("error"):
        notes.append(f"erreur : {case['error']}")
    duel = case.get("duel") or {}
    speed = case.get("speed") or {}
    return {
        "kind": "vram",
        "date": case.get("started") or datetime.fromtimestamp(
            os.path.getmtime(path)).isoformat(timespec="seconds"),
        "label": label,
        "model": fam or pretty_model(label),
        "quant": quant,
        "ctx": ctx,
        "gpu": gpu_name,
        "idle_mib": _median(idle),
        "loaded_mib": _median(loaded),
        "peak_gen_mib": peak(gen),
        "peak_duel_mib": peak(duel_w),
        "peak_prefill_mib": peak(fill_w or window("fill", "fill_end")),
        "peak_mib": peak_all,
        "peak_net_mib": (round(peak_all - (idle_med or 0), 1)
                         if peak_all is not None else None),
        "total_gb": round(peak_all / 1024, 2) if peak_all else None,
        "budget_gb": budget,
        "fits_8gb": (peak_all is not None and peak_all <= 7800),
        "fits_16gb": (peak_all is not None and peak_all <= 15800),
        "fits_8gb_net": (peak_all is not None and (peak_all - (idle_med or 0)) <= 7800),
        "fits_16gb_net": (peak_all is not None and (peak_all - (idle_med or 0)) <= 15800),
        "rss_mib": rss,
        "kv_offload": "--no-kv-offload" not in extra,
        "ngl": ngl,
        "loading_s": case.get("load_s"),
        "tg": speed.get("tg_t_s"),
        "pp": speed.get("pp_t_s"),
        "duel_run_id": duel.get("run_id"),
        "n_samples": len(series),
        "note": " · ".join(notes) or None,
        "source": os.path.relpath(path, os.path.dirname(VAULT_DOCS)),
    }


# ---------------------------------------------------------------- mirror
MIRROR_RULES = [
    ("case-*.log", "vitesse"),
    ("case-*.full.log", "vitesse"),
    ("fill-*.out", "contexte"),
    ("vram-*.csv", "vram"),
    ("vram-*.json", "vram"),
    ("vram-*.phases", "vram"),
    ("vram-*.meta.txt", "vram"),
    ("case-*.json", "vitesse"),
    ("smalltasks/results_*.json", "batteries"),
]


def _copy_if_needed(src, dest):
    """Copie src -> dest si necessaire. sshfs (vault) refuse chmod/utime : on se
    contente de copyfile et on ignore les erreurs de metadonnees."""
    try:
        if os.path.exists(dest) and os.path.getsize(dest) == os.path.getsize(src):
            return False
        shutil.copyfile(src, dest)
        try:
            os.chmod(dest, 0o644)
        except OSError:
            pass
        return True
    except PermissionError:
        # fichier deja present mais appartenant a un autre uid (montage sshfs) :
        # si la taille colle, on considere que le contenu est bon
        if os.path.exists(dest) and os.path.getsize(dest) == os.path.getsize(src):
            return False
        raise


def mirror_logs(dest_root, log_src=LOG_SRC, verbose=True):
    copied = 0
    for pattern, sub in MIRROR_RULES:
        for src in sorted(glob.glob(os.path.join(log_src, pattern))):
            dest_dir = os.path.join(dest_root, sub)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, os.path.basename(src))
            try:
                if _copy_if_needed(src, dest):
                    copied += 1
            except OSError as exc:
                if verbose:
                    print(f"  ! copie impossible {src} -> {dest} : {exc}", file=sys.stderr)
    return copied


def mirror_duels(dest_root, verbose=True):
    copied = 0
    for run_json in sorted(glob.glob(os.path.join(RUNS_DIR, "*", "run.json"))):
        run_id = os.path.basename(os.path.dirname(run_json))
        if run_id == "latest":
            continue
        dest_dir = os.path.join(dest_root, "duels", run_id)
        dest = os.path.join(dest_dir, "run.json")
        try:
            os.makedirs(dest_dir, exist_ok=True)
            if _copy_if_needed(run_json, dest):
                copied += 1
        except OSError as exc:
            if verbose:
                print(f"  ! copie impossible {run_json} : {exc}", file=sys.stderr)
    return copied


# ---------------------------------------------------------------- vault
def scan_vault_notes():
    notes = []
    for path in sorted(glob.glob(os.path.join(VAULT_WIKI, "**", "*.md"), recursive=True)):
        txt = read_text(path)[:4000]
        title = None
        for line in txt.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        notes.append({
            "path": os.path.relpath(path, VAULT_ROOT),
            "title": title or os.path.basename(path),
            "mtime": datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d"),
        })
    return notes


def build(mirror=True, verbose=True):
    if mirror:
        campaign_dir = os.path.join(VAULT_DOCS, CAMPAIGN)
        if verbose:
            print(f"-> miroir des logs {LOG_SRC} vers {campaign_dir}")
        n1 = mirror_logs(campaign_dir)
        n2 = mirror_duels(VAULT_DOCS)
        if verbose:
            print(f"   {n1} fichier(s) de log, {n2} run.json de duel")

    entries = []
    entries += scan_duels()

    campaign_dirs = [d for d in glob.glob(os.path.join(VAULT_DOCS, "*")) if os.path.isdir(d)]
    cases = []
    for root in [VAULT_DOCS] + campaign_dirs:
        for pattern in ("**/case-*.full.log", "**/case-*.log"):
            cases += glob.glob(os.path.join(root, pattern), recursive=True)
    seen = set()
    for path in sorted(set(cases)):
        # un .full.log prime : il contient en plus le bench
        base = os.path.basename(path).replace(".full.log", ".log")
        if base in seen:
            continue
        entry = parse_case_log(path)
        if entry:
            seen.add(base)
            entries.append(entry)

    for path in sorted(glob.glob(os.path.join(VAULT_DOCS, "**/fill-*.out"), recursive=True)):
        entries.append(parse_fill_out(path))
    for path in sorted(glob.glob(os.path.join(VAULT_DOCS, "**/results_*.json"), recursive=True)):
        entry = parse_battery_json(path)
        if entry:
            entries.append(entry)
    for path in sorted(glob.glob(os.path.join(VAULT_DOCS, "**/humaneval-summary*.json"), recursive=True)):
        entries += parse_humaneval(path)

    # --- empreinte VRAM (samples 0,5 s + jalons de phase) --------------------
    vram_entries = []
    for path in sorted(glob.glob(os.path.join(VAULT_DOCS, "**/vram-*.csv"), recursive=True)):
        entry = parse_vram_csv(path)
        if entry:
            vram_entries.append(entry)
    entries += vram_entries

    # croisement duel <-> VRAM : le pic du duel alimente la ligne du duel
    by_run = {e["duel_run_id"]: e for e in vram_entries if e.get("duel_run_id")}
    for e in entries:
        if e.get("kind") == "duel" and e.get("run_id") in by_run:
            v = by_run[e["run_id"]]
            e["vram_peak_duel_mib"] = v.get("peak_duel_mib")
            e["vram_total_gb"] = (round(v["peak_duel_mib"] / 1024, 2)
                                  if v.get("peak_duel_mib") else None)
            e["vram_source"] = v.get("source")

    # croisement vitesse <-> VRAM : TG/PP du meme label quand le cas ne les a pas
    speed_by_label = {e.get("label"): e for e in entries if e.get("kind") == "speed"}
    for v in vram_entries:
        s = speed_by_label.get(v.get("label"))
        if s:
            v["tg"] = v.get("tg") or s.get("tg")
            v["pp"] = v.get("pp") or s.get("pp")

    index = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": f"bench_index.py ({MACHINE})",
        "campaign": CAMPAIGN,
        "vault_root": VAULT_ROOT,
        "docs": os.path.relpath(VAULT_DOCS, VAULT_ROOT),
        "counts": {},
        "entries": entries,
        "vault_notes": scan_vault_notes(),
    }
    for e in entries:
        index["counts"][e["kind"]] = index["counts"].get(e["kind"], 0) + 1
    return index


def main():
    ap = argparse.ArgumentParser(description="Index unifie des benchmarks LLM (vault = source de verite)")
    ap.add_argument("--no-mirror", action="store_true", help="ne copie rien dans le vault")
    ap.add_argument("--print", dest="do_print", action="store_true", help="affiche l'index, n'ecrit rien")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    verbose = not args.quiet
    if not os.path.isdir(VAULT_DOCS):
        print(f"ATTENTION : dossier du vault absent ({VAULT_DOCS}) — index partiel", file=sys.stderr)

    index = build(mirror=not args.no_mirror, verbose=verbose)
    payload = json.dumps(index, ensure_ascii=False, indent=1)

    if args.do_print:
        print(payload)
        return 0

    targets = []
    if os.path.isdir(os.path.dirname(VAULT_DOCS)):
        targets.append(os.path.join(VAULT_DOCS, "index.json"))
    targets.append(LOCAL_INDEX)
    for target in targets:
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(payload + "\n")
            if verbose:
                print(f"-> ecrit {target}")
        except OSError as exc:
            print(f"! ecriture impossible {target} : {exc}", file=sys.stderr)

    if verbose:
        print("   entrees : " + ", ".join(f"{k}={v}" for k, v in sorted(index["counts"].items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
