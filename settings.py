#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
settings.py — réglages RÉELS du serveur llama.cpp local (chantier v1.6).

But : l'app doit savoir *quels réglages le serveur sert vraiment* au moment d'un
run, pour distinguer deux variantes qui ne diffèrent que par un flag (« avec ou
sans MTP », « ctx 128k ou 256k », « tel ou tel flag »). MTP / draft fausse par
ailleurs les scores HumanEval (~15 % d'écart historique) : la variante doit donc
être visible partout (label + clé stable).

Deux sources, fusionnées :
  1. la **cmdline du process** llama.cpp (lue dans `/proc/*/cmdline`) : vérité du
     lancement — `--mtp`, `-c 262144`, `-fa on`, `-ctk turbo4`, etc. ;
  2. l'endpoint **`/props`** du serveur : contexte réellement annoncé (`n_ctx`),
     modèle chargé, version de build.

Règles tenues :
  * stdlib uniquement, Python 3.11+ ;
  * `detect()` ne lève JAMAIS d'exception (serveur éteint, /proc vide, droits,
    JSON invalide) et renvoie TOUJOURS un dict complet — `None` = inconnu ;
  * aucun flag n'est jeté en silence : tout ce qui n'est pas reconnu reste listé
    brut dans `flags` (process) et `declared_flags` (UI) ;
  * aucune hypothèse silencieuse : `--mtp` absent d'une cmdline *lisible* ⇒
    `mtp=False` (le serveur ne l'a pas) ; cmdline *illisible* ⇒ `mtp=None`.

Testabilité : `detect(port, props_json=…, proc_snapshot=…)` n'accède NI au réseau
NI à /proc dès que l'injection correspondante est fournie (`props_json` coupe
l'appel HTTP, `proc_snapshot` coupe le scan de /proc).
"""

import json
import os
import re
import unicodedata
import urllib.request

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

# Timeout court pour /props : un serveur éteint doit répondre vite.
PROPS_TIMEOUT_S = 3.0

# Port par défaut de llama.cpp (utilisé quand le process ne passe pas --port).
DEFAULT_LLM_PORT = 8080

# Formats de cache KV considérés comme « standard » (non affichés).
_KV_STANDARD = {"f16", "f32", "bf16"}

# --------------------------------------------------------------------------
# Table des flags llama.cpp reconnus
# --------------------------------------------------------------------------
# alias (court/long) -> clé interne. Toute option absente de cette table est
# conservée brute dans `flags` et comptée dans `unknown_flags`.
_FLAG_ALIASES = {
    "-c": "ctx", "--ctx-size": "ctx",
    "--mtp": "mtp", "--no-mtp": "no_mtp",
    "-ngl": "ngl", "--n-gpu-layers": "ngl",
    "-fa": "fa", "--flash-attn": "fa",
    "-ctk": "ctk", "--cache-type-k": "ctk",
    "-ctv": "ctv", "--cache-type-v": "ctv",
    "--n-cpu-moe": "n_cpu_moe",
    "--draft-max": "draft_max",
    "--draft-min": "draft_min",
    "--draft-p-min": "draft_p_min",
    "--split-mode": "split_mode",
    "-ts": "tensor_split", "--tensor-split": "tensor_split",
    "--jinja": "jinja",
    "--no-mmap": "no_mmap", "--mmap": "mmap",
    "-np": "parallel", "--parallel": "parallel",
    "--rope-scaling": "rope_scaling",
    "--reasoning-format": "reasoning_format",
    "-m": "model", "--model": "model",
    "--port": "port", "--host": "host",
}

# Flags qui prennent une valeur (le token suivant).
_VALUE_KINDS = {
    "ctx", "ngl", "ctk", "ctv", "n_cpu_moe", "draft_max", "draft_min",
    "draft_p_min", "split_mode", "tensor_split", "parallel", "rope_scaling",
    "reasoning_format", "model", "port", "host",
}

# Valeurs textuelles acceptées après -fa/--flash-attn (arg optionnel).
_FA_VALUES = {
    "on": True, "off": False, "auto": None,
    "true": True, "false": False, "yes": True, "no": False,
    "1": True, "0": False, "o": True, "n": False,
}

# --------------------------------------------------------------------------
# Dictionnaire canonique (toutes les clés toujours présentes)
# --------------------------------------------------------------------------

def _empty():
    """Dict complet : chaque clé existe, valeur None = inconnu."""
    return {
        # contexte
        "n_ctx": None,             # contexte retenu (cmdline prioritaire, sinon /props)
        "n_ctx_cmdline": None,     # valeur de -c/--ctx-size du process
        "n_ctx_props": None,       # valeur annoncée par /props (par slot si --parallel)
        # décodage spéculatif
        "mtp": None,               # True/False/None
        "draft": None,             # True/False/None (flags --draft-* présents)
        "draft_max": None, "draft_min": None, "draft_p_min": None,
        # GPU / mémoire / KV
        "flash_attn": None,        # True/False/None
        "n_gpu_layers": None,      # int si numérique, sinon None (« tuned »)
        "n_gpu_layers_raw": None,  # valeur brute (-ngl tuned …)
        "n_cpu_moe": None,
        "cache_type_k": None, "cache_type_v": None,
        "kv": None,                # ex. "turbo4/turbo3"
        "split_mode": None, "tensor_split": None,
        "mmap": None,              # --no-mmap => False
        # divers utiles au run
        "parallel": None, "jinja": None,
        "rope_scaling": None, "reasoning_format": None,
        # identité du modèle
        "model_path": None, "model": None,   # model = nom court lisible
        "build": None,
        "default_generation_settings": None, "total_slots": None,
        # provenance
        "flags": [],               # flags BRUTS de la cmdline (process)
        "declared_flags": [],      # flags BRUTS déclarés depuis l'UI
        "unknown_flags": [],       # sous-ensemble non reconnu (compté, jamais jeté)
        "source": None,            # "cmdline" | "props" | "declared"
        "port": None,
        "error": None,
    }


# --------------------------------------------------------------------------
# Petits convertisseurs tolérants
# --------------------------------------------------------------------------

def _to_int(v):
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, str):
            s = v.strip()
            if re.fullmatch(r"-?\d+", s):
                return int(s)
            return None
        return int(v)
    except Exception:
        return None


def _to_float(v):
    try:
        if isinstance(v, bool):
            return None
        return float(v)
    except Exception:
        return None


def _to_ctx(v):
    """« 128k » / "131072" / 131072 -> 131072. Sinon None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    s = str(v).strip().lower().replace(" ", "").replace("_", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)k", s)
    if m:
        return int(float(m.group(1)) * 1024)
    if re.fullmatch(r"\d+", s):
        return int(s)
    return None


def _split_flags(text):
    """Découpe une chaîne de flags (« --mtp -fa on ») en tokens, guillemets gérés."""
    if isinstance(text, (list, tuple)):
        return [str(t) for t in text if str(t) != ""]
    if not isinstance(text, str):
        return []
    import shlex  # stdlib
    try:
        return shlex.split(text)
    except Exception:
        return text.split()


# --------------------------------------------------------------------------
# Parsing d'une liste de flags (cmdline process OU déclaration UI)
# --------------------------------------------------------------------------

def _parse_flags(tokens):
    """
    tokens : liste d'arguments (ex. argv[1:], ou flags déclarés).

    Retourne (parsed, unknown, positional) :
      * parsed     : dict {clé interne -> valeur brute} pour les flags reconnus ;
      * unknown    : tokens de flags non reconnus (jamais jetés en silence) ;
      * positional : premier argument non-flag (chemin du modèle, ex. après -m).
    """
    parsed, unknown, positional = {}, [], None
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        name, inline = tok, None
        if tok.startswith("--") and "=" in tok:
            name, inline = tok.split("=", 1)

        key = _FLAG_ALIASES.get(name)
        if key is None:
            # token non reconnu : on le garde BRUT s'il ressemble à un flag
            if tok.startswith("-") and tok not in ("-", "--") and not re.fullmatch(r"-\d+(\.\d+)?", tok):
                unknown.append(tok)
            elif not tok.startswith("-") and positional is None:
                positional = tok
            i += 1
            continue

        if key in ("mtp", "no_mtp", "jinja", "no_mmap", "mmap"):
            parsed[key] = True
            i += 1
        elif key == "fa":
            # -fa peut être nu, ou suivi de on/off/auto/o/n…
            nxt = tokens[i + 1] if i + 1 < n else None
            if nxt is not None and nxt.lower() in _FA_VALUES:
                parsed["fa"] = nxt.lower()
                i += 2
            else:
                parsed["fa"] = "on"  # flag nu => activé
                i += 1
        else:  # flag à valeur
            if inline is not None:
                parsed[key] = inline
                i += 1
            else:
                parsed[key] = tokens[i + 1] if i + 1 < n else ""
                i += 2
    return parsed, unknown, positional


def _fields_from_parsed(parsed):
    """Convertit le dict brut du parser en champs normalisés du contrat."""
    f = {}
    if "ctx" in parsed:
        f["n_ctx"] = _to_ctx(parsed["ctx"])
    if "mtp" in parsed:
        f["mtp"] = True
    if "no_mtp" in parsed:
        f["mtp"] = False
    if "ngl" in parsed:
        raw = parsed["ngl"]
        f["n_gpu_layers_raw"] = raw
        f["n_gpu_layers"] = _to_int(raw)  # « tuned » / « auto » -> None
    if "fa" in parsed:
        f["flash_attn"] = _FA_VALUES.get(str(parsed["fa"]).lower(), None)
    if "ctk" in parsed:
        f["cache_type_k"] = parsed["ctk"]
    if "ctv" in parsed:
        f["cache_type_v"] = parsed["ctv"]
    if "n_cpu_moe" in parsed:
        f["n_cpu_moe"] = _to_int(parsed["n_cpu_moe"])
    if "parallel" in parsed:
        f["parallel"] = _to_int(parsed["parallel"])
    if "draft_max" in parsed:
        f["draft_max"] = _to_int(parsed["draft_max"])
    if "draft_min" in parsed:
        f["draft_min"] = _to_int(parsed["draft_min"])
    if "draft_p_min" in parsed:
        f["draft_p_min"] = _to_float(parsed["draft_p_min"])
    if "split_mode" in parsed:
        f["split_mode"] = parsed["split_mode"]
    if "tensor_split" in parsed:
        f["tensor_split"] = parsed["tensor_split"]
    if "jinja" in parsed:
        f["jinja"] = True
    if "no_mmap" in parsed:
        f["mmap"] = False
    elif "mmap" in parsed:
        f["mmap"] = True
    if "rope_scaling" in parsed:
        f["rope_scaling"] = parsed["rope_scaling"]
    if "reasoning_format" in parsed:
        f["reasoning_format"] = parsed["reasoning_format"]
    if "model" in parsed:
        f["model_path"] = parsed["model"]
    return f


def _combine_kv(out):
    """Renseigne `kv` à partir de cache_type_k / cache_type_v."""
    k, v = out.get("cache_type_k"), out.get("cache_type_v")
    if k or v:
        out["kv"] = f"{k or 'f16'}/{v or 'f16'}"


# --------------------------------------------------------------------------
# Lecture des sources réelles
# --------------------------------------------------------------------------

def _fetch_props(base_url):
    """GET /props (timeout court). Lève sur erreur — l'appelant protège."""
    url = base_url.rstrip("/") + "/props"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=PROPS_TIMEOUT_S) as resp:
        body = resp.read().decode("utf-8", "replace")
    return json.loads(body)


def _coerce_props(props_json):
    """props_json peut être un dict, une chaîne JSON ou des bytes."""
    if props_json is None:
        return None
    if isinstance(props_json, dict):
        return props_json
    if isinstance(props_json, (bytes, bytearray)):
        props_json = bytes(props_json).decode("utf-8", "replace")
    if isinstance(props_json, str):
        return json.loads(props_json or "{}")
    return None


def _scan_proc():
    """Liste [{pid, cmdline:[args]}] de tous les process dont la cmdline est lisible."""
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                raw = fh.read()
        except Exception:
            continue  # droits, process disparu… : on ignore
        if not raw:
            continue
        args = [a for a in raw.decode("utf-8", "replace").split("\x00") if a != ""]
        if args:
            found.append({"pid": int(entry), "cmdline": args})
    return found


def _looks_like_llama(args):
    """True si la cmdline ressemble à un vrai binaire llama.cpp (llama-server…).

    On compare le *nom du binaire* (argv[0]) et non une sous-chaîne : sinon
    « ollama serve » serait pris pour un llama-server (il contient « llama »).
    """
    if not args:
        return False
    exe = os.path.basename(str(args[0])).lower()
    if exe in ("llama", "llama-server", "llama-cli", "llama-cpp", "llamaserver"):
        return True
    return exe.startswith("llama-") or "llama.cpp" in exe


def _match_port(args, port):
    """Un process correspond si son --port == port, ou s'il n'en a pas (défaut 8080)."""
    declared = None
    for i, a in enumerate(args):
        if a == "--port" and i + 1 < len(args):
            declared = args[i + 1]
        elif str(a).startswith("--port="):
            declared = str(a).split("=", 1)[1]
    if declared is None:
        return int(port) == DEFAULT_LLM_PORT
    try:
        return int(declared) == int(port)
    except Exception:
        return False


def _pick_proc(snapshot, port):
    """Choisit la cmdline du process llama.cpp lié à `port` dans un snapshot."""
    for proc in snapshot or []:
        argv = proc.get("cmdline") if isinstance(proc, dict) else proc
        argv = _split_flags(argv) if isinstance(argv, str) else list(argv or [])
        if not argv or not _looks_like_llama(argv):
            continue
        if _match_port(argv, port):
            return [str(a) for a in argv]
    return None


def _apply_props(out, props):
    """Renseigne les champs issus de /props."""
    if not isinstance(props, dict) or not props:
        return
    dgs = props.get("default_generation_settings")
    if isinstance(dgs, dict):
        out["default_generation_settings"] = dgs
        out["n_ctx_props"] = _to_ctx(dgs.get("n_ctx"))
    if out["n_ctx_props"] is None:
        out["n_ctx_props"] = _to_ctx(props.get("n_ctx"))
    out["model_path"] = out["model_path"] or props.get("model_path") or props.get("model")
    out["build"] = props.get("build_info") or props.get("build")
    out["total_slots"] = _to_int(props.get("total_slots"))


# --------------------------------------------------------------------------
# Détection (API principale)
# --------------------------------------------------------------------------

def detect(port=DEFAULT_LLM_PORT, props_json=None, proc_snapshot=None, base_url=None):
    """
    Réglages réellement servis, fusionnés cmdline + /props.

    port          : port du serveur llama.cpp (8080 par défaut).
    props_json    : dict/str/bytes de /props — si fourni, AUCUN appel réseau.
    proc_snapshot : [{“pid”, “cmdline”}] — si fourni, AUCUN scan de /proc.
    base_url      : base HTTP à interroger (défaut http://127.0.0.1:<port>).

    Ne lève jamais : en cas de pépin, un dict complet est renvoyé avec `error`.
    """
    try:
        return _detect(port, props_json, proc_snapshot, base_url)
    except Exception as exc:  # filet de sécurité absolu
        out = _empty()
        out["port"] = port
        out["source"] = "declared"
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def _detect(port, props_json, proc_snapshot, base_url):
    out = _empty()
    out["port"] = int(port) if str(port).lstrip("-").isdigit() else port
    errors = []

    # --- source 1 : /props -------------------------------------------------
    props = None
    if props_json is not None:
        try:
            props = _coerce_props(props_json)
        except Exception as exc:
            errors.append(f"/props injecté illisible ({type(exc).__name__})")
    else:
        try:
            url = base_url or f"http://127.0.0.1:{port}"
            props = _fetch_props(url)
        except Exception as exc:
            errors.append(f"/props indisponible ({type(exc).__name__})")

    # --- source 2 : cmdline du process -------------------------------------
    argv = None
    if proc_snapshot is not None:
        argv = _pick_proc(proc_snapshot, port)
    else:
        try:
            argv = _pick_proc(_scan_proc(), port)
        except Exception as exc:
            errors.append(f"/proc illisible ({type(exc).__name__})")

    cmdline_readable = argv is not None

    # --- fusion ------------------------------------------------------------
    if cmdline_readable:
        raw_flags = argv[1:] if (argv and not argv[0].startswith("-")) else list(argv)
        parsed, unknown, positional = _parse_flags(raw_flags)
        fields = _fields_from_parsed(parsed)
        out["flags"] = raw_flags
        out["unknown_flags"] = unknown
        for k, v in fields.items():
            out[k] = v
        out["n_ctx_cmdline"] = fields.get("n_ctx")
        if out["model_path"] is None and positional:
            out["model_path"] = positional
        # Règle explicite (contrat §6) : cmdline lisible et --mtp absent => False.
        if out["mtp"] is None:
            out["mtp"] = False
        # draft = présence d'un flag --draft-* (cmdline lisible => False si absent).
        if out["draft"] is None:
            out["draft"] = any(
                out[k] is not None
                for k in ("draft_max", "draft_min", "draft_p_min")
            )
        out["source"] = "cmdline"

    _apply_props(out, props)
    if not cmdline_readable and isinstance(props, dict) and props:
        out["source"] = "props"
    if out["source"] is None:
        out["source"] = "declared"  # rien de détecté : on retombe sur du déclaré

    # n_ctx retenu : la cmdline fait foi (c'est le -c réellement lancé) ; /props
    # sert de secours et reste exposé séparément via n_ctx_props.
    out["n_ctx"] = out["n_ctx_cmdline"] if out["n_ctx_cmdline"] is not None else out["n_ctx_props"]
    _combine_kv(out)
    out["model"] = short_model(out.get("model_path"))
    if errors:
        out["error"] = " | ".join(errors)
    return out


# --------------------------------------------------------------------------
# Réglages déclarés (UI) -> même forme
# --------------------------------------------------------------------------

def normalize(raw):
    """
    Normalise des réglages DÉCLARÉS depuis l'UI vers la forme du contrat.

    Accepte : {"flags": "--mtp -fa on -c 262144"} ou {"n_ctx": "256k", ...}
    ou directement un dict déjà canonique. Absence de `--mtp` => mtp=None
    (non spécifié), jamais False : une déclaration n'est pas une cmdline.
    """
    out = _empty()
    if not isinstance(raw, dict):
        out["source"] = "declared"
        return out

    src_flags = raw.get("flags")
    if src_flags is None:
        src_flags = raw.get("declared_flags", [])
    tokens = _split_flags(src_flags) if isinstance(src_flags, (str, list, tuple)) else []

    parsed, unknown, positional = _parse_flags(tokens)
    fields = _fields_from_parsed(parsed)
    for k, v in fields.items():
        out[k] = v
    if out["model_path"] is None and positional:
        out["model_path"] = positional

    # Champs scalaires explicites : ils priment sur les flags (plus récents/plus précis).
    for key, conv in (("n_ctx", _to_ctx), ("model_path", str), ("build", str),
                      ("cache_type_k", str), ("cache_type_v", str)):
        if raw.get(key) is not None:
            out[key] = conv(raw[key])
    for key in ("flash_attn", "mtp", "draft", "jinja", "mmap", "split_mode",
                "tensor_split", "rope_scaling", "reasoning_format"):
        if raw.get(key) is not None:
            out[key] = raw[key]
    for key in ("n_gpu_layers", "n_gpu_layers_raw", "n_cpu_moe", "parallel",
                "draft_max", "draft_min", "draft_p_min", "total_slots"):
        if raw.get(key) is not None:
            out[key] = raw[key]

    out["flags"] = tokens
    out["declared_flags"] = tokens
    out["unknown_flags"] = unknown
    out["source"] = raw.get("source") or "declared"
    out["default_generation_settings"] = raw.get("default_generation_settings")
    _combine_kv(out)
    out["model"] = short_model(out.get("model_path"))
    return out


# --------------------------------------------------------------------------
# Modèle : nom court lisible
# --------------------------------------------------------------------------

_QUANT_RE = re.compile(r"^(UD|IQ\d|Q\d|K_|BPW|F16|BF16|MXFP|M\d|GGUF|Q\d_K)", re.I)


def short_model(model_path):
    """« /…/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf » -> « Qwen3.6-35B-A3B »."""
    if not model_path:
        return None
    name = os.path.basename(str(model_path))
    for ext in (".gguf", ".GGUF", ".bin"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    parts = [p for p in re.split(r"[-_]", name) if p]
    kept = []
    for p in parts:
        if _QUANT_RE.match(p):  # fin du nom « utile » : quant / format
            break
        kept.append(p)
    if not kept:
        kept = parts[:2]
    kept = kept[:3]
    return "-".join(kept) if kept else (name or None)


# --------------------------------------------------------------------------
# Libellé FR (<= 60 car.) et clé stable
# --------------------------------------------------------------------------

def _ctx_text(n_ctx):
    if not isinstance(n_ctx, int) or n_ctx <= 0:
        return None
    if n_ctx % 1024 == 0:
        return f"ctx {n_ctx // 1024}k"
    return f"ctx {n_ctx}"


def variant_label(settings):
    """
    Libellé FR court et lisible (<= 60 caractères) d'une variante.
    Ex. « MTP on · ctx 256k · Qwen3.6-35B », « sans MTP · ctx 128k · flash-attn ».
    Un signe discret « +2 flags » signale les flags inconnus.
    """
    s = settings if isinstance(settings, dict) else {}
    mtp = s.get("mtp")
    parts = []
    if mtp is True:
        parts.append("MTP on")
    elif mtp is False:
        parts.append("sans MTP")
    ctx = _ctx_text(s.get("n_ctx"))
    if ctx:
        parts.append(ctx)
    if s.get("flash_attn") is True:
        parts.append("flash-attn")
    kv = s.get("kv")
    if kv and kv.split("/")[0].lower() not in _KV_STANDARD:
        parts.append(f"kv {kv}")
    if s.get("model"):
        parts.append(str(s["model"]))

    unknown = s.get("unknown_flags") or []
    suffix = f"+{len(unknown)} flags" if unknown else ""

    if not parts:  # rien de connu : libellé explicite (jamais une chaîne vide)
        return suffix or "réglages inconnus"

    # Respect strict de la limite de 60 caractères : on rogne les extras d'abord.
    def join(ps, sfx=suffix):
        txt = " · ".join(ps)
        if sfx:
            txt = f"{txt} {sfx}" if txt else sfx
        return txt

    if len(join(parts)) <= 60:
        return join(parts)
    # 1) on retire kv, 2) flash-attn, 3) on raccourcit / tronque.
    for drop in ("kv", "flash-attn"):
        parts = [p for p in parts if not p.startswith(drop)]
        if len(join(parts)) <= 60:
            return join(parts)
    txt = join(parts)
    if len(txt) > 60:
        txt = txt[:59].rstrip(" ·") + "…"
    return txt


def _slug(text):
    """minuscules, sans accent ni espace — utilisable comme clé."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text


def variant_key(settings):
    """
    Identifiant court, stable, sans accent ni espace (clé de série / clé de
    masquage). Ex. « mtp-on_ctx256k_fa_qwen3-6-35b ».
    """
    s = settings if isinstance(settings, dict) else {}
    comps = []

    mtp = s.get("mtp")
    if mtp is True:
        comps.append("mtp-on")
    elif mtp is False:
        comps.append("mtp-off")
    else:
        comps.append("mtp-na")

    n_ctx = s.get("n_ctx")
    if isinstance(n_ctx, int) and n_ctx > 0:
        comps.append(f"ctx{n_ctx // 1024}k" if n_ctx % 1024 == 0 else f"ctx{n_ctx}")
    else:
        comps.append("ctx-na")

    if s.get("flash_attn") is True:
        comps.append("fa")
    elif s.get("flash_attn") is False:
        comps.append("no-fa")

    kv = s.get("kv")
    if kv and kv.split("/")[0].lower() not in _KV_STANDARD:
        comps.append(f"kv-{_slug(kv.replace('/', '-'))}")

    if s.get("model"):
        comps.append(_slug(s["model"]))

    unknown = s.get("unknown_flags") or []
    if unknown:
        comps.append(f"x{len(unknown)}flags")

    if not comps:
        return "default"
    return "_".join(_slug(c) for c in comps if c)


# --------------------------------------------------------------------------
# Ligne de commande (debug)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    _port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LLM_PORT
    print(json.dumps(detect(_port), ensure_ascii=False, indent=1))
