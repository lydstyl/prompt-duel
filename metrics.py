#!/usr/bin/env python3
"""metrics.py — registre des familles de tests LLM : metriques chiffrees + duree,
normalisation des entrees de l'index et agregation pour le graphique (v1.6).

Aucune dependance : stdlib uniquement. Aucune I/O : ce module ne lit rien, il
travaille sur les dictionnaires d'entrees produits par `bench_index.py`
(et les fixtures de `tests/`).

Vocabulaire :
  * famille / `kind` : le type de test (`speed`, `context`, `battery`, `humaneval`,
    `duel`, `vram` + les tests de l'app : `vitesse`, `memoire`, `intelligence`) ;
  * entree : un dictionnaire de l'index (une execution d'un test) ;
  * `entry_id` : identifiant STABLE et UNIQUE d'une entree (derive de kind+source,
    jamais de la date de generation ni de la position dans la liste) ;
  * variante : ce qui distingue deux executions d'un meme test pour un meme modele
    (ctx, MTP, thinking, temperature, max_tokens, flags, kv, libelle).

Usage en ligne de commande (debug) :
    python3 metrics.py bench_index.json --metric pp
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import unicodedata

# --------------------------------------------------------------- briques
# Un MetricSpec est un dict FIGE a 4 cles : key, label, unit, better.
# Un KindSpec est un dict FIGE a 3 cles : label, metrics (liste), duration.
# « better » : "high" = plus grand = mieux, "low" = plus petit = mieux.


def _m(key, label, unit="", better="high"):
    return {"key": key, "label": label, "unit": unit, "better": better}


# Metriques partagees par plusieurs familles (memes libelle / unite / sens).
PP = _m("pp", "Prefill", "t/s")
TG = _m("tg", "Décodage", "t/s")
PP_TPS = _m("pp_tps", "Prefill", "t/s")
TG_TPS = _m("tg_tps", "Décodage", "t/s")
PREFILL_TPS = _m("prefill_tps", "Prefill", "t/s")
TOK_S = _m("tok_s", "Débit", "t/s")
NEEDLE = _m("needle", "Aiguille retrouvée", "", "high")
TOKENS = _m("tokens", "Tokens ingérés", "tok")
LOAD_S = _m("load_s", "Chargement", "s", "low")
VRAM = _m("vram", "VRAM pic", "Go", "low")
VRAM_PEAK = _m("vram_peak", "VRAM pic", "Go", "low")
PEAK_VRAM = _m("peak_vram", "VRAM pic", "Go", "low")
SCORE_AVG = _m("score_avg", "Score moyen", "0→1")
SCORE_PCT = _m("score_pct", "Score", "%")
LAT_MEDIAN = _m("lat_median", "Latence médiane", "s", "low")
LAT_MAX = _m("lat_max", "Latence max", "s", "low")
AVG_LATENCY = _m("avg_latency_s", "Latence moyenne", "s", "low")
AVG_TOK_S = _m("avg_tokens_per_sec", "Débit", "t/s")
FAILS = _m("fails", "Tâches ratées", "n", "low")
PASSED = _m("passed", "Réussis", "tests")
TOTAL = _m("total", "Total", "tests")
WALL_S = _m("wall_s", "Durée", "s", "low")
DURATION_S = _m("duration_s", "Durée", "s", "low")
PROMPT_N = _m("prompt_n", "Tokens du prompt", "tok")
COMPLETION_N = _m("completion_n", "Tokens générés", "tok")


def _duration(key, label, unit="s"):
    return {"key": key, "label": label, "unit": unit}


# --------------------------------------------------------------- registre
# Etat reel des 68 entrees de l'index du vault + les benches de l'app (v1.6).
KINDS = {
    "speed": {
        "label": "Vitesse (bench 128 tok)",
        "metrics": [PP, TG, VRAM, LOAD_S],
        "duration": _duration("load_s", "Chargement du modèle"),
    },
    "context": {
        "label": "Contexte long (remplissage + aiguille)",
        "metrics": [PREFILL_TPS, TOKENS, NEEDLE, PEAK_VRAM],
        "duration": _duration("elapsed_s", "Remplissage complet"),
    },
    "battery": {
        "label": "Batterie FR (13 petites tâches)",
        "metrics": [SCORE_AVG, LAT_MEDIAN, LAT_MAX, FAILS],
        "duration": _duration("total_wall_s", "Batterie complète"),
    },
    "humaneval": {
        "label": "HumanEval",
        "metrics": [SCORE_PCT, AVG_LATENCY, AVG_TOK_S, PASSED, TOTAL, WALL_S],
        "duration": _duration("wall_s", "Run complet"),
    },
    "duel": {
        "label": "Duel de prompts (one-shot)",
        "metrics": [TOK_S, _m("prompts_ok", "Prompts réussis", "prompts"),
                    _m("prompts_total", "Prompts", "prompts"),
                    _m("tokens_out", "Tokens générés", "tok")],
        "duration": _duration("duration_s", "Duel complet"),
    },
    # --- benches lances par l'app (benches/<run>/bench.json) ---------------
    "vitesse": {
        "label": "Vitesse (app : prefill + TG)",
        "metrics": [PP_TPS, TG_TPS, PROMPT_N, COMPLETION_N, VRAM_PEAK],
        "duration": _duration("duree_s", "Test de vitesse"),
    },
    "memoire": {
        "label": "Mémoire de contexte (app)",
        "metrics": [NEEDLE, PREFILL_TPS, TOKENS, TG_TPS],
        "duration": _duration("duree_s", "Test de mémoire"),
    },
    "intelligence": {
        "label": "Intelligence (app : tâches FR + HumanEval)",
        "metrics": [SCORE_AVG, SCORE_PCT, PASSED, TOTAL, LAT_MEDIAN,
                    AVG_LATENCY, AVG_TOK_S],
        "duration": _duration("duree_s", "Test d'intelligence"),
    },
    # --- empreinte VRAM (sampler, deja gere par bench_index.py) ------------
    "vram": {
        "label": "Empreinte VRAM",
        "metrics": [_m("peak_mib", "Pic VRAM", "MiB", "low"),
                    _m("peak_net_mib", "Pic net (hors bureau)", "MiB", "low"),
                    _m("idle_mib", "Au repos", "MiB", "low"),
                    _m("loaded_mib", "Modèle chargé", "MiB", "low"),
                    _m("rss_mib", "RSS llama-server", "MiB", "low"),
                    _m("total_gb", "Pic VRAM", "Go", "low"),
                    PP, TG],
        "duration": _duration("loading_s", "Chargement"),
    },
}

# Alias FR acceptes a l'entree (confort pour les appels cote UI).
KIND_ALIASES = {"batterie": "battery", "contexte": "context", "battery_fr": "battery"}

# Ordre d'affichage des familles (axe X du graphique).
KIND_RANK = {
    "vitesse": 0, "speed": 0,
    "memoire": 1, "context": 1,
    "intelligence": 2, "battery": 2,
    "humaneval": 3,
    "duel": 4,
    "vram": 5,
}

# Duree de secours si la cle primaire manque (les cles varient selon la famille).
DURATION_FALLBACK = {
    "speed": ("duree_s", "elapsed_s"),
    "context": ("duree_s",),
    "battery": ("duree_s",),
    "humaneval": ("duree_s",),
    "duel": ("wall_s",),
    "vitesse": ("wall_s",),
    "memoire": ("wall_s",),
    "intelligence": ("wall_s",),
    "vram": ("duree_s",),
}

# Table des suffixes de contexte (identique a celle de bench_index.py).
CTX_SUFFIX = {
    "8k": 8192, "16k": 16384, "32k": 32768, "48k": 49152, "64k": 65536,
    "96k": 98304, "128k": 131072, "131k": 131072, "192k": 196608,
    "224k": 229376, "256k": 262144, "262k": 262144, "512k": 524288,
}


def canon_kind(kind):
    """Nom canonique d'une famille (resout les alias FR)."""
    if not kind:
        return None
    low = str(kind).strip().lower()
    return KIND_ALIASES.get(low, low)


def metric_specs(kind):
    """[{"key","label","unit","better"}] pour une famille (vide si inconnue)."""
    spec = KINDS.get(canon_kind(kind)) or {}
    return [dict(m) for m in spec.get("metrics") or []]


def duration_spec(kind):
    """{"key","label","unit"} : la duree d'un test de cette famille."""
    spec = KINDS.get(canon_kind(kind)) or {}
    return dict(spec.get("duration") or _duration("duration_s", "Durée"))


def metric_spec(metric_key, kind=None):
    """Spec d'une metrique : registre de la famille d'abord, puis registre global,
    puis spec devinee (label = cle, sans unite) pour ne jamais rien inventer."""
    if kind:
        for m in metric_specs(kind):
            if m["key"] == metric_key:
                return m
    for spec in KINDS.values():
        for m in spec["metrics"]:
            if m["key"] == metric_key:
                return dict(m)
    return _m(metric_key, str(metric_key), "", "high")


def all_metric_keys():
    """Toutes les cles de metriques connues, triees (utile a l'UI)."""
    keys = set()
    for spec in KINDS.values():
        for m in spec["metrics"]:
            keys.add(m["key"])
    return sorted(keys)


# --------------------------------------------------------------- texte / nombres
def strip_accents(text):
    """Retire les accents (NFKD) — identifiants sans accents obligatoires."""
    nfkd = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def slug(text, maxlen=None):
    """Minuscule sans accents, tout ce qui n'est pas alphanumerique -> '-'.

    "Q4_K_XL" -> "q4-k-xl" ; "Qwen3.8-27B" -> "qwen3-8-27b".
    """
    s = strip_accents(text).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if maxlen and len(s) > maxlen:
        s = s[:maxlen].rstrip("-")
    return s


def parse_number(value):
    """Valeur numerique d'un champ d'entree, ou None si non chiffrable.

    Gere : nombres, booleens (1/0), listes (nombre d'elements, ex. `fails`),
    chaines FR ("0,846"), chaines composees ("12.4 / 12.8 Go" -> 12.8 = le pic),
    tirets / vides ("-" -> None).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple, set)):
        return float(len(value))
    if isinstance(value, dict):
        return None
    text = str(value).replace("\u202f", "").replace("\xa0", "").strip()
    if not text or text in ("-", "—", "?", "n/a", "N/A", "None"):
        return None
    found = re.findall(r"-?\d+(?:[.,]\d+)?", text)
    if not found:
        return None
    vals = [float(x.replace(",", ".")) for x in found]
    return max(vals) if len(vals) > 1 else vals[0]


def _as_int(value):
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def ctx_short(ctx):
    """262144 -> "256k", 131072 -> "128k", 40000 -> "40000"."""
    n = _as_int(ctx)
    if n is None:
        return ""
    if n % 1024 == 0 and n // 1024 >= 8:
        return f"{n // 1024}k"
    return str(n)


def _short_count(n):
    """8192 -> "8k", 128 -> "128" (pour max_tokens et tokens)."""
    n = _as_int(n)
    if n is None:
        return ""
    if n % 1024 == 0 and n >= 8192:
        return f"{n // 1024}k"
    return str(n)


def _num_token(value):
    """0.37 -> "037", 0.2 -> "02", 1 -> "1" (lisible, sans point)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ""
    text = f"{f:g}".replace("-", "m").replace(".", "")
    return text


# --------------------------------------------------------------- MTP / thinking
# « MTP on/off » = le cas d'usage n°1 : ces deux indices doivent produire DEUX
# variantes distinctes. Si l'information est absente de l'entree, on retourne
# None et on n'ajoute AUCUN jeton : on n'invente jamais de MTP.
_OFF_RX = re.compile(
    r"no[-_ ]?mtp|mtp[-_ ]?(off|no|false|désactivé|desactive|0)\b|sans\s+mtp"
    r"|no[-_ ]?draft\b|draft[-_ ]?(off|no|false)\b|--no-draft")
_ON_RX = re.compile(r"\bmtp\b|\bdraft\b|speculative|spec\s*decode")


def mtp_hint(*texts):
    """True/False/None : MTP (ou draft speculatif) actif d'apres les textes.

    None = l'information n'existe pas (variante neutre).
    """
    joined = " ".join(str(t) for t in texts if t).lower()
    if not joined.strip():
        return None
    if _OFF_RX.search(joined):
        return False
    if _ON_RX.search(joined):
        return True
    return None


_THINK_OFF_RX = re.compile(r"nothink|no[-_ ]think|think(ing)?[-_ ]?(off|no)"
                           r"|sans\s+(think|reasoning)|thinking[-_ ]?off")
_THINK_ON_RX = re.compile(r"\bthinking\b|\bthink\b|reasoning|raisonnement")


def think_hint(*texts):
    """True/False/None : mode thinking d'apres les textes (None = inconnu)."""
    joined = " ".join(str(t) for t in texts if t).lower()
    if not joined.strip():
        return None
    if _THINK_OFF_RX.search(joined):
        return False
    if _THINK_ON_RX.search(joined):
        return True
    return None


# --------------------------------------------------------------- contexte
_CTX_KV_RX = re.compile(r"(?:^|[^0-9a-z])(\d{1,3})k\b", re.I)
_CTX_FULL_RX = re.compile(r"ctx\s*[:=]?\s*(\d{3,})", re.I)


def ctx_from_text(text):
    """Contexte lu dans un libelle / un texte de config, ou None.

    "35b_iq4nl_262k_prod" -> 262144 ; "ctx131072 KV q8_0" -> 131072.
    """
    if not text:
        return None
    full = _CTX_FULL_RX.search(str(text))
    if full:
        return int(full.group(1))
    for m in _CTX_KV_RX.finditer(str(text)):
        token = f"{m.group(1)}k"
        if token in CTX_SUFFIX:
            return CTX_SUFFIX[token]
    return None


# --------------------------------------------------------------- entry_id
def entry_id(entry):
    """Identifiant STABLE et UNIQUE d'une entree : `kind:source`.

    `source` est unique sur les 68 entrees reelles de l'index (chemin relatif du
    fichier d'origine, + suffixe `#<test>` pour les benches de l'app). Jamais de
    date de generation, jamais d'index de liste : deux executions du meme
    artefact gardent le meme identifiant, et reordonner la liste ne change rien.

    Sans `source`, on retombe sur une empreinte deterministe du contenu
    identifiant (dernier recours, jamais un compteur).
    """
    if not isinstance(entry, dict):
        return "?"
    kind = canon_kind(entry.get("kind")) or "?"
    source = entry.get("source") or entry.get("run_url") or entry.get("run_id")
    if source:
        return f"{kind}:{source}"
    seed = {k: entry.get(k) for k in
            ("test_id", "label", "title", "model", "model_slug", "ctx", "n_ctx",
             "config", "method", "mode", "status", "date")
            if entry.get(k) is not None}
    digest = hashlib.sha1(
        json.dumps(seed, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    return f"{kind}:anon-{digest}"


# --------------------------------------------------------------- variantes
def _model_token(*candidates):
    """Jeton court, stable et sans accents identifiant un modele.

    Longueur plafonnee a 28 caracteres : au-dela on tronque a la frontiere d'un
    jeton et on ajoute 4 caracteres d'empreinte, pour ne jamais confondre deux
    modeles (deux quantifications du meme modele restent distinctes).
    """
    for cand in candidates:
        if not cand:
            continue
        base = os.path.basename(str(cand))
        base = re.sub(r"\.gguf$", "", base, flags=re.I)
        s = slug(base)
        if not s:
            continue
        if len(s) > 28:
            s = s[:23].rstrip("-") + "-" + hashlib.md5(s.encode("utf-8")).hexdigest()[:4]
        return s
    return ""


def _ctxish(token):
    return bool(re.fullmatch(r"(ctx)?\d{1,3}k|ctx\d{3,}", token or ""))


_MTP_TOKENS = {"mtp", "nomtp", "no-mtp", "draft", "nodraft", "no-draft"}
_THINK_TOKENS = {"think", "thinking", "nothink", "no-think"}

# Mots-cles de reglage reconnus dans le NOM d'un artefact (chemin `source`).
_SOURCE_KEYWORDS = {"mtp", "nomtp", "no-mtp", "draft", "nodraft", "no-draft",
                    "f16", "bf16", "mmproj", "nokvoff"}


def _split_tokens(text):
    return [t for t in re.split(r"[^0-9A-Za-zÀ-ÿ]+", str(text or "")) if t]


def _method_flag(method):
    """« bench 128 tok (lignes run N) » -> « bench128-lignes » : lisible et stable."""
    if not method:
        return ""
    low = strip_accents(method).lower()
    num = re.search(r"bench\s*(\d+)", low)
    paren = re.search(r"\(([^)]*)\)", low)
    parts = []
    if num:
        parts.append(f"bench{num.group(1)}")
    if paren:
        first = _split_tokens(paren.group(1))
        if first:
            parts.append(slug(first[0]))
    return "-".join(p for p in parts if p)


def _flags_from_entry(entry, model_slug, ctx):
    """Jetons de reglages (sans accents) d'une entree — ordre trie, sans doublon.

    Sources : `config` (HumanEval, texte libre de flags), `method` (vitesse),
    `note` (courte), `kv`, jetons du `label` non deja portes par le modele/ctx,
    `machine`, et une eventuelle liste `flags` deja presente.
    """
    out = []
    lower_slug = slug(model_slug or "")

    def add(token):
        token = slug(token)
        if not token or token in out:
            return
        out.append(token)

    # 1. texte de config : flags libres, sauf ctx / mtp / thinking / nombres nus
    cfg = entry.get("config")
    kv_value = None
    if cfg:
        kv_match = re.search(r"\bkv\s*[:=]?\s*([0-9A-Za-z_]+)", str(cfg), re.I)
        if kv_match:
            kv_value = slug(kv_match.group(1))
        for raw in re.split(r"[|,;]+|\s+", str(cfg)):
            tok = slug(raw)
            if not tok:
                continue
            if _ctxish(tok) or tok.isdigit() or tok in _MTP_TOKENS or tok in _THINK_TOKENS:
                continue
            if tok == "kv" or (kv_value and tok == kv_value):
                continue
            add(tok)
        if kv_value:
            add(f"kv-{kv_value}")

    # 2. methode du bench de vitesse
    method_flag = _method_flag(entry.get("method"))
    if method_flag:
        add(method_flag)

    # 3. note courte ("n_max 4", "split 0,65/0,35")
    note = entry.get("note")
    if note and not isinstance(note, (list, dict)):
        note_slug = slug(note)[:24]
        if note_slug:
            add(note_slug)

    # 4. kv declare au niveau de l'entree (famille `context`)
    if entry.get("kv"):
        add(f"kv-{slug(entry.get('kv'))}")

    # 5. jetons du label qui n'appartiennent pas deja au modele / au ctx
    model_tokens = set(slug(model_slug or "").split("-"))
    label = entry.get("label")
    if label:
        for tok in _split_tokens(label):
            s = slug(tok)
            if not s or s.isdigit() or s in model_tokens:
                continue
            if _ctxish(s) or s in _MTP_TOKENS or s in _THINK_TOKENS:
                continue
            if ctx and s == ctx_short(ctx):
                continue
            add(s)

    # 6. machine (HumanEval : gabriel / louis / marie)
    if entry.get("machine"):
        add(f"mach-{entry.get('machine')}")

    # 7. liste de flags deja normalisee (reglages declares depuis l'UI)
    for raw in entry.get("flags") or []:
        add(raw)

    # 8. mots-cles de reglage du nom d'artefact (`source`) : liste BLANCHE
    #    volontairement courte. Beaucoup d'entrees HumanEval anciennes n'ont pas
    #    de `config` ; leur nom de dossier porte la seule trace reelle du reglage
    #    (« ...-nodraft-2026-08-11 » = MTP off, « ...-iq4nl-f16-32k » = KV f16).
    #    Tout le reste du chemin (dates, machine) est du bruit : jamais indexe.
    src = str(entry.get("source") or "")
    if src:
        for tok in _split_tokens(src):
            low = slug(tok)
            if low in _SOURCE_KEYWORDS:
                add(low)

    return sorted(out)


def _variant_fields(entry_or_settings):
    """Champs communs entre une entree d'index et un dict de reglages."""
    d = entry_or_settings or {}
    raw_model = d.get("model_slug") or d.get("model") or d.get("model_path")
    model_slug = _model_token(raw_model)
    ctx = (_as_int(d.get("ctx")) or _as_int(d.get("n_ctx"))
           or ctx_from_text(d.get("label")) or ctx_from_text(d.get("config")))

    texts = [d.get("config"), d.get("notes"), d.get("note"), d.get("method"),
             d.get("label"), d.get("summary"), d.get("title"), d.get("model"),
             d.get("source")]
    mtp = d.get("mtp") if isinstance(d.get("mtp"), bool) else None
    if mtp is None:
        mtp = mtp_hint(*texts)

    thinking = d.get("thinking") if isinstance(d.get("thinking"), bool) else None
    if thinking is None:
        mode = str(d.get("mode") or "").lower()
        if mode in ("think", "thinking"):
            thinking = True
        elif mode in ("nothink", "no-think"):
            thinking = False
    if thinking is None:
        thinking = think_hint(*texts)

    if isinstance(entry_or_settings, dict) and entry_or_settings.get("kind"):
        flags = _flags_from_entry(d, model_slug, ctx)
    else:  # dict de reglages : on prend la liste telle quelle, normalisee
        flags = sorted({slug(f) for f in (d.get("flags") or []) if slug(f)})

    return {
        "ctx": ctx,
        "model_slug": model_slug,
        "model": d.get("model") or d.get("model_slug") or d.get("model_path"),
        "mtp": mtp,
        "thinking": thinking,
        "temperature": d.get("temperature"),
        "max_tokens": _as_int(d.get("max_tokens")),
        "kv": d.get("kv"),
        "flags": flags,
    }


def variant_key(variant):
    """Identifiant COURT, stable, sans accents et lisible d'une variante.

    Exemple : `mtp-on_ctx256k_t037`. C'est la cle de serie du graphique ET la
    cle de masquage cote `visibility.py` : elle ne contient jamais de date.
    Le modele est prefixe quand il est connu (deux modeles ne doivent pas se
    confondre dans une meme serie) ; les reglages absents n'apparaissent pas.
    """
    if not variant:
        return "standard"
    d = dict(variant) if isinstance(variant, dict) else {}
    if not any(k in d for k in ("ctx", "n_ctx", "flags", "mtp", "thinking",
                                "temperature", "max_tokens", "model", "model_slug")):
        # appelle avec un dict de reglages minimal : on relit les champs connus
        d = _variant_fields(d)
    parts = []
    model = d.get("model_slug") or _model_token(d.get("model"))
    if model:
        parts.append(slug(model))
    mtp = d.get("mtp") if isinstance(d.get("mtp"), bool) else None
    if mtp is None:
        flags = [slug(f) for f in (d.get("flags") or [])]
        if any(f in ("nomtp", "no-mtp", "nodraft") for f in flags):
            mtp = False
        elif any(f in ("mtp", "draft") for f in flags):
            mtp = True
    if mtp is True:
        parts.append("mtp-on")
    elif mtp is False:
        parts.append("mtp-off")
    ctx = _as_int(d.get("ctx")) or _as_int(d.get("n_ctx"))
    if ctx:
        parts.append("ctx" + ctx_short(ctx))
    thinking = d.get("thinking") if isinstance(d.get("thinking"), bool) else None
    if thinking is None:
        flags = [slug(f) for f in (d.get("flags") or [])]
        if any(f in ("nothink", "no-think") for f in flags):
            thinking = False
        elif any(f in ("think", "thinking") for f in flags):
            thinking = True
    if thinking is True:
        parts.append("think")
    elif thinking is False:
        parts.append("nothink")
    if d.get("temperature") is not None:
        tok = _num_token(d.get("temperature"))
        if tok:
            parts.append("t" + tok)
    max_tokens = _as_int(d.get("max_tokens"))
    if max_tokens:
        parts.append("max" + _short_count(max_tokens))
    kv = d.get("kv")
    kv_token = "kv-" + slug(kv) if kv else ""
    if kv_token:
        parts.append(kv_token)
    extras = [f for f in sorted({slug(f) for f in (d.get("flags") or []) if slug(f)
                                 and slug(f) not in _MTP_TOKENS | _THINK_TOKENS
                                 and slug(f) != kv_token})]
    parts.extend(extras)
    key = "_".join(p for p in parts if p)
    return key or "standard"


def variant_label(variant):
    """Libelle FR d'une variante (< 60 caracteres), ou « standard » si neutre."""
    if not variant:
        return "standard"
    d = dict(variant)
    if not any(k in d for k in ("ctx", "n_ctx", "flags", "mtp", "thinking",
                                "temperature", "max_tokens", "model", "model_slug")):
        d = _variant_fields(d)
    parts = []
    model = d.get("model") or d.get("model_slug")
    if model:
        parts.append(str(model).split("(")[0].strip()[:34])
    ctx = _as_int(d.get("ctx")) or _as_int(d.get("n_ctx"))
    if ctx:
        parts.append(ctx_short(ctx))
    if d.get("mtp") is True:
        parts.append("MTP on")
    elif d.get("mtp") is False:
        parts.append("MTP off")
    if d.get("thinking") is True:
        parts.append("think")
    elif d.get("thinking") is False:
        parts.append("nothink")
    if d.get("temperature") is not None:
        parts.append(f"T {float(d['temperature']):g}")
    max_tokens = _as_int(d.get("max_tokens"))
    if max_tokens:
        parts.append("max " + _short_count(max_tokens))
    kv = d.get("kv")
    kv_token = "kv-" + slug(kv) if kv else ""
    if kv:
        parts.append("KV " + str(kv))
    parts.extend(str(f) for f in (d.get("flags") or []) if slug(f) != kv_token)
    text = " · ".join(p for p in parts if p)
    if not text:
        return "standard"
    if len(text) > 60:
        text = text[:57].rstrip(" ·") + "…"
    return text


def entry_variant(entry):
    """Variante d'une entree : ce qui distingue deux executions d'un meme test.

    -> {"key","label","ctx","flags","thinking","model"} (+ champs additifs :
       model_slug, mtp, temperature, max_tokens, kv, kind).
    """
    if not isinstance(entry, dict):
        return {"key": "standard", "label": "standard", "ctx": None, "flags": [],
                "thinking": None, "model": None, "model_slug": "", "mtp": None,
                "temperature": None, "max_tokens": None, "kv": None, "kind": None}
    fields = _variant_fields(entry)
    variant = {
        "key": variant_key(fields),
        "label": variant_label(fields),
        "ctx": fields["ctx"],
        "flags": list(fields["flags"]),
        "thinking": fields["thinking"],
        "model": fields["model"],
        "model_slug": fields["model_slug"],
        "mtp": fields["mtp"],
        "temperature": fields["temperature"],
        "max_tokens": fields["max_tokens"],
        "kv": fields["kv"],
        "kind": canon_kind(entry.get("kind")),
    }
    return variant


# --------------------------------------------------------------- duree
def entry_duration_s(entry):
    """Duree du test pour CETTE entree (la cle depend de la famille, cf. §5)."""
    if not isinstance(entry, dict):
        return None
    kind = canon_kind(entry.get("kind"))
    keys = [duration_spec(kind)["key"]] + list(DURATION_FALLBACK.get(kind, ()))
    for key in keys:
        value = parse_number(entry.get(key))
        if value is not None:
            return value
    return None


# --------------------------------------------------------------- valeurs metriques
def entry_metric_values(entry):
    """Valeurs NORMALISEES des metriques de la famille (chiffres, ou None).

    Seules les metriques du registre sont extraites ; une chaine comme
    "12.4 / 12.8 Go" devient 12.8 (le pic), `needle` devient 1.0 / 0.0, une
    liste `fails` devient son nombre d'elements. Famille inconnue : on prend
    tous les champs chiffrables de l'entree.
    """
    if not isinstance(entry, dict):
        return {}
    kind = canon_kind(entry.get("kind"))
    if kind in KINDS:
        keys = [m["key"] for m in KINDS[kind]["metrics"]]
    else:
        keys = [k for k, v in entry.items()
                if isinstance(v, (int, float, bool, list, str))
                and k not in ("kind", "source", "date", "label", "model")]
    out = {}
    for key in keys:
        if key in entry:
            out[key] = parse_number(entry[key])
    return out


def normalize_entries(entries):
    """Ajoute `entry_id`, `variant`, `metric_values` et `duration_s` a chaque entree.

    Ne modifie PAS les entrees recues (copies superficielles), ne change ni
    l'ordre ni le nombre d'entrees.
    """
    out = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        norm = dict(entry)
        norm["entry_id"] = entry_id(entry)
        norm["variant"] = entry_variant(entry)
        norm["metric_values"] = entry_metric_values(entry)
        duration = entry_duration_s(entry)
        if duration is not None or "duration_s" not in norm:
            norm["duration_s"] = duration
        out.append(norm)
    return out


# --------------------------------------------------------------- groupes
_FAMILY_FR = {
    "vitesse": "Vitesse", "memoire": "Mémoire", "intelligence": "Intelligence",
    "speed": "Vitesse", "context": "Contexte", "battery": "Batterie FR",
    "humaneval": "HumanEval", "duel": "Duel", "vram": "VRAM",
}


def _group_from_test_id(test_id):
    """Groupe d'un test de l'app a partir de son identifiant (`vitesse-256k`...)."""
    text = str(test_id)
    m = re.match(r"^([a-z]+)[-_](.+)$", text)
    fam = m.group(1) if m else text
    rest = m.group(2) if m else ""
    if fam in ("batterie", "battery"):
        num = re.sub(r"\D", "", rest)
        label = "Batterie FR" + (f" · {num} tâches" if num else "")
        return "batterie FR", label
    if fam == "humaneval":
        return text, f"HumanEval · {rest.replace('-', ' ')} problèmes"
    fr = _FAMILY_FR.get(fam, fam.capitalize())
    if rest == "court":
        return text, f"{fr} courte"
    if re.fullmatch(r"\d+k", rest):
        return text, f"{fr} {rest}"
    return text, f"{fr} {rest}".strip()


def group_of(entry):
    """(cle, libelle) du groupe = le test (axe X), pour une entree."""
    if not isinstance(entry, dict):
        return "autre", "Autres"
    kind = canon_kind(entry.get("kind"))
    test_id = entry.get("test_id") or entry.get("test")
    if test_id:
        return _group_from_test_id(test_id)
    ctx = entry.get("ctx") or entry.get("n_ctx") or ctx_from_text(entry.get("label"))
    short = ctx_short(ctx)
    if kind in ("vitesse", "speed"):
        return (f"vitesse-{short}" if short else "vitesse"), f"Vitesse {short}".strip()
    if kind == "memoire":
        return (f"memoire-{short}" if short else "memoire"), f"Mémoire {short}".strip()
    if kind == "context":
        return (f"contexte-{short}" if short else "contexte"), f"Contexte {short}".strip()
    if kind in ("intelligence", "battery", "humaneval"):
        # `intelligence` regroupe la batterie FR et HumanEval : on tranche sur les
        # metriques reellement presentes dans l'entree.
        if kind == "intelligence":
            if entry.get("score_pct") is None and entry.get("passed") is None:
                kind = "battery"
            else:
                kind = "humaneval"
        if kind == "battery":
            tasks = entry.get("tasks")
            key = "batterie FR" if tasks in (None, 13) else f"batterie FR {tasks}"
            label = f"Batterie FR · {tasks} tâches" if tasks else "Batterie FR"
            return key, label
        total = entry.get("total")
        return (f"humaneval-{total}" if total else "humaneval",
                f"HumanEval · {total} problèmes" if total else "HumanEval")
    if kind == "duel":
        n = entry.get("prompts_total")
        return (f"duel-{n}" if n else "duel"), (f"Duel · {n} prompts" if n else "Duel")
    if kind == "vram":
        label = str(entry.get("label") or "vram")
        return f"vram-{label}", f"VRAM {label}"
    return ("autre", "Autres")


def _group_sort(entry, key):
    """Ordre stable et lisible des groupes : famille, puis taille, puis cle."""
    kind = canon_kind(entry.get("kind"))
    rank = KIND_RANK.get(kind, 9)
    number = parse_number(entry.get("ctx") or entry.get("n_ctx")) or entry.get("total") or 0
    if not number:
        tail = re.search(r"(\d+)\s*$", str(key))
        number = int(tail.group(1)) if tail else 0
    return (rank, float(number or 0), str(key))


# --------------------------------------------------------------- graphique
def _recency(entry):
    """Cle de tri « le plus recent gagne » : (date ISO, entry_id)."""
    date = str(entry.get("date") or "")
    return (date, entry_id(entry))


def series_for_chart(entries, *, metric_key, kind=None, include_partial=False):
    """Donnees pretes a tracer pour UNE metrique.

    -> {"metric": {"key","label","unit","better"},
        "groups": [{"key","label"}],          # axe X : un test / une famille
        "series": [{"key","label","hidden",
                    "points": [{"group","value","duration_s","entry_id","date","status"}]}],
        "duplicates": [{"series","group","kept","dropped"}]}   # collisions tranchees

    Regles : une entree qui ne PORTE PAS la metrique demandee est ignoree (une
    entree `speed` n'a pas de `score_pct`) ; un seul point par (serie, groupe) —
    en cas de collision on garde l'entree la plus recente et on le signale.
    """
    metric = metric_spec(metric_key, kind)
    wanted = canon_kind(kind) if kind else None
    norm = normalize_entries(entries or [])
    if wanted:
        norm = [e for e in norm if canon_kind(e.get("kind")) == wanted]
    # Un duel partiel (prompts perdus en route) n'est pas un resultat comparable : il
    # reste dans l'index et dans /benchmarks, mais ne se trace pas par defaut. La page
    # /graph l'inclut sur demande (`?partiels=1`) et compte les points ecartes.
    partiels_exclus = 0
    if not include_partial:
        gardees = []
        for entry in norm:
            if entry.get("partiel"):
                partiels_exclus += 1
            else:
                gardees.append(entry)
        norm = gardees

    groups = {}          # cle -> (tri, libelle)
    buckets = {}         # cle de serie -> {"variant":..., "points": {groupe: (recency, point)}}
    duplicates = []
    hidden_seen = {}

    for entry in norm:
        values = entry.get("metric_values") or {}
        if metric_key not in values or values[metric_key] is None:
            continue
        gkey, glabel = group_of(entry)
        variant = entry.get("variant") or entry_variant(entry)
        skey = variant.get("key") or "standard"
        point = {
            "group": gkey,
            "value": values[metric_key],
            "duration_s": entry_duration_s(entry),
            "entry_id": entry.get("entry_id") or entry_id(entry),
            "date": entry.get("date"),
            "status": entry.get("status") or "ok",
        }
        if gkey not in groups:
            groups[gkey] = (_group_sort(entry, gkey), glabel)
        bucket = buckets.setdefault(skey, {"variant": variant, "points": {}})
        hidden_seen.setdefault(skey, [])
        hidden_seen[skey].append(bool(entry.get("hidden")))
        previous = bucket["points"].get(gkey)
        if previous is None:
            bucket["points"][gkey] = (_recency(entry), point)
            continue
        # collision (serie, groupe) : la plus recente gagne, l'autre est signalee
        if _recency(entry) > previous[0]:
            kept, dropped = point, previous[1]
            bucket["points"][gkey] = (_recency(entry), point)
        else:
            kept, dropped = previous[1], point
        found = None
        for dup in duplicates:
            if dup["series"] == skey and dup["group"] == gkey:
                found = dup
                break
        if found is None:
            duplicates.append({"series": skey, "group": gkey, "kept": kept["entry_id"],
                               "dropped": [dropped["entry_id"]]})
        else:
            found["kept"] = kept["entry_id"]
            if dropped["entry_id"] not in found["dropped"]:
                found["dropped"].append(dropped["entry_id"])

    ordered_groups = [{"key": key, "label": label}
                      for key, (_, label) in sorted(groups.items(), key=lambda kv: kv[1][0])]
    group_order = {g["key"]: i for i, g in enumerate(ordered_groups)}

    series = []
    for skey in sorted(buckets):
        bucket = buckets[skey]
        # bucket["points"] : {groupe: (cle de recence, point)}
        points = [pair[1] for pair in bucket["points"].values()
                  if isinstance(pair, tuple) and len(pair) == 2
                  and isinstance(pair[1], dict)]
        points.sort(key=lambda p: group_order.get(p["group"], 999))
        variant = bucket["variant"]
        series.append({
            "key": skey,
            "label": variant.get("label") or skey,
            "hidden": bool(hidden_seen.get(skey)) and all(hidden_seen[skey]),
            "points": points,
        })
    duplicates.sort(key=lambda d: (d["series"], d["group"]))
    return {"metric": metric, "groups": ordered_groups, "series": series,
            "duplicates": duplicates, "partiels_exclus": partiels_exclus,
            "include_partial": bool(include_partial)}


# --------------------------------------------------------------- CLI de debug
def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="Resume d'un index de benchmarks")
    ap.add_argument("index", help="fichier index.json (ou - pour stdin)")
    ap.add_argument("--metric", default="pp")
    ap.add_argument("--kind", default=None)
    args = ap.parse_args(argv)
    if args.index == "-":
        data = json.load(sys.stdin)
    else:
        with open(args.index, encoding="utf-8") as fh:
            data = json.load(fh)
    entries = data.get("entries") if isinstance(data, dict) else data
    norm = normalize_entries(entries)
    print(f"entrées : {len(norm)}   ids uniques : {len({e['entry_id'] for e in norm})}")
    data_chart = series_for_chart(entries, metric_key=args.metric, kind=args.kind)
    points = sum(len(s["points"]) for s in data_chart["series"])
    print(f"métrique {args.metric} : {len(data_chart['series'])} séries, "
          f"{len(data_chart['groups'])} groupes, {points} points, "
          f"{len(data_chart['duplicates'])} doublon(s)")
    for s in data_chart["series"]:
        print(f"  · {s['key']}  [{s['label']}]  {len(s['points'])} point(s)")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
