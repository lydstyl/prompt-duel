#!/usr/bin/env python3
"""bench_tests.py — tests cochables de VITESSE, MÉMOIRE DE CONTEXTE et INTELLIGENCE.

Module **stdlib seule**, importé par `app.py` (prompt-duel). Il porte trois choses :

  1. le CATALOGUE des tests cochables dans l'UI (id, groupe, titre, description,
     prérequis de contexte, méthode d'estimation) ;
  2. l'ESTIMATION de durée avant lancement — jamais devinée : elle vient des mesures
     déjà faites sur ce modèle (`bench_calib.json`), sinon de valeurs par défaut ;
  3. l'EXÉCUTION de chaque test, avec un état de progression poussé au fur et à mesure.

Familles
--------
`vitesse`      PP (prefill) et TG (génération) en t/s — prompt court + contexte saturé
               à 128k/256k. Mesure directe : `/completion` de llama.cpp (repli chat si le
               modèle n'est pas génératif en brut), timings officiels du serveur.
`memoire`      remplissage réel du contexte (92 %) + needle à 85 % de profondeur :
               rappel exact du code caché + PP réel à cette taille de contexte.
`intelligence` batterie de 13 petites tâches FR (`~/llm/small_tasks.py` sur .224) et
               HumanEval 50 / 164 problèmes (`~/llm-bench/run_he.py` sur .102 contre
               l'API du .224) : harnais **du parc**, donc mêmes formats d'artefacts que
               `bench_index.py` et scores comparables aux runs historiques.

Tout ce qui est mesuré est écrit dans le dossier de bench (`bench.json`, `bench.md`,
`<test>.log`) : un test interrompu laisse quand même ses chiffres.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request

# ---------------------------------------------------------------- valeurs par défaut
# Cadence supposée d'un LLM local **quand rien n'a encore été mesuré** sur ce modèle.
# Volontairement prudentes : mieux vaut une estimation haute qu'un créneau raté.
DEFAULT_PP_TPS = 450.0
DEFAULT_TG_TPS = 30.0

# Plafond généreux : un prefill de 256k peut prendre 10-20 min sur un modèle lent.
PREFILL_TIMEOUT_S = 3600

NEEDLE = "Note technique confidentielle : le code de vérification du projet est LUMIERE-777-42."
NEEDLE_TOKEN = "LUMIERE-777-42"
QUESTION = "Question : quel est le code de vérification du projet ? Réponds uniquement le code."

BLOCK = ("La mer océane couvre presque les trois quarts de la surface du globe. "
         "Elle régule le climat, transporte la chaleur des tropiques vers les pôles et "
         "abrite une diversité biologique encore largement inconnue. Les courants marins "
         "dessinent des routes invisibles que les navigateurs ont apprises à lire. "
         "Les sédiments conservent la mémoire des climats disparus. ")

# prompt du bench de vitesse courte (~490 tokens), identique à bench-llama-server.py
FILLER = ("reflexion sur le deploiement local de grands modeles de langage mixtes experts "
          "la memoire unifiee et la repartition des experts entre carte graphique et memoire vive "
          "un cache d experts chauds conserve les poids les plus utilises en VRAM tandis que les "
          "experts froids restent dans la memoire hote ce qui reduit considerablement les lectures "
          "disque et ameliore le debit de generation de tokens par seconde sur du materiel grand public "
          "la table de n grammes reste sur le disque solide et n est lue que quelques kilo octets par token")
SHORT_PROMPT = ("Explique de facon detaillee et technique le fonctionnement suivant.\n"
                + FILLER * 4 + "\n\nResume ce qui precede en trois phrases.")

# ---------------------------------------------------------------- catalogue

GROUPS = [
    {"id": "vitesse", "title": "Vitesse — prefill et génération (t/s)",
     "note": "appels directs à /completion, cache vidé : le chiffre comparable entre modèles"},
    {"id": "memoire", "title": "Mémoire de contexte — needle 128k / 256k",
     "note": "remplissage réel du contexte, fait caché à 85 % de profondeur"},
    {"id": "intelligence", "title": "Intelligence — rapide et complet",
     "note": "batterie FR notée par du code · HumanEval pass@1 (50 = contrôle, 164 = corpus)"},
]

TESTS = [
    # ---- vitesse -------------------------------------------------------
    {"id": "vitesse-court", "group": "vitesse", "kind": "vitesse",
     "title": "Vitesse courte — PP + TG",
     "desc": "3 tirs de 128 tokens sur un prompt de ~490 tokens, cache_prompt=false. "
             "Le chiffre de référence d'un modèle à l'autre (bench 489→128 tok).",
     "runs": 3, "n_predict": 128,
     "est": {"tin": 489, "tout": 128, "count": 3, "overhead": 8},
     "needs_ctx": 8192},

    {"id": "vitesse-128k", "group": "vitesse", "kind": "vitesse-long",
     "title": "Vitesse à 128k — prefill long + TG",
     "desc": "prefill d'un prompt de ~120k tokens puis 128 tokens générés : PP et TG "
             "sur contexte saturé (le PP chute quand le KV est plein, c'est ce qu'on mesure).",
     "ctx": 131072, "fill": 0.92, "n_predict": 128,
     "est": {"tin": 120586, "tout": 128, "count": 1, "overhead": 30},
     "needs_ctx": 131072},

    {"id": "vitesse-256k", "group": "vitesse", "kind": "vitesse-long",
     "title": "Vitesse à 256k — prefill long + TG",
     "desc": "idem avec ~249k tokens de remplissage : coût réel du prefill à la taille "
             "maximale du contexte (nécessite un serveur chargé à 262144).",
     "ctx": 262144, "fill": 0.95, "n_predict": 128,
     "est": {"tin": 249036, "tout": 128, "count": 1, "overhead": 40},
     "needs_ctx": 262144},

    # ---- mémoire de contexte -------------------------------------------
    {"id": "memoire-128k", "group": "memoire", "kind": "needle",
     "title": "Mémoire 128k — rappel à 85 %",
     "desc": "remplit 92 % de 131 072 tokens, place « LUMIERE-777-42 » à 85 % de profondeur "
             "et le redemande (temp 0) : rappel réel + PP à 120k tokens.",
     "ctx": 131072, "fill": 0.92, "depth": 0.85, "n_predict": 32,
     "est": {"tin": 120586, "tout": 32, "count": 1, "overhead": 25},
     "needs_ctx": 131072},

    {"id": "memoire-256k", "group": "memoire", "kind": "needle",
     "title": "Mémoire 256k — rappel à 85 %",
     "desc": "idem avec 241k tokens de remplissage : c'est là que la plupart des modèles "
             "décrochent (cliff de rappel observé au-delà de ~128-190k selon le quant).",
     "ctx": 262144, "fill": 0.92, "depth": 0.85, "n_predict": 32,
     "est": {"tin": 241172, "tout": 32, "count": 1, "overhead": 35},
     "needs_ctx": 262144},

    # ---- intelligence ---------------------------------------------------
    {"id": "batterie-13", "group": "intelligence", "kind": "batterie",
     "title": "Batterie rapide — 13 petites tâches FR",
     "desc": "13 tâches réelles (classement, extraction JSON, calcul, format, refus "
             "d'halluciner, retrieval dans ~7k tokens), notées par du code — pas par un humain.",
     "script": "small_tasks.py", "outdir": "smalltasks",
     "est": {"tin": 400, "tout": 250, "count": 13, "overhead": 20},
     "needs_ctx": 8192},

    {"id": "humaneval-50", "group": "intelligence", "kind": "humaneval",
     "title": "HumanEval rapide — problèmes 0-49",
     "desc": "50 problèmes Python exécutés (pass@1) depuis .102 contre l'API du .224. "
             "Contrôle qualité quasi gratuit (~1 min à 150 t/s) avant un run complet.",
     "problems": "0-49",
     "est": {"tin": 250, "tout": 900, "count": 50, "overhead": 15},
     "needs_ctx": 8192},

    {"id": "humaneval-164", "group": "intelligence", "kind": "humaneval",
     "title": "HumanEval complet — 164 problèmes",
     "desc": "le jeu complet : seul score comparable au corpus historique (76-98 %). "
             "Long — à lancer quand le serveur peut être monopolisé.",
     "problems": "0-163",
     "est": {"tin": 250, "tout": 900, "count": 164, "overhead": 30},
     "needs_ctx": 8192},
]

TESTS_BY_ID = {t["id"]: t for t in TESTS}


def test_ids():
    return [t["id"] for t in TESTS]


# ---------------------------------------------------------------- contexte d'exécution

class BenchEnv:
    """Tout ce qu'un test doit savoir de la machine — explicite, jamais lu dans l'environnement.

    `app.py` remplit cet objet au lancement du bench (identité du modèle figée AVANT le
    premier test, comme pour les duels) : les chiffres restent attribuables à un modèle.
    """

    def __init__(self, base, port, label, run_dir, local=True, dry=False, ssh="", he_dir="",
                 he_results="/home/lydstyl/llm-bench/results", llm_home="", log_src="/home/gab/llm",
                 nvidia=True, model_slug=None, model_path=None, n_ctx=None, python="python3"):
        self.base = (base or "http://127.0.0.1:8080").rstrip("/")
        self.port = int(port or 8080)
        self.local = bool(local)              # le LLM sert sur cette machine (scripts + VRAM OK)
        self.dry = bool(dry)
        self.label = label
        self.run_dir = run_dir
        self.ssh = ssh                        # cible ssh du harnais HumanEval ("" = désactivé)
        self.he_dir = he_dir
        self.he_results = he_results
        self.llm_home = llm_home or os.path.expanduser("~/llm")
        self.log_src = log_src
        self.nvidia = bool(nvidia) and self.local
        self.model_slug = model_slug
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.python = python


# ---------------------------------------------------------------- VRAM

def vram_now():
    """Une ligne « 0: 10440 MiB · 1: 13564 MiB » (None si nvidia-smi absent)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    parts = []
    for line in out.splitlines():
        f = [p.strip() for p in line.split(",")]
        if len(f) >= 3:
            try:
                parts.append(f"{int(f[0])}: {int(f[1])/1024:.1f}/{int(f[2])/1024:.1f} Go")
            except ValueError:
                continue
    return " · ".join(parts) or None


class VramSampler:
    """Échantillonne la VRAM (nvidia-smi, 0,5 s) pendant toute la durée d'un test.

    Le PIC est le seul chiffre qui compte : les buffers de calcul et le KV cache se
    matérialisent pendant le prefill, pas au repos. Best-effort — sans nvidia-smi,
    `text()` renvoie None au lieu d'échouer.
    """

    def __init__(self, enabled=True, interval=0.5):
        self.enabled = enabled
        self.interval = interval
        self.peaks = {}
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            for idx, used in _nvidia_rows():
                self.peaks[idx] = max(self.peaks.get(idx, 0), used)
            self._stop.wait(self.interval)

    def start(self):
        if not self.enabled:
            return
        if not _nvidia_rows():
            self.enabled = False
            return
        self._thread = threading.Thread(target=self._loop, name="bench-vram", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def text(self):
        if not self.peaks:
            return None
        return " · ".join(f"{g}: {v/1024:.1f} Go" for g, v in sorted(self.peaks.items()))


def _nvidia_rows():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        f = [p.strip() for p in line.split(",")]
        if len(f) >= 2:
            try:
                rows.append((int(f[0]), int(f[1])))
            except ValueError:
                pass
    return rows


# ---------------------------------------------------------------- calibration + estimation

def load_calib(path):
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_calib(path, calib):
    try:
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(calib, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def calib_key(model_slug, n_ctx):
    return f"{model_slug or '?'}@{n_ctx or 0}"


def update_calib(calib, key, fields, last_summary=None, test_id=None):
    """Enregistre les mesures d'un test : les estimations suivantes s'appuient dessus."""
    cur = calib.setdefault(key, {})
    for k in ("pp_tps", "tg_tps", "battery_s", "he_s_problem", "tasks"):
        if fields.get(k) is not None:
            cur[k] = fields[k]
    # PP mesuré PAR PALIER DE CONTEXTE : à 256k le PP réel est bien plus bas qu'à
    # 128k (899 vs 1203 t/s mesurés sur le 35B) — on garde la mesure du palier
    # plutôt qu'un facteur de dégradation générique.
    by = fields.get("pp_by_ctx") or {}
    if by:
        cur.setdefault("pp_by_ctx", {}).update({k: v for k, v in by.items() if v})
    cur["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if last_summary and test_id:
        per_test = cur.setdefault("tests", {})
        per_test[test_id] = {"summary": last_summary, "at": cur["updated"]}
    return cur


def fmt_dur(seconds):
    if seconds is None:
        return "—"
    seconds = int(round(seconds))
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


def estimate_seconds(test, entry=None):
    """Durée estimée (s) + provenance ('mesuré' si ce modèle a déjà été testé, sinon 'défaut')."""
    e = entry or {}
    pp = e.get("pp_tps") or DEFAULT_PP_TPS
    tg = e.get("tg_tps") or DEFAULT_TG_TPS
    source = "mesuré" if (e.get("pp_tps") and e.get("tg_tps")) else "défaut"
    est = test["est"]
    count = est.get("count", 1)
    overhead = est.get("overhead", 5)
    if test["kind"] == "humaneval":
        per = e.get("he_s_problem")
        if per:
            return per * count + overhead, "mesuré"
        return overhead + count * (est["tin"] / pp + est["tout"] / tg), source
    if test["kind"] == "batterie" and e.get("battery_s"):
        return float(e["battery_s"]) + 10, "mesuré"
    # à 256k le PP réel chute (KV plein, reprocessing) : on utilise le PP MESURÉ du
    # palier si on l'a, sinon on dégrade génériquement l'estimation.
    ctx = test.get("ctx")
    if ctx:
        measured = (e.get("pp_by_ctx") or {}).get(str(ctx))
        if measured:
            pp = measured
        elif ctx > 131072:
            pp *= 0.75
    return float(overhead) + count * est["tin"] / pp + count * est["tout"] / tg, source


def catalog(entry=None, availability=None):
    """Catalogue sérialisable pour l'UI : estimé, provenance, disponibilité, dernier chiffre."""
    entry = entry or {}
    last = entry.get("tests") or {}
    out = []
    for t in TESTS:
        est_s, est_src = estimate_seconds(t, entry)
        avail, reason = (True, None)
        if availability:
            avail, reason = availability(t)
        prev = last.get(t["id"]) or {}
        out.append({
            "id": t["id"], "group": t["group"], "title": t["title"], "desc": t["desc"],
            "needs_ctx": t.get("needs_ctx"),
            "est_s": round(est_s, 1),
            "est_label": fmt_dur(est_s),
            "est_source": est_src,
            "available": avail,
            "reason": reason,
            "last": prev.get("summary"),
            "last_at": prev.get("at"),
        })
    return out


# ---------------------------------------------------------------- HTTP (llama.cpp)

def _post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8", "replace"))
    return d, time.time() - t0


def _tokenize(base, text, timeout=900):
    d, _ = _post(f"{base}/tokenize", {"content": text}, timeout)
    return len(d.get("tokens") or [])


def _timings(d):
    t = d.get("timings") or {}
    if t:
        def r(v):
            try:
                return round(float(v), 1)
            except (TypeError, ValueError):
                return v
        return {"pp_tps": r(t.get("prompt_per_second")), "tg_tps": r(t.get("predicted_per_second")),
                "prompt_n": t.get("prompt_n"), "predicted_n": t.get("predicted_n")}
    u = d.get("usage") or {}
    return {"pp_tps": None, "tg_tps": None, "prompt_n": u.get("prompt_tokens"),
            "predicted_n": u.get("completion_tokens")}


def _completion(base, prompt, n_predict, timeout, emit=None, prefer_chat=False):
    """Un tir de vitesse : /completion (repli CHAT) ou directement CHAT.

    ⚠️ Le repli coûte un PREFILL COMPLET : sur un prompt de 245k tokens, un tir brut
    qui rend EOS au 1er token fait payer deux fois le prefill (constaté : 552 s au
    lieu de 280 s à 256k). D'où `prefer_chat=True` pour les tirs à contexte saturé :
    on va droit au endpoint chat, qui est celui qui génère réellement.
    Un `predicted_n` < 8 en brut n'est PAS « le modèle est lent », c'est le harnais
    qui a fini (modèle chat-tuné sans marqueurs de template dans le prompt brut).
    """
    if prefer_chat:
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": n_predict,
                "temperature": 0.0, "top_k": 1, "seed": 1, "cache_prompt": False,
                "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
        d, wall = _post(f"{base}/v1/chat/completions", body, timeout)
        t = _timings(d)
        t["wall_s"] = round(wall, 2)
        t["mode"] = "chat"
        return t
    d, wall = _post(f"{base}/completion", {"prompt": prompt, "n_predict": n_predict,
                                          "temperature": 0.0, "top_k": 1, "seed": 1,
                                          "cache_prompt": False}, timeout)
    t = _timings(d)
    mode = "brut"
    if (t.get("predicted_n") or 0) < 8:
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": n_predict,
                "temperature": 0.0, "top_k": 1, "seed": 1, "cache_prompt": False,
                "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
        d, wall = _post(f"{base}/v1/chat/completions", body, timeout)
        t = _timings(d)
        mode = "chat"
        if emit:
            emit(f"  tir brut non génératif → repli chat (le prefill est payé deux fois)")
    t["wall_s"] = round(wall, 2)
    t["mode"] = mode
    return t


def _fill_prompt(base, target_tokens, emit=None):
    """Construit un texte d'environ `target_tokens` tokens (comptés par /tokenize)."""
    nblock = max(1, _tokenize(base, BLOCK))
    reps = max(1, target_tokens // nblock)
    text = BLOCK * reps
    got = _tokenize(base, text)
    guard = 0
    while got > target_tokens and guard < 40:
        reps -= max(1, (got - target_tokens) // nblock)
        reps = max(1, reps)
        text = BLOCK * reps
        new = _tokenize(base, text)
        if new == got:            # plus rien à couper proprement
            break
        got = new
        guard += 1
    if emit:
        emit(f"  remplissage : {got} tokens (cible {target_tokens})")
    return text, got


# ---------------------------------------------------------------- exécutions

def _run_vitesse(test, env, emit, stop_check):
    results = []
    for i in range(test["runs"]):
        if stop_check():
            raise RuntimeError("arrêt demandé")
        t = _completion(env.base, SHORT_PROMPT, test["n_predict"], PREFILL_TIMEOUT_S, emit)
        results.append(t)
        emit(f"  tir {i + 1}/{test['runs']} : PP {t.get('pp_tps') or 0:.1f} t/s "
             f"({t.get('prompt_n')} tok) | TG {t.get('tg_tps') or 0:.1f} t/s "
             f"({t.get('predicted_n')} tok) | {t['wall_s']}s")
    pp = [t["pp_tps"] for t in results if t.get("pp_tps")]
    tg = [t["tg_tps"] for t in results if t.get("tg_tps") and (t.get("predicted_n") or 0) >= 8]
    m_pp = round(sum(pp) / len(pp), 1) if pp else None
    m_tg = round(sum(tg) / len(tg), 1) if tg else None
    return {
        "metrics": {"pp_tps": m_pp, "tg_tps": m_tg, "runs": results},
        "calib": {"pp_tps": m_pp, "tg_tps": m_tg},
        "ok": bool(m_pp and m_tg),
        "summary": f"PP {m_pp or '—'} t/s · TG {m_tg or '—'} t/s ({len(tg)} tir(s) valide(s))",
    }


def _run_vitesse_long(test, env, emit, stop_check):
    target = int(test["ctx"] * test.get("fill", 0.92))
    text, got = _fill_prompt(env.base, target, emit)
    if stop_check():
        raise RuntimeError("arrêt demandé")
    t = _completion(env.base, text, test["n_predict"], PREFILL_TIMEOUT_S, emit, prefer_chat=True)
    emit(f"  PP {t.get('pp_tps') or 0:.1f} t/s sur {t.get('prompt_n')} tokens | "
         f"TG {t.get('tg_tps') or 0:.1f} t/s | {t['wall_s']}s")
    return {
        "metrics": {"pp_tps": t.get("pp_tps"), "tg_tps": t.get("tg_tps"),
                    "prompt_n": t.get("prompt_n"), "completion_n": t.get("predicted_n"),
                    "mode": t.get("mode"), "wall_s": t["wall_s"]},
        "calib": {"pp_tps": t.get("pp_tps"), "tg_tps": t.get("tg_tps"),
                  "pp_by_ctx": {str(test["ctx"]): t.get("pp_tps")}},
        "ok": bool(t.get("pp_tps")),
        "summary": (f"PP {t.get('pp_tps') or '—'} t/s sur {got} tok · "
                    f"TG {t.get('tg_tps') or '—'} t/s"),
    }


def _run_needle(test, env, emit, stop_check):
    target = int(test["ctx"] * test.get("fill", 0.92))
    text, got = _fill_prompt(env.base, target, emit)
    depth = test.get("depth", 0.85)
    pos = int(len(text) * depth)
    cut = text.find(" ", pos)
    cut = pos if cut < 0 else cut
    text = text[:cut] + "\n" + NEEDLE + "\n" + text[cut:]
    got = _tokenize(env.base, text)
    if stop_check():
        raise RuntimeError("arrêt demandé")
    emit(f"  needle à {int(depth * 100)} % — prompt final {got} tokens")
    body = {"messages": [{"role": "user", "content": text + "\n\n" + QUESTION}],
            "max_tokens": test.get("n_predict", 32), "temperature": 0.0,
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    d, wall = _post(f"{env.base}/v1/chat/completions", body, PREFILL_TIMEOUT_S)
    msg = ((((d.get("choices") or [{}])[0]).get("message") or {}).get("content") or "")
    t = _timings(d)
    needle_ok = NEEDLE_TOKEN in msg.upper().replace(" ", "-")
    emit(f"  rappel : {'OK' if needle_ok else 'ÉCHEC'} — réponse {msg.strip()[:60]!r}")
    return {
        "metrics": {"needle_ok": needle_ok, "answer": msg.strip()[:200],
                    "prompt_n": t.get("prompt_n") or got, "pp_tps": t.get("pp_tps"),
                    "tg_tps": t.get("tg_tps"), "wall_s": round(wall, 1)},
        "calib": {"pp_tps": t.get("pp_tps"),
                  "pp_by_ctx": {str(test["ctx"]): t.get("pp_tps")}},
        "ok": bool(needle_ok),
        "summary": (f"rappel {'OK' if needle_ok else 'ÉCHEC'} · "
                    f"PP {t.get('pp_tps') or '—'} t/s · {round(wall)} s"),
    }


def _run_batterie(test, env, emit, stop_check):
    script = os.path.join(env.llm_home, test["script"])
    if not os.path.exists(script):
        raise RuntimeError(f"harnais absent : {script}")
    outdir = os.path.join(env.llm_home, test.get("outdir") or "smalltasks")
    cmd = [env.python, script, str(env.port), env.label, outdir, "nothink"]
    emit(f"  batterie : {' '.join(cmd)}")
    rc, _ = _spawn_stream(cmd, emit, stop_check, cwd=env.llm_home)
    json_path = os.path.join(outdir, f"results_{env.label}_nothink.json")
    metrics = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, encoding="utf-8") as fh:
                d = json.load(fh)
            s = d.get("summary") or {}
            metrics = {"score_sum": s.get("score_sum"), "score_full": s.get("score_full"),
                       "tasks": s.get("tasks"), "score_avg": s.get("score_avg"),
                       "lat_median": s.get("lat_median"), "lat_max": s.get("lat_max"),
                       "battery_s": s.get("total_wall_s"), "fails": s.get("fails"),
                       "source": json_path}
            try:
                shutil.copyfile(json_path, os.path.join(env.run_dir, "batterie-13.json"))
            except OSError:
                pass
        except (OSError, ValueError) as exc:
            emit(f"  résumé batterie illisible : {exc}")
    if not metrics:
        return {"metrics": {"rc": rc}, "ok": False, "calib": {},
                "summary": "aucun résumé de batterie produit"}
    return {
        "metrics": metrics,
        "calib": {"battery_s": metrics.get("battery_s")},
        "ok": bool(rc == 0 and metrics.get("score_sum") is not None),
        "summary": (f"score {metrics.get('score_sum')}/{metrics.get('tasks')} "
                    f"· complets {metrics.get('score_full')}/{metrics.get('tasks')} "
                    f"· latence méd {metrics.get('lat_median')} s"),
    }


def _run_humaneval(test, env, emit, stop_check):
    if not env.ssh or not env.he_dir:
        raise RuntimeError("cible ssh du harnais HumanEval non configurée")
    problems = test["problems"]
    out = f"{env.he_results}/from224-{env.label}-{problems.replace(',', '_')}"
    display = f"{env.model_slug or '?'} [{int(env.n_ctx or 0) // 1024}k]"
    remote = ("cd %s && python3 run_he.py %s %s %s %s && cat %s/humaneval-summary.json"
              % (_q(env.he_dir), _q(display), _q(env.model_path or "local"), _q(problems),
                 _q(out), _q(out)))
    emit(f"  HumanEval {problems} via {env.ssh} — modèle « {display} »")
    rc, txt = _spawn_stream(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                             env.ssh, remote], emit, stop_check)
    summary = None
    # le résumé est imprimé par `cat` après le rapport : on garde le DERNIER objet JSON
    for m in re.finditer(r'\{\s*"timestamp".*?\n\}', txt, re.S):
        try:
            summary = json.loads(m.group(0))
        except ValueError:
            continue
    metrics = {"rc": rc, "problems": problems, "remote_dir": out, "display_name": display}
    ok = False
    if summary:
        machines = summary.get("machines") or {}
        m = next(iter(machines.values()), {}) if machines else {}
        metrics.update({"passed": m.get("passed"), "total": m.get("total"),
                        "score_pct": m.get("score_pct"),
                        "avg_latency_s": m.get("avg_latency_s"),
                        "wall_s": m.get("wall_time_s") or summary.get("wall_time_s"),
                        "avg_tokens_per_sec": m.get("avg_tokens_per_sec")})
        ok = rc == 0 and m.get("score_pct") is not None
    return {
        "metrics": metrics,
        "calib": ({"he_s_problem": round(metrics["avg_latency_s"], 2)}
                  if metrics.get("avg_latency_s") else {}),
        "ok": ok,
        "summary": (f"pass@1 {metrics.get('passed')}/{metrics.get('total')} "
                    f"({metrics.get('score_pct')} %)" if ok else
                    (f"run interrompu (rc={rc})" if rc else "rapport HumanEval introuvable")),
    }


def _spawn_stream(cmd, emit, stop_check, cwd=None):
    """Exécute une commande en poussant sa sortie ligne à ligne ; tuable par le Stop.

    Renvoie (code_retour, sortie_complète). Le process est lancé dans son propre groupe
    (`start_new_session`) pour qu'un Stop tue aussi ses enfants (ssh, python harnais).
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=cwd, start_new_session=True)
    lines = []
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            emit(line)
            if stop_check():
                raise RuntimeError("arrêt demandé")
        return proc.wait(), "\n".join(lines)
    except RuntimeError:
        _kill_tree(proc)
        raise
    finally:
        if proc.poll() is None:
            _kill_tree(proc, sig=9)


def _kill_tree(proc, sig=15):
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _q(s):
    return "'" + str(s).replace("'", "'\\''") + "'"


RUNNERS = {
    "vitesse": _run_vitesse,
    "vitesse-long": _run_vitesse_long,
    "needle": _run_needle,
    "batterie": _run_batterie,
    "humaneval": _run_humaneval,
}


def _run_dry(test, env, emit, stop_check):
    """DRY_RUN : aucun appel réseau — chiffres factices, pour tester l'UI sans llama.cpp."""
    time.sleep(0.4)
    emit("  DRY RUN : aucun appel à llama.cpp, chiffres factices")
    kind = test["kind"]
    if kind == "vitesse":
        return {"metrics": {"pp_tps": 612.0, "tg_tps": 46.0, "runs": []}, "calib": {},
                "ok": True, "summary": "PP 612 t/s · TG 46 t/s (dry)"}
    if kind == "vitesse-long":
        return {"metrics": {"pp_tps": 505.0, "tg_tps": 41.0}, "calib": {},
                "ok": True, "summary": "PP 505 t/s · TG 41 t/s (dry)"}
    if kind == "needle":
        return {"metrics": {"needle_ok": True, "answer": NEEDLE_TOKEN}, "calib": {},
                "ok": True, "summary": "rappel OK · PP 420 t/s (dry)"}
    if kind == "batterie":
        return {"metrics": {"score_sum": 11.5, "tasks": 13, "score_full": 10,
                            "lat_median": 0.5, "battery_s": 12.0}, "calib": {},
                "ok": True, "summary": "score 11.5/13 (dry)"}
    return {"metrics": {"passed": 40, "total": 50, "score_pct": 80.0, "avg_latency_s": 1.0},
            "calib": {}, "ok": True, "summary": "pass@1 40/50 (80.0 %) (dry)"}


def run_test(test, env, emit, stop_check):
    """Exécute UN test et renvoie son résultat — l'erreur est un STATUT, pas une exception."""
    t0 = time.time()
    res = {"id": test["id"], "group": test["group"], "title": test["title"],
           "status": "running", "metrics": {}, "summary": None, "erreur": None,
           "duree_s": None, "calib": {}, "log": f"{test['id']}.log", "vram_peak": None}
    emit(f"=== {test['title']} ===")
    buf = []

    def emit_log(line):
        buf.append(line)
        emit(line)

    runner = _run_dry if env.dry else RUNNERS[test["kind"]]
    sampler = VramSampler(enabled=env.nvidia)
    sampler.start()
    try:
        out = runner(test, env, emit_log, stop_check)
        res["metrics"] = out.get("metrics") or {}
        res["summary"] = out.get("summary")
        res["calib"] = out.get("calib") or {}
        res["status"] = "ok" if out.get("ok") else "partial"
    except Exception as exc:
        res["status"] = "error"
        res["erreur"] = f"{exc.__class__.__name__}: {exc}"
        emit_log(f"ERREUR {res['erreur']}")
    finally:
        sampler.stop()
    res["duree_s"] = round(time.time() - t0, 1)
    res["vram_peak"] = sampler.text()
    try:
        os.makedirs(env.run_dir, exist_ok=True)
        with open(os.path.join(env.run_dir, res["log"]), "w", encoding="utf-8") as fh:
            fh.write("\n".join(buf) + "\n")
    except OSError:
        pass
    emit(f"→ {test['id']} : {res['status']} en {res['duree_s']}s — {res['summary'] or res['erreur']}")
    return res


# ---------------------------------------------------------------- rapport

def render_md(bench):
    """Rapport lisible du bench (même logique que les autres rapports du parc : md + json)."""
    lines = [f"# Tests prompt-duel — {bench.get('run_id')}", ""]
    lines.append(f"- modèle : `{bench.get('model_path')}`")
    lines.append(f"- slug / ctx : `{bench.get('model_slug')}` · {bench.get('n_ctx')}")
    lines.append(f"- endpoint : {bench.get('endpoint')} · dry_run={bench.get('dry_run')}")
    lines.append(f"- début / fin : {bench.get('started_at')} → {bench.get('finished_at')}")
    if bench.get("vram_note"):
        lines.append(f"- VRAM au lancement : {bench['vram_note']}")
    if bench.get("notes"):
        lines.append(f"- note : {bench['notes']}")
    lines.append("")
    lines.append("| test | statut | estimé | réel | chiffres |")
    lines.append("|---|---|---|---|---|")
    for r in bench.get("results") or []:
        lines.append("| %s | %s | %s | %s | %s |" % (
            r.get("title"), r.get("status"), fmt_dur(r.get("est_s")), fmt_dur(r.get("duree_s")),
            r.get("summary") or r.get("erreur") or "—"))
    lines.append("")
    for r in bench.get("results") or []:
        m = r.get("metrics") or {}
        if not m:
            continue
        lines.append(f"## {r.get('title')}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(m, ensure_ascii=False, indent=1)[:4000])
        lines.append("```")
        lines.append("")
    lines.append("_Généré par prompt-duel (bench_tests.py) — chiffres mesurés sur ce serveur, "
                 "aucune valeur recopiée d'une fiche modèle._")
    return "\n".join(lines) + "\n"
