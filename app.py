#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Duel 224 — lanceur de prompts one-shot vers un LLM local (llama.cpp).

Envoie 1 à N prompts choisis, un par un (1 seul slot llama.cpp), et écrit pour
chaque prompt un fichier HTML autonome, rangé dans un dossier de run dont le nom
porte l'identité du modèle réellement servi (slug + contexte + réglages).

v1.2 : les tirs se font en **streaming SSE** — le texte produit par le LLM est
poussé dans un tampon mémoire (LiveStore) que l'UI lit en incrémental via
`/api/live`, ce qui donne un vrai suivi en direct (texte, raisonnement, vitesse,
aperçu HTML). Couper le streaming : `LLM_STREAM=0` (retour au tir en un bloc).

v1.3 : **réglages par run** (température, max_tokens, raisonnement choisis dans
l'UI — le nom du dossier de run les enregistre) et **comparaison de deux runs**
(`/api/compare`, page `/compare` avec les deux rendus HTML côte à côte) pour
opposer deux modèles ou deux réglages prompt par prompt.

v1.4.1 : envoi de `reasoning_effort` **au premier niveau** du body en plus de
`chat_template_kwargs.enable_thinking` — certains modèles (Ternary Bonsai 2 de
PrismML) ignorent ce dernier et raisonnent par défaut : sans ce champ, un run
« sans raisonnement » est en réalité un run « avec raisonnement ». Valeurs :
`none` (run demandé sans raisonnement) / `medium` (avec). Les runtimes qui ne
connaissent pas ce champ l'ignorent → rétrocompatible (Qwen3.6/3.8, Swift).

v1.5.0 : **tests cochables dans l'app** — vitesse (prefill/génération en t/s), mémoire
de contexte (needle à 128k et 256k), intelligence (batterie de 13 tâches FR, HumanEval
50/164), avec **estimation de la durée avant lancement** (mesures du modèle si connues,
sinon valeurs par défaut) et résultats écrits dans `benches/<run>/` (+ `bench.json`,
`bench.md`, `<test>.log`). Module `bench_tests.py`.

v1.6.0 : **réglages et variantes** — l'app détecte les réglages RÉELS du serveur
llama.cpp (cmdline du process + `/props` : ctx, MTP/draft, flash-attn, flags) et les
enregistre dans chaque run et chaque bench (`settings` + `variant`) ; un run « MTP on »
n'est jamais réutilisé pour un « MTP off ». Page **/graph** (SVG inline, sans CDN ni
JavaScript obligatoire) : un graphique par métrique chiffrée + les durées, filtres par
test et par variante. **Masquer / supprimer / restaurer** un résultat depuis /graph et
/benchmarks (`visibility.json`, état local non versionné). Modules neufs : `metrics.py`,
`visibility.py`, `settings.py`, `charts.py`, `page_graph.py` ; harnais de tests
`scripts/run-tests.sh` (unitaires `tests/` + end-to-end `e2e/` sous Playwright).

v1.6.3 : **/graph en deux étapes** — la page s'ouvre sur le CHOIX, plus sur un mur de
graphiques. Étape 1 : le catalogue des réglages (une case par variante, avec son nombre
de points et la date du dernier résultat) et la règle écrite en clair « coche de 2 à 6,
puis Comparer » — aucun graphique. Étape 2 (`/graph?sel=<clé>&sel=<clé>`, formulaire GET,
aucun JavaScript) : uniquement les courbes des réglages cochés, un lien « ← modifier la
sélection » qui revient au choix (cases conservées), puis le panneau « affiner »
(familles, partiels, masqués) et le tableau des résultats. Bornes explicites : moins de
2 réglages → retour au choix avec la raison (« avec un seul, il n'y a rien à comparer ») ;
plus de 6 → les 6 premiers sont tracés, les autres nommés dans un bandeau (lisibilité de
la légende) ; clé inconnue → jamais appliquée, mais signalée. À l'étape 1, le reste de la
page (résultats, filtres, masquage réversible) descend dans un dépliant natif, donc reste
accessible sans JavaScript. Nouveau `metrics.series_catalogue`, constantes
`MIN_SELECTION`/`MAX_SELECTION`, alias d'URL `sel`, `selection`, `series`, `serie`, `llm`,
`reglages`.

v1.6.2 : **mesures honnêtes dans l'index** — trois chiffres faux disparaissent de
`/graph` et `/benchmarks`. (1) Vitesse : un log de cas contient plusieurs passes (passe
brute avortée, repli chat) ; les moyenner divisait le débit par deux (48,9 t/s publiés
au lieu de 97,7) — seules les répétitions complètes du banc sont appariées désormais.
(2) VRAM : le sampler nvidia-smi continue d'écrire après l'arrêt du cas ; le pic est
borné à la fenêtre `start … stop` des jalons de phase (12 118 MiB publiés au lieu de
8 940), l'écart hors fenêtre reste dans la note. (3) Duels : un duel qui a perdu un
tiers de ses prompts est marqué `partiel` — il reste dans l'index mais sort des courbes
(case « tracer aussi les résultats partiels », `?partiels=1` pour `/api/graph`), et les
duels rangés dans `echecs/` ne sont plus remiroités. Côté serveur, `wait_for_free_slot`
ne perd plus un prompt quand `/slots` manque (serveur OpenAI-only) et réessaie 3 fois
quand le moteur est injoignable.

v1.6.1 : **bancs vitesse/mémoire sur un serveur sans routes natives** — Strata et les
moteurs OpenAI-only ne servent ni `/tokenize` ni `/completion` : les tests de vitesse et
de mémoire tombaient en erreur 404 en 0,1 s. `bench_tests.py` sonde une fois par serveur,
bascule sur `/v1/chat/completions` (mêmes champs `timings` : `prompt_per_second`,
`predicted_per_second`), estime le remplissage par un ratio caractères/token mesuré côté
serveur SUR LE TEXTE DE REMPLISSAGE lui-même puis lit le compte exact dans `prompt_n`.
Trois justesses de mesure au passage :
le remplissage se corrige dans les deux sens (un test « 128k » ne se publie pas à 113k) ;
un tir servi par le cache du serveur (`cache_n`) est écarté de la moyenne de prefill au
lieu de la tirer vers le bas ; le needle envoie `cache_prompt: false`, lit aussi
`reasoning_content` (avec 128 tokens de marge) et publie le contexte RÉELLEMENT servi
(évalué + cache). Un modèle qui raisonne brûlait sinon son budget avant d'écrire, et un
rappel juste était compté comme un échec.

Python 3 stdlib UNIQUEMENT — aucune dépendance externe.

Lancement :
    cd ~/apps/prompt-duel
    python3 app.py                      # bind 0.0.0.0:8791
    PORT=8899 python3 app.py            # port alternatif
    LLM_DRY_RUN=1 python3 app.py        # sans llama.cpp (tests / démo)
    LLM_STREAM=0 python3 app.py         # pas de streaming (tir en un bloc)

UI : http://<ip-de-la-machine>:8791
"""

import html
import errno
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import bench_tests  # tests vitesse / mémoire de contexte / intelligence (v1.5)
from benchmarks import render_benchmarks_page  # page /benchmarks (index des tests)

# v1.6 : la logique descend dans des modules (contrat §5) ; app.py reste le routeur.
import metrics      # registre des métriques, variantes, séries du graphique
import settings     # réglages réellement servis par le serveur llama.cpp
import visibility   # résultats masqués / supprimés (tombstones) + filtres

try:
    import page_graph      # page /graph, écrite par un second lot (v1.6)
except ImportError:        # module pas encore livré : /graph répondra 503, jamais de crash
    page_graph = None

APP_VERSION = "1.6.3"

# Libellé affiché dans l'UI (utile si plusieurs instances)
APP_TITLE = os.environ.get("APP_TITLE") or "Prompt Duel"

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
# RUNS_DIR peut être déplacé (chemin absolu) : pratique pour lancer une instance
# de test à côté de l'instance de prod sans toucher à ses runs.
RUNS_DIR = Path(os.environ.get("RUNS_DIR") or (BASE_DIR / "runs"))
RUNS_LATEST = RUNS_DIR / "latest"
PROMPTS_FILE = BASE_DIR / "prompts.json"

# --- tests (v1.5) ----------------------------------------------------------
# Un dossier de bench par session de tests, à côté des runs de duel.
BENCH_DIR = Path(os.environ.get("BENCH_DIR") or (BASE_DIR / "benches"))
BENCH_CALIB_FILE = BENCH_DIR / "bench_calib.json"
# Harnais HumanEval : lancé depuis le .102 (le serveur qui héberge l'app est le .224).
BENCH_HE_SSH = os.environ.get("BENCH_HE_SSH", "lydstyl@192.168.3.102")
BENCH_HE_DIR = os.environ.get("BENCH_HE_DIR", "/home/lydstyl/llm-bench")
BENCH_HE_RESULTS = os.environ.get("BENCH_HE_RESULTS", "/home/lydstyl/llm-bench/results")
# Harnais locaux du .224 (batterie de 13 tâches, bench de vitesse du parc).
BENCH_LLM_HOME = os.environ.get("BENCH_LLM_HOME", "/home/gab/llm")
BENCH_LOG_SRC = os.environ.get("BENCH_LOG_SRC", BENCH_LLM_HOME)

# --- index unifié des tests (v1.6) -----------------------------------------
# Mêmes sources que benchmarks.py : BENCH_INDEX (tests/e2e), le vault (source de
# vérité), puis la copie locale. LECTURE SEULE : jamais écrit par l'app.
BENCH_INDEX_CANDIDATES = [
    os.environ.get("BENCH_INDEX") or "",
    "/home/gab/NAS/AgentsMirror/vaults/personnel/documents/llm-benchmarks/index.json",
    str(BASE_DIR / "bench_index.json"),
]

HOST = os.environ.get("HOST", "0.0.0.0")
try:
    PORT = int(os.environ.get("PORT") or 8791)
except ValueError:
    PORT = 8791

LLM_BASE = (os.environ.get("LLM_BASE") or "http://127.0.0.1:8080").rstrip("/")

DRY_RUN = (os.environ.get("LLM_DRY_RUN") or "").strip().lower() not in ("", "0", "false", "no", "off")
DRY_N_CTX = 8192
DRY_DELAY_S = 3.0

SYSTEM_PROMPT = (
    "Tu génères un unique fichier HTML autonome et complet. "
    "Réponds UNIQUEMENT avec le code HTML, sans explication et sans balise markdown."
)

def _env_int(name, default):
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


TEMPERATURE = 0.2
TOP_P = 0.95
# v1.1 : 16384 — les prompts 3D/jeu produisent 8k→15k tokens (8192 tronquait le HTML).
# Surchargeable globalement par l'env MAX_TOKENS, ou prompt par prompt via le champ
# optionnel "max_tokens" de prompts.json.
MAX_TOKENS = _env_int("MAX_TOKENS", 16384)
ENABLE_THINKING = False          # OBLIGATOIRE : sinon la réponse part dans reasoning_content
CACHE_PROMPT = False             # interdit la réutilisation du cache KV

# v1.3 : ces réglages ne sont plus figés — l'UI peut les choisir pour chaque run
# (température, max_tokens, raisonnement). Ils restent les valeurs PAR DÉFAUT.
DEFAULT_PARAMS = {
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "max_tokens": MAX_TOKENS,
    "enable_thinking": ENABLE_THINKING,
    "cache_prompt": CACHE_PROMPT,
}
PARAMS_MIN_TOKENS = 64
PARAMS_MAX_TOKENS = 400000

REQUEST_TIMEOUT_S = 1800         # un tir peut être très long
SLOT_POLL_S = 5                  # poll de /slots en attente de slot libre
HEALTH_TIMEOUT_S = 3
PROBE_CACHE_S = 3.0              # l'UI polle toutes les 1 s : on lisse les sondes
LOG_MAX = 2000
RUNS_LIST_LIMIT = 30
RUNS_COMPARE_LIMIT = 500     # runs listés pour les sélecteurs de comparaison (/api/runs)
RUN_REUSE_S = 1800               # idempotence : on reprend un run identique récent

# --- suivi en direct (v1.2) -------------------------------------------------
# Le tir se fait en streaming SSE et chaque morceau est poussé dans LIVE.
# LLM_STREAM=0 revient au comportement v1.1 (un seul bloc en fin de tir).
STREAM = (os.environ.get("LLM_STREAM") or "1").strip().lower() not in ("", "0", "false", "no", "off")
LIVE_MAX_CHARS = _env_int("LIVE_MAX_CHARS", 400000)   # borne mémoire du tampon live
LIVE_POLL_MS = _env_int("LIVE_POLL_MS", 800)          # cadence de poll de l'UI

TERMINAL_STATUSES = ("ok", "no_html", "error", "deja_genere", "skipped")

# --------------------------------------------------------------------------
# Utilitaires
# --------------------------------------------------------------------------

_ACCENTS = str.maketrans(
    "àâäáãåçéèêëíìîïñóòôöõúùûüýÿÀÂÄÁÃÅÇÉÈÊËÍÌÎÏÑÓÒÔÖÕÚÙÛÜÝ",
    "aaaaaaceeeeiiiinooooouuuuyyAAAAAACEEEEIIIINOOOOOUUUUY",
)


def slugify(value):
    """minuscules, [a-z0-9._-], le reste -> '-', sans accent ni espace."""
    s = str(value or "").translate(_ACCENTS).lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-._")
    return s or "x"


def now_iso():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def normalize_params(raw=None):
    """Réglages d'un tir, validés (accepte du partiel, des chaînes, du JSON d'UI).

    Toujours un dict complet : temperature, top_p, max_tokens, enable_thinking,
    cache_prompt. Une valeur hors bornes est ignorée (on garde le défaut) : l'app
    ne doit jamais partir en vrille à cause d'un champ de formulaire.
    """
    p = dict(DEFAULT_PARAMS)
    raw = raw if isinstance(raw, dict) else {}
    try:
        t = float(raw.get("temperature", p["temperature"]))
        if 0.0 <= t <= 2.0:
            p["temperature"] = round(t, 3)
    except (TypeError, ValueError):
        pass
    try:
        mt = int(raw.get("max_tokens", p["max_tokens"]))
        if PARAMS_MIN_TOKENS <= mt <= PARAMS_MAX_TOKENS:
            p["max_tokens"] = mt
    except (TypeError, ValueError):
        pass
    think = raw.get("enable_thinking", p["enable_thinking"])
    if isinstance(think, str):
        think = think.strip().lower() not in ("", "0", "false", "no", "off")
    p["enable_thinking"] = bool(think)
    return p


def params_suffix(params):
    """Bouts de nom de dossier qui identifient les réglages NON standard.

    Le max_tokens par défaut n'apparaît pas : les dossiers de runs déjà produits
    (v1.1/v1.2) gardent donc exactement le même nom, et restent réutilisables.
    """
    bits = []
    if params.get("max_tokens") != MAX_TOKENS:
        bits.append(f"mt{params['max_tokens']}")
    return ("__" + "__".join(bits)) if bits else ""


def params_label(params):
    """Résumé court et lisible des réglages : « t0.2 · nothink · mt16384 »."""
    params = normalize_params(params)
    return (f"t{params['temperature']:g} · "
            f"{'think' if params['enable_thinking'] else 'nothink'} · "
            f"mt{params['max_tokens']}")


def model_slug_from_path(path):
    name = os.path.basename(str(path or "").strip())
    if name.lower().endswith(".gguf"):
        name = name[:-5]
    return slugify(name)


def extract_html(text):
    """Retire les fences markdown et renvoie de <!DOCTYPE html|<html à </html>."""
    if not text:
        return None
    t = text.strip()
    m = re.search(r"<!DOCTYPE html", t, re.I)
    if not m:
        m = re.search(r"<html[\s>]", t, re.I)
    if not m:
        return None
    start = m.start()
    end = t.lower().rfind("</html>")
    if end == -1 or end < start:
        return None
    out = t[start:end + len("</html>")]
    # fences résiduelles éventuelles à l'intérieur du bloc extrait
    out = re.sub(r"(?m)^\s*```[a-zA-Z0-9]*\s*$", "", out).strip()
    return out or None


class LogStore:
    """Journal borné, indexé de façon absolue (polling `since`)."""

    def __init__(self, maxlen=LOG_MAX):
        self._lock = threading.Lock()
        self._lines = []
        self._next = 0
        self._maxlen = maxlen

    def add(self, message):
        stamp = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            idx = self._next
            self._next += 1
            self._lines.append({"i": idx, "t": stamp, "msg": str(message)})
            if len(self._lines) > self._maxlen:
                del self._lines[:len(self._lines) - self._maxlen]
        print(f"[{stamp}] {message}", flush=True)
        return idx

    def since(self, n):
        with self._lock:
            return [dict(l) for l in self._lines if l["i"] >= n], self._next


LOG = LogStore()


class LiveStore:
    """Tampon du tir en cours : texte + raisonnement, lus en incrémental.

    Le worker pousse chaque morceau via append() ; l'UI lit /api/live avec deux
    curseurs (en caractères) et ne reçoit que ce qu'elle n'a pas déjà vu.
    Le tampon est borné (LIVE_MAX_CHARS) : au-delà on jette le début et on le
    signale (`truncated` + `text_reset`) pour que l'UI reparte du bon pied.
    """

    def __init__(self, cap=LIVE_MAX_CHARS):
        self._lock = threading.Lock()
        self._cap = max(10000, int(cap))
        self._clear_locked()

    def _clear_locked(self):
        self.run_id = None
        self.pid = None
        self.title = None
        self.slug = None
        self.mode = None            # "stream" | "bloc" | "dry"
        self.status = "idle"        # idle | waiting | running | statut final
        self.text = ""
        self.reasoning = ""
        self.dropped = 0            # caractères jetés en tête de text
        self.rdropped = 0
        self.chunks = 0             # morceaux reçus (= tokens, en streaming)
        self.truncated = False
        self.started_at = None
        self.finished_at = None
        self.duree_s = None
        self.completion_tokens = None
        self.tok_s = None
        self.finish_reason = None
        self.erreur = None
        self.max_tokens = None
        self.params = normalize_params()
        self._t0 = None
        self.updated_at = None

    # -- écriture (worker) -------------------------------------------------
    def begin(self, run_id, pid, title, slug, mode="stream", max_tokens=None, params=None):
        with self._lock:
            self._clear_locked()
            self.run_id = run_id
            self.pid, self.title, self.slug = pid, title, slug
            self.mode, self.max_tokens = mode, max_tokens
            self.params = normalize_params(params)
            self.status = "running"
            self.started_at = now_iso()
            self._t0 = time.time()
            self.updated_at = self._t0

    def set_status(self, status):
        with self._lock:
            if self.status == "idle":
                return
            self.status = status
            self.updated_at = time.time()

    def append(self, kind, piece):
        """Callback appelé à chaque morceau : append("content"|"reasoning", texte)."""
        if not piece:
            return
        with self._lock:
            if kind == "reasoning":
                self.reasoning += piece
                if len(self.reasoning) > self._cap:
                    cut = len(self.reasoning) - self._cap
                    self.reasoning = self.reasoning[cut:]
                    self.rdropped += cut
            else:
                self.text += piece
                if len(self.text) > self._cap:
                    cut = len(self.text) - self._cap
                    self.text = self.text[cut:]
                    self.dropped += cut
                    self.truncated = True
            self.chunks += 1
            self.updated_at = time.time()

    def finish(self, status, duree_s=None, completion_tokens=None, tok_s=None,
               finish_reason=None, erreur=None):
        with self._lock:
            if self.status == "idle":
                return
            self.status = status
            self.finished_at = now_iso()
            self.duree_s = duree_s if duree_s is not None else (
                round(time.time() - self._t0, 2) if self._t0 else None)
            self.completion_tokens = completion_tokens
            self.tok_s = tok_s
            self.finish_reason = finish_reason
            self.erreur = erreur
            self.updated_at = time.time()

    def clear_if_other_run(self, run_id):
        """Nouveau run : on efface le tir précédent, mais on garde le dernier
        résultat affiché tant qu'on reste dans le même run."""
        with self._lock:
            if self.run_id != run_id:
                self._clear_locked()

    def text_partial(self):
        """Texte déjà produit (pour le sauver si le tir se termine mal)."""
        with self._lock:
            return self.text

    def is_current(self, run_id, pid):
        """True si le tampon décrit bien ce tir-là (garde-fou avant d'écrire/terminer)."""
        with self._lock:
            return self.run_id == run_id and self.pid == pid

    # -- lecture (UI) ------------------------------------------------------
    def snapshot(self, text_cursor=0, reasoning_cursor=0):
        with self._lock:
            t_total = self.dropped + len(self.text)
            r_total = self.rdropped + len(self.reasoning)
            t_reset = text_cursor < self.dropped
            r_reset = reasoning_cursor < self.rdropped
            t_from = 0 if t_reset else max(0, text_cursor - self.dropped)
            r_from = 0 if r_reset else max(0, reasoning_cursor - self.rdropped)
            running = self.status in ("running", "waiting")
            elapsed = round(time.time() - self._t0, 1) if (running and self._t0) else self.duree_s
            return {
                "run_id": self.run_id,
                "id": self.pid,
                "title": self.title,
                "slug": self.slug,
                "mode": self.mode,
                "status": self.status,
                "running": running,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "duree_s": self.duree_s,
                "elapsed_s": elapsed,
                "chunks": self.chunks,
                "chars": t_total,
                "reasoning_chars": r_total,
                "completion_tokens": self.completion_tokens,
                "tok_s": self.tok_s,
                "tok_s_estime": (round(self.chunks / elapsed, 2)
                                 if (running and elapsed and self.chunks) else None),
                "finish_reason": self.finish_reason,
                "erreur": self.erreur,
                "max_tokens": self.max_tokens,
                "params": dict(self.params),
                "params_label": params_label(self.params),
                "truncated": self.truncated,
                "updated_at": (datetime.fromtimestamp(self.updated_at).strftime("%H:%M:%S")
                               if self.updated_at else None),
                # incréments depuis les curseurs du client
                "text": self.text[t_from:],
                "reasoning": self.reasoning[r_from:],
                "text_reset": t_reset,
                "reasoning_reset": r_reset,
                "cursor": t_total,
                "reasoning_cursor": r_total,
            }


LIVE = LiveStore()


# --------------------------------------------------------------------------
# Appels HTTP vers llama.cpp
# --------------------------------------------------------------------------

def http_request(url, method="GET", payload=None, timeout=10, raw_body=None):
    """Renvoie (status, body_text). Lève en cas d'erreur réseau/HTTP."""
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    elif raw_body is not None:
        data = raw_body
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", "replace")
        return resp.status, body


def fetch_model_identity():
    """GET /v1/models + /props : identité du LLM réellement servi."""
    _, raw = http_request(f"{LLM_BASE}/v1/models", timeout=10)
    data = json.loads(raw)
    path = ""
    items = data.get("data") if isinstance(data, dict) else data
    if isinstance(items, list) and items:
        first = items[0]
        if isinstance(first, dict):
            path = first.get("id") or first.get("model") or first.get("name") or ""
        else:
            path = str(first)
    if not path:
        raise RuntimeError("/v1/models ne renvoie aucun modèle")

    n_ctx = None
    try:
        _, praw = http_request(f"{LLM_BASE}/props", timeout=10)
        pdata = json.loads(praw)
        n_ctx = ((pdata.get("default_generation_settings") or {}).get("n_ctx"))
        if n_ctx is None:
            n_ctx = pdata.get("n_ctx")
    except Exception as exc:  # /props absent : on continue sans casser le run
        LOG.add(f"/props indisponible ({exc.__class__.__name__}) — n_ctx inconnu")

    return str(path), n_ctx


def fetch_slot_busy():
    """True/False depuis GET /slots (clé is_processing)."""
    _, raw = http_request(f"{LLM_BASE}/slots", timeout=5)
    data = json.loads(raw)
    if isinstance(data, dict):
        slots = data.get("slots") or []
    else:
        slots = data or []
    for slot in slots:
        if isinstance(slot, dict) and slot.get("is_processing"):
            return True
    return False


def erase_slot_best_effort():
    """POST /slots/0?action=erase — best effort, ne fait jamais échouer le tir."""
    url = f"{LLM_BASE}/slots/0?action=erase"
    try:
        status, _ = http_request(url, method="POST", raw_body=b"", timeout=10)
        LOG.add(f"slot erase : HTTP {status}")
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 404, 405, 501):
            # 501 = « Not Implemented » : vu sur ce llama.cpp (branché sur /slots/0?action=erase)
            LOG.add(f"slot erase non supporté — stateless par construction (HTTP {exc.code})")
        else:
            LOG.add(f"slot erase : HTTP {exc.code} — on continue")
    except Exception:
        LOG.add("slot erase non supporté — stateless par construction")


def _chat_payload(prompt, max_tokens, stream=False, params=None):
    params = normalize_params(params)
    payload = {
        "model": "local",  # ignoré par llama.cpp (serveur mono-modèle)
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "max_tokens": int(max_tokens or params["max_tokens"]),
        "cache_prompt": params["cache_prompt"],
        "chat_template_kwargs": {"enable_thinking": params["enable_thinking"]},
        # v1.4.1 — champ de PREMIER NIVEAU : certains modèles (Ternary Bonsai 2,
        # PrismML) ignorent `chat_template_kwargs.enable_thinking` et raisonnent
        # par défaut. Seul `reasoning_effort` les éteint ("none") ; "medium"
        # raccourcit le raisonnement quand il est demandé. Les runtimes qui ne
        # connaissent pas le champ l'ignorent simplement (rétrocompatible).
        "reasoning_effort": "medium" if params["enable_thinking"] else "none",
        "stream": bool(stream),
    }
    if stream:
        # llama.cpp renvoie alors un dernier morceau avec usage + timings :
        # on garde les vrais compteurs de tokens même en streaming.
        payload["stream_options"] = {"include_usage": True}
    # garantie "sans historique" : exactement 1 message système + 1 utilisateur
    assert len(payload["messages"]) == 2
    return payload


class StreamIncomplete(RuntimeError):
    """Streaming interrompu : dit combien de morceaux étaient déjà arrivés.

    0 morceau → on peut retenter en un bloc sans rien perdre ; au-delà, le tir a
    bel et bien commencé et un second tir complet serait du gaspillage pur.
    """

    def __init__(self, cause, morceaux):
        super().__init__(f"{cause} (après {morceaux} morceau(x) reçu(s))")
        self.cause = cause
        self.morceaux = morceaux


def chat_completion_stream(prompt, max_tokens=None, on_delta=None, params=None):
    """POST /v1/chat/completions en streaming SSE (llama.cpp).

    Renvoie (payload, duree_s) avec EXACTEMENT la même forme que
    chat_completion() : le reste du worker ne voit pas la différence.
    on_delta(kind, morceau) est appelé au fil de l'eau ("content"|"reasoning").
    """
    payload = _chat_payload(prompt, max_tokens, stream=True, params=params)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{LLM_BASE}/v1/chat/completions", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")

    content, reasoning = [], []
    usage, timings, finish_reason = {}, {}, None
    n_content = 0
    saw_done = False
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} sur /v1/chat/completions")
            for raw in resp:                  # itération ligne à ligne (SSE)
                line = raw.decode("utf-8", "replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue                  # ligne vide, commentaire keep-alive
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    saw_done = True
                    break
                try:
                    obj = json.loads(chunk)
                except Exception:
                    continue                  # morceau illisible : on ne casse pas le tir
                if isinstance(obj.get("usage"), dict) and obj["usage"]:
                    usage = obj["usage"]
                if isinstance(obj.get("timings"), dict) and obj["timings"]:
                    timings = obj["timings"]
                for ch in obj.get("choices") or []:
                    if not isinstance(ch, dict):
                        continue
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]
                    delta = ch.get("delta") or ch.get("message") or {}
                    piece = delta.get("content")
                    if piece:
                        content.append(piece)
                        n_content += 1
                        if on_delta:
                            on_delta("content", piece)
                    rpiece = delta.get("reasoning_content")
                    if rpiece:
                        reasoning.append(rpiece)
                        if on_delta:
                            on_delta("reasoning", rpiece)
    except StreamIncomplete:
        raise
    except Exception as exc:
        raise StreamIncomplete(exc, n_content) from exc
    elapsed = time.time() - t0

    # Aucun morceau SSE et pas de finish_reason : le serveur a probablement ignoré
    # "stream": true (il a renvoyé du JSON d'un bloc). On le signale pour que
    # l'appelant retombe sur le tir en un bloc, plutôt que de conclure « réponse vide ».
    if not content and not reasoning and finish_reason is None and not usage:
        raise StreamIncomplete(RuntimeError("aucun morceau SSE reçu (streaming non supporté ?)"), 0)

    # Flux coupé en cours de route (connexion fermée sans [DONE] ni finish_reason) :
    # mieux vaut le dire que de laisser croire à une réponse complète.
    if finish_reason is None and not saw_done:
        raise StreamIncomplete(RuntimeError("flux SSE terminé sans finish_reason ni [DONE]"), n_content)

    # Filet de sécurité : build llama.cpp sans stream_options → pas d'usage final.
    # En streaming llama.cpp émet ~1 token par morceau : on estime, et on le dit.
    estime = False
    if not usage.get("completion_tokens") and n_content:
        usage = dict(usage)
        usage["completion_tokens"] = n_content
        usage["completion_tokens_estimes"] = True
        estime = True
    if estime and not timings.get("predicted_per_second") and elapsed > 0:
        timings = dict(timings)
        timings["predicted_per_second"] = round(n_content / elapsed, 2)

    return {
        "choices": [{
            "message": {"content": "".join(content), "reasoning_content": "".join(reasoning)},
            "finish_reason": finish_reason,
        }],
        "usage": usage,
        "timings": timings,
        "_stream": {"morceaux": n_content, "tokens_estimes": estime, "duree_s": round(elapsed, 2)},
    }, elapsed


def chat_completion(prompt, max_tokens=None, on_delta=None, params=None):
    """POST /v1/chat/completions. Renvoie (payload, duree_s).

    Si `on_delta` est fourni et que le streaming est actif (LLM_STREAM=1, défaut),
    le tir se fait en SSE et le texte arrive au fur et à mesure. Si le streaming
    échoue AVANT le moindre morceau, on retombe proprement sur le tir en un bloc.
    `params` porte les réglages du run (température, top_p, max_tokens, thinking).
    """
    params = normalize_params(params)
    if on_delta is not None and STREAM:
        try:
            data, elapsed = chat_completion_stream(prompt, max_tokens=max_tokens,
                                                  on_delta=on_delta, params=params)
            n = (data.get("_stream") or {}).get("morceaux") or 0
            LOG.add(f"streaming SSE : {n} morceau(x) reçu(s) en {elapsed:.1f} s")
            return data, elapsed
        except StreamIncomplete as exc:
            if exc.morceaux:
                # le tir a commencé : on ne le relance pas une seconde fois
                raise RuntimeError(f"streaming interrompu : {exc}") from exc
            LOG.add(f"streaming indisponible ({exc.cause.__class__.__name__}: {exc.cause}) — "
                    "repli sur le tir en un bloc")
        except Exception as exc:
            LOG.add(f"streaming indisponible ({exc.__class__.__name__}: {exc}) — "
                    "repli sur le tir en un bloc")
        # repli : on garde on_delta, la réponse entière sera poussée d'un coup à la fin

    payload = _chat_payload(prompt, max_tokens, stream=False, params=params)
    t0 = time.time()
    status, raw = http_request(
        f"{LLM_BASE}/v1/chat/completions",
        method="POST",
        payload=payload,
        timeout=REQUEST_TIMEOUT_S,
    )
    elapsed = time.time() - t0
    if status != 200:
        raise RuntimeError(f"HTTP {status} sur /v1/chat/completions")
    data = json.loads(raw)
    if on_delta:                              # mode bloc mais on veut quand même le live
        msg = ((data.get("choices") or [{}])[0].get("message") or {})
        if msg.get("reasoning_content"):
            on_delta("reasoning", msg["reasoning_content"])
        if msg.get("content"):
            on_delta("content", msg["content"])
    return data, elapsed


def dry_run_completion(pid, on_delta=None):
    """Mode LLM_DRY_RUN=1 : aucun appel réseau, HTML bidon valide.

    Simule aussi le streaming (mêmes appels on_delta) pour pouvoir tester le
    suivi en direct sans llama.cpp.
    """
    body = (
        "<!DOCTYPE html>\n<html lang=\"fr\"><head><meta charset=\"utf-8\">"
        f"<title>DRY RUN prompt {pid}</title></head>\n"
        "<body style=\"background:#111;color:#eee;font-family:sans-serif\">"
        f"<h1>DRY RUN prompt {pid}</h1><p>Généré sans contacter llama.cpp.</p>"
        "</body></html>"
    )
    if on_delta:
        pas = max(1, len(body) // 12)
        for i in range(0, len(body), pas):
            on_delta("content", body[i:i + pas])
            time.sleep(DRY_DELAY_S / 12.0)
    else:
        time.sleep(DRY_DELAY_S)
    return {
        "choices": [{"message": {"content": body, "reasoning_content": ""}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 123 + pid, "completion_tokens": 456 + pid, "total_tokens": 579 + 2 * pid},
        "timings": {"predicted_per_second": 12.5 + pid, "predicted_n": 456 + pid},
    }, DRY_DELAY_S


# --------------------------------------------------------------------------
# État global
# --------------------------------------------------------------------------

class State:
    def __init__(self):
        self.lock = threading.RLock()
        self.prompts = []
        self.groups = []
        self.prompts_version = 1
        self.probe = {
            "status": "unknown",
            "reachable": False,
            "model_path": None,
            "model_slug": None,
            "n_ctx": None,
            "slot_busy": None,
            "error": None,
            "checked_at": None,
            "dry_run": DRY_RUN,
        }
        self._probe_at = 0.0
        self.run = None
        self.stop_requested = False
        self.worker = None
        self._run_json_path = None
        # --- tests (v1.5) : un seul worker à la fois, comme pour les duels ------
        self.bench = None
        self.bench_worker = None
        self.bench_stop = False
        self._bench_json_path = None


STATE = State()


DEFAULT_GROUP = {"id": "divers", "title": "Prompts (sans groupe)", "note": "groupe de secours"}


def load_catalog(path=None):
    """Charge prompts.json et renvoie (prompts, groups, version).

    Accepte la forme v2 (objet {version, groups, prompts}) ET l'ancienne forme v1
    (tableau racine) : en v1, tous les prompts tombent dans le groupe de secours
    « divers ». Ne plante jamais, ne perd jamais un prompt.
    """
    src = Path(path) if path is not None else PROMPTS_FILE
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.add(f"ERREUR lecture prompts.json : {exc!r}")
        return [], [], 1

    if isinstance(data, list):
        raw_prompts, groups_raw, version = data, [], 1
    elif isinstance(data, dict):
        raw_prompts = data.get("prompts") or []
        groups_raw = data.get("groups") or []
        try:
            version = int(data.get("version") or 2)
        except (TypeError, ValueError):
            version = 2
    else:
        LOG.add("ERREUR prompts.json : la racine doit être un objet (v2) ou un tableau (v1)")
        return [], [], 1
    if not isinstance(raw_prompts, list):
        LOG.add("ERREUR prompts.json : 'prompts' doit être une liste")
        raw_prompts = []

    groups = []
    for g in groups_raw:
        if not isinstance(g, dict):
            continue
        gid = slugify(g.get("id") or "")
        if not gid or any(x["id"] == gid for x in groups):
            continue
        groups.append({
            "id": gid,
            "title": str(g.get("title") or gid),
            "note": str(g.get("note") or ""),
        })
    known = {g["id"] for g in groups}
    if DEFAULT_GROUP["id"] not in known:
        groups.append(dict(DEFAULT_GROUP))

    problems, prompts, seen = [], [], set()
    for pos, item in enumerate(raw_prompts, 1):
        if not isinstance(item, dict):
            problems.append(f"entrée #{pos} ignorée : ce n'est pas un objet")
            continue
        try:
            pid = int(item.get("id"))
        except (TypeError, ValueError):
            problems.append(f"entrée #{pos} ignorée : id absent ou non entier")
            continue
        if pid in seen:
            problems.append(f"prompt {pid} : doublon ignoré")
            continue
        seen.add(pid)

        gid = slugify(item.get("group") or "") if item.get("group") else ""
        if not gid or gid not in known:
            if gid:
                problems.append(f"prompt {pid} : groupe inconnu → divers")
            gid = DEFAULT_GROUP["id"]

        p = {
            "id": pid,
            "slug": slugify(item.get("slug") or f"prompt-{pid}"),
            "title": str(item.get("title") or f"Prompt {pid}"),
            "group": gid,
            "category": str(item.get("category") or ""),
            "prompt": str(item.get("prompt") or ""),
            "criteria": [str(c) for c in (item.get("criteria") or [])],
        }
        if item.get("max_tokens") is not None:      # porte de sortie optionnelle par prompt
            try:
                p["max_tokens"] = int(item["max_tokens"])
            except (TypeError, ValueError):
                problems.append(f"prompt {pid} : max_tokens illisible, ignoré")
        if item.get("source"):                      # provenance (prompts importés d'ailleurs)
            p["source"] = str(item["source"])
        prompts.append(p)

    prompts.sort(key=lambda p: p["id"])
    for msg in problems:
        LOG.add(f"prompts.json : {msg}")

    # groupes réellement utilisés, dans l'ordre du fichier
    used = []
    for g in groups:
        ids = [p["id"] for p in prompts if p["group"] == g["id"]]
        if not ids and g["id"] == DEFAULT_GROUP["id"]:
            continue                                 # pas de section « divers » vide
        used.append({"id": g["id"], "title": g["title"], "note": g["note"],
                     "count": len(ids), "prompt_ids": ids})
    return prompts, used, version


def load_prompts(path=None):
    """Liste des prompts (compat v1/v2). Chaque prompt porte son 'group'."""
    prompts, _groups, _version = load_catalog(path)
    return prompts


_ALREADY_DONE_CACHE = {"at": 0.0, "key": None, "value": {}}


def list_already_done(model_slug=None, n_ctx=None, params=None, max_age=5.0):
    """{ "<id>": "<run_id>" } des prompts déjà générés avec les MÊMES modèle, ctx et réglages.

    Lecture seule, tolérante à l'absence/illisibilité des fichiers, jamais bloquante.
    Résultat mémorisé quelques secondes (l'UI polle /api/state toutes les secondes).
    """
    params = normalize_params(params)
    key = (model_slug, n_ctx, params["temperature"], params["max_tokens"],
           params["enable_thinking"])
    now = time.time()
    if _ALREADY_DONE_CACHE["key"] == key and (now - _ALREADY_DONE_CACHE["at"]) < max_age:
        return dict(_ALREADY_DONE_CACHE["value"])
    done = {}
    if not RUNS_DIR.exists():
        return done
    try:
        dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir() and not p.is_symlink()]
    except Exception:
        return done
    dirs.sort(key=lambda p: p.name, reverse=True)
    for d in dirs:
        rj = d / "run.json"
        if not rj.exists():
            continue
        try:
            data = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            continue
        if model_slug and data.get("model_slug") != model_slug:
            continue
        if n_ctx is not None and data.get("n_ctx") != n_ctx:
            continue
        # mêmes réglages → même sortie attendue : température, max_tokens, raisonnement
        rp = data.get("params") or {}
        if rp.get("temperature") != params["temperature"]:
            continue
        if rp.get("max_tokens") != params["max_tokens"]:
            continue
        if bool(rp.get("enable_thinking")) != params["enable_thinking"]:
            continue
        for res in data.get("results") or []:
            try:
                rid = int(res.get("id"))
            except (TypeError, ValueError):
                continue
            rel = res.get("rel")
            if str(rid) in done or not rel:
                continue
            try:
                if (d / rel / "index.html").exists():
                    done[str(rid)] = d.name
            except Exception:
                continue
    _ALREADY_DONE_CACHE.update({"at": now, "key": key, "value": dict(done)})
    return done


def _query_flag(path_or_query, name):
    """Case a cocher portee par une URL (« ?partiels=1 ») : vraie sauf valeur fausse."""
    texte = str(path_or_query or "")
    if "?" in texte:
        texte = urlparse(texte).query
    valeurs = parse_qs(texte.lstrip("?"), keep_blank_values=True).get(name)
    if not valeurs:
        return False
    return str(valeurs[-1]).strip().lower() not in ("0", "non", "false", "off", "no")


def probe_llm(force=False):
    """Sonde /health, /v1/models, /props, /slots. Résultat mémorisé PROBE_CACHE_S."""
    with STATE.lock:
        if not force and (time.time() - STATE._probe_at) < PROBE_CACHE_S and STATE.probe["status"] != "unknown":
            return STATE.probe

        if DRY_RUN:
            STATE.probe = {
                "status": "dry",
                "reachable": True,
                "model_path": "dry-run",
                "model_slug": "dry-run",
                "n_ctx": DRY_N_CTX,
                "slot_busy": False,
                "error": None,
                "checked_at": now_iso(),
                "dry_run": True,
            }
            STATE._probe_at = time.time()
            return STATE.probe

        probe = {
            "status": "off",
            "reachable": False,
            "model_path": None,
            "model_slug": None,
            "n_ctx": None,
            "slot_busy": None,
            "error": None,
            "checked_at": now_iso(),
            "dry_run": False,
        }
        try:
            status, raw = http_request(f"{LLM_BASE}/health", timeout=HEALTH_TIMEOUT_S)
            ok = False
            try:
                ok = str(json.loads(raw).get("status", "")).lower() == "ok"
            except Exception:
                ok = (status == 200)
            if status != 200 or not ok:
                raise RuntimeError(f"/health = HTTP {status} {raw[:80]}")
            probe["reachable"] = True
        except urllib.error.HTTPError as exc:
            probe["error"] = f"HTTP {exc.code} sur /health"
            probe["status"] = "off"
        except Exception as exc:
            probe["error"] = f"{exc.__class__.__name__}: {exc}"
            probe["status"] = "off"

        if probe["reachable"]:
            try:
                path, n_ctx = fetch_model_identity()
                probe["model_path"] = path
                probe["model_slug"] = model_slug_from_path(path)
                probe["n_ctx"] = n_ctx
            except Exception as exc:
                probe["error"] = f"identité modèle illisible : {exc}"
            try:
                probe["slot_busy"] = fetch_slot_busy()
            except Exception:
                probe["slot_busy"] = None
            probe["status"] = "busy" if probe["slot_busy"] else "ok"

        STATE.probe = probe
        STATE._probe_at = time.time()
        return probe


# --------------------------------------------------------------------------
# Réglages, variantes et index unifié (v1.6)
#
# Les réglages réellement servis (cmdline + /props) sont capturés AU LANCEMENT
# d'un run ou d'un bench, puis figés dans run.json / meta.json / bench.json :
# un résultat n'est jamais attribuable à un réglage supposé.
# --------------------------------------------------------------------------

# Une entrée de graphique par métrique principale (famille -> métrique chiffrée).
GRAPH_METRICS = (
    ("speed", "pp"),
    ("context", "prefill_tps"),
    ("battery", "score_avg"),
    ("humaneval", "score_pct"),
    ("duel", "tok_s"),
    ("vitesse", "pp_tps"),
    ("memoire", "prefill_tps"),
    ("intelligence", "score_pct"),
    ("intelligence", "score_avg"),
    ("vram", "peak_mib"),
)

# Actions acceptées par POST /api/visibility (tout le reste -> 400).
# « restore » = ré-afficher : il lève le masque ET le tombstone, pour que la
# paire « hide -> restore » du contrat §7-E1.7 ramène bien l'entrée.
def _action_restore(state, ids):
    visibility.restore(state, ids)
    return visibility.unhide(state, ids)


VISIBILITY_ACTIONS = {
    "hide": visibility.hide,
    "unhide": visibility.unhide,
    "delete": visibility.delete,
    "restore": _action_restore,
}

_SETTINGS_CACHE = {"at": 0.0, "port": None, "value": None}
VISIBILITY_LOCK = threading.RLock()


def llm_port():
    """Port du serveur llama.cpp derrière LLM_BASE (8080 par défaut)."""
    try:
        return urlparse(LLM_BASE).port or 8080
    except Exception:
        return 8080


def detect_settings(force=False, max_age=5.0):
    """`settings.detect` du serveur courant, mémorisé quelques secondes.

    /api/state est pollé chaque seconde : sans ce cache, chaque poll ferait un
    appel /props et un scan de /proc. `settings.detect` ne lève jamais ; ici on
    ajoute quand même un filet (jamais de traceback dans l'UI).
    """
    now = time.time()
    port = llm_port()
    with STATE.lock:
        cached = dict(_SETTINGS_CACHE)
    if (not force and cached["value"] is not None and cached["port"] == port
            and (now - cached["at"]) < max_age):
        return dict(cached["value"])
    try:
        value = settings.detect(port)
    except Exception as exc:  # pragma: no cover - ceinture et bretelles
        value = {"source": "declared", "error": f"{exc.__class__.__name__}: {exc}",
                 "variant_key": "default", "variant_label": "réglages inconnus"}
    with STATE.lock:
        _SETTINGS_CACHE.update({"at": now, "port": port, "value": value})
    return dict(value)


def capture_settings(declared=None):
    """Réglages AU LANCEMENT d'un run / d'un bench : détection fraîche + UI.

    Les réglages déclarés par l'UI ne remplissent que les champs que la
    détection n'a pas pu lire (la cmdline réelle fait foi). Jamais d'exception.
    """
    captured = bench_tests.settings_capture(llm_port(), declared)
    if captured is None:  # repli : capture côté bench_tests indisponible
        captured = detect_settings(force=True)
        captured["variant_key"] = settings.variant_key(captured)
        captured["variant_label"] = settings.variant_label(captured)
    captured["captured_at"] = now_iso()
    return captured


def variant_of(captured):
    """Variante (contrat §6) d'un run / d'un bench, calculée par metrics."""
    try:
        return metrics.entry_variant(captured or {})
    except Exception:
        return {"key": "standard", "label": "standard"}


def load_bench_index():
    """Index unifié des tests (mêmes sources que /benchmarks), ou {}."""
    for path in BENCH_INDEX_CANDIDATES:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            LOG.add(f"index de benchmarks illisible ({path}) : {exc}")
            return {}
        if isinstance(data, dict):
            data["_source"] = path
            return data
        return {}
    return {}


def visibility_state():
    """État de visibilité persisté (jamais d'exception, jamais None)."""
    with VISIBILITY_LOCK:
        return visibility.load()


def apply_visibility_action(action, ids):
    """Applique une action de visibilité puis persiste l'état.

    -> (ok, payload) : `payload` est la réponse JSON à renvoyer (400 si l'action
    est inconnue ou si la liste d'identifiants est vide).
    """
    fn = VISIBILITY_ACTIONS.get(str(action or "").strip().lower())
    if fn is None:
        return False, {"ok": False, "error": f"action inconnue : {action!r}",
                       "action": action}
    if isinstance(ids, (str, bytes)):
        ids = [ids]
    ids = [str(i) for i in ids if str(i)] if isinstance(ids, (list, tuple)) else []
    if not ids:
        return False, {"ok": False, "error": "ids doit être une liste non vide",
                       "action": action}
    with VISIBILITY_LOCK:
        state = visibility.load()
        fn(state, ids)
        visibility.save(state)
        clean = visibility.normalize_state(state)
    LOG.add(f"visibilité : {action} — {len(ids)} entrée(s)")
    return True, {"ok": True, "action": str(action).strip().lower(), "ids": ids,
                  "state": clean}


# --------------------------------------------------------------------------
# Gestion des runs sur disque
# --------------------------------------------------------------------------

def run_suffix(model_slug, n_ctx, params=None, variant_key=None):
    """Bouts de nom de dossier identifiant un run (modèle, ctx, réglages).

    `variant_key` (settings.variant_key) est ajouté à la fin : deux exécutions
    qui ne diffèrent que par un réglage serveur (MTP par ex.) ne partagent
    JAMAIS un dossier — sans quoi find_reusable_run confondrait un run « MTP on »
    avec un run « MTP off ».
    """
    params = normalize_params(params)
    ctx = n_ctx if n_ctx else 0
    think = "think" if params["enable_thinking"] else "nothink"
    suffix = (f"__{model_slug}__ctx{ctx}__t{params['temperature']:g}__{think}"
              f"{params_suffix(params)}")
    if variant_key:
        suffix += "__" + slugify(variant_key)
    return suffix


def find_reusable_run(suffix, ids):
    """Un run identique récent est repris -> les prompts déjà générés sont sautés."""
    if not RUNS_DIR.exists():
        return None
    try:
        candidates = sorted(
            [p for p in RUNS_DIR.iterdir()
             if p.is_dir() and not p.is_symlink() and p.name.endswith(suffix)],
            key=lambda p: p.name,
            reverse=True,
        )
    except Exception:
        return None
    for cand in candidates:
        meta = cand / "run.json"
        if not meta.exists():
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            prev_ids = sorted(int(x) for x in (data.get("selected_ids") or []))
        except Exception:
            continue
        if prev_ids != sorted(ids):
            continue
        try:
            started = datetime.strptime(data.get("started_at") or "", "%Y-%m-%dT%H:%M:%S")
        except Exception:
            continue
        age = (datetime.now() - started).total_seconds()
        if 0 <= age <= RUN_REUSE_S:
            return cand
    return None


def update_latest_symlink(run_dir):
    try:
        tmp = RUNS_DIR / ".latest.tmp"
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        os.symlink(run_dir.name, str(tmp))
        os.replace(str(tmp), str(RUNS_LATEST))
    except Exception as exc:
        LOG.add(f"symlink runs/latest non mis à jour : {exc}")


def write_run_json():
    with STATE.lock:
        run = STATE.run
        path = STATE._run_json_path
        if not run or not path:
            return
        payload = dict(run)
        payload["results"] = [dict(r) for r in run.get("results", [])]
    tmp = Path(str(path) + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(path))
    except Exception as exc:
        LOG.add(f"ERREUR écriture run.json : {exc!r}")


def list_past_runs(limit=RUNS_LIST_LIMIT):
    if not RUNS_DIR.exists():
        return []
    dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir() and not p.is_symlink()]
    dirs.sort(key=lambda p: p.name, reverse=True)
    out = []
    for d in dirs[:limit]:
        info = {
            "run_id": d.name,
            "dir": str(d),
            "url": f"/runs/{d.name}/",
            "started_at": None,
            "finished_at": None,
            "model_slug": None,
            "n_ctx": None,
            "dry_run": None,
            "status": None,
            "params": {},
            "params_label": "",
            "selected_ids": [],
            "counts": {},
        }
        meta = d / "run.json"
        if meta.exists():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                info["started_at"] = data.get("started_at")
                info["finished_at"] = data.get("finished_at")
                info["model_slug"] = data.get("model_slug")
                info["n_ctx"] = data.get("n_ctx")
                info["dry_run"] = data.get("dry_run")
                info["status"] = data.get("status")
                info["params"] = data.get("params") or {}
                info["params_label"] = params_label(info["params"])
                info["selected_ids"] = data.get("selected_ids") or []
                counts = {}
                for res in data.get("results") or []:
                    st = res.get("status") or "?"
                    counts[st] = counts.get(st, 0) + 1
                info["counts"] = counts
            except Exception:
                pass
        out.append(info)
    return out


_RUNS_CACHE = {"at": 0.0, "limit": None, "value": []}


def list_past_runs_cached(limit=RUNS_COMPARE_LIMIT, max_age=3.0):
    """Version mémorisée de list_past_runs, pour lister TOUS les runs (sélecteurs de
    comparaison) sans relire des centaines de run.json à chaque poll."""
    now = time.time()
    if _RUNS_CACHE["limit"] == limit and (now - _RUNS_CACHE["at"]) < max_age:
        return _RUNS_CACHE["value"]
    value = list_past_runs(limit=limit)
    _RUNS_CACHE.update({"at": now, "limit": limit, "value": value})
    return value


# --------------------------------------------------------------------------
# Comparaison de deux runs (v1.3)
# --------------------------------------------------------------------------

def run_id_is_safe(run_id):
    """Un run_id est un NOM DE DOSSIER simple — jamais un chemin."""
    rid = str(run_id or "")
    if not rid or len(rid) > 200:
        return False
    if rid in (".", "..") or "/" in rid or "\\" in rid or rid.startswith("."):
        return False
    return True


def load_run_details(run_id):
    """Détails d'un run pour la comparaison : identité, réglages, un enregistrement par prompt.

    Lit run.json + vérifie l'existence du fichier produit (index.html, sinon raw.txt,
    sinon partial.txt). Renvoie None si le run n'existe pas / n'est pas lisible.
    """
    if not run_id_is_safe(run_id):
        return None
    d = RUNS_DIR / str(run_id)
    rj = d / "run.json"
    if not d.is_dir() or d.is_symlink() or not rj.exists():
        return None
    try:
        data = json.loads(rj.read_text(encoding="utf-8"))
    except Exception:
        return None

    params = data.get("params") or {}
    counts = {}
    for r in data.get("results") or []:
        st = r.get("status") or "?"
        counts[st] = counts.get(st, 0) + 1
    out = {
        "run_id": str(run_id),
        "url": f"/runs/{run_id}/",
        "started_at": data.get("started_at"),
        "finished_at": data.get("finished_at"),
        "model_slug": data.get("model_slug"),
        "n_ctx": data.get("n_ctx"),
        "params": params,
        "params_label": params_label(params),
        "status": data.get("status"),
        "dry_run": data.get("dry_run"),
        "app_version": data.get("app_version"),
        "counts": counts,
        "selected_ids": data.get("selected_ids") or [],
        "results": {},
    }
    for r in data.get("results") or []:
        try:
            rid = int(r.get("id"))
        except (TypeError, ValueError):
            continue
        rel = str(r.get("rel") or "")
        if not rel or ".." in rel:
            continue
        open_file = None
        for name in ("index.html", "raw.txt", "partial.txt", "reasoning.txt"):
            try:
                if (d / rel / name).exists():
                    open_file = name
                    break
            except Exception:
                continue
        out["results"][rid] = {
            "id": rid,
            "title": r.get("title"),
            "group": r.get("group"),
            "status": r.get("status"),
            "duree_s": r.get("duree_s"),
            "tok_s": r.get("tok_s"),
            "prompt_tokens": r.get("prompt_tokens"),
            "completion_tokens": r.get("completion_tokens"),
            "erreur": r.get("erreur"),
            "rel": rel,
            "file": open_file,
            "open_url": (f"/runs/{run_id}/{rel}/{open_file}" if open_file else None),
        }
    return out


def _delta(b, a):
    """b - a, ou None si l'une des deux valeurs manque."""
    try:
        if a is None or b is None:
            return None
        return round(float(b) - float(a), 2)
    except (TypeError, ValueError):
        return None


def _delta_pct(b, a):
    """Écart relatif (b - a) / a en %, arrondi à 0,1. None si une valeur manque ou si a = 0."""
    try:
        if a is None or b is None:
            return None
        a = float(a)
        b = float(b)
        if a == 0:
            return None
        return round((b - a) / abs(a) * 100.0, 1)
    except (TypeError, ValueError):
        return None


def compare_rows(a, b, params_keys=("duree_s", "tok_s", "completion_tokens", "prompt_tokens")):
    """Une ligne par prompt (union des deux runs), avec les deux côtés et les écarts."""
    ids = sorted(set(a["results"]) | set(b["results"]))
    rows = []
    for rid in ids:
        ra, rb = a["results"].get(rid), b["results"].get(rid)
        ref = ra or rb or {}
        delta = {}
        for k in params_keys:
            delta[k] = _delta((rb or {}).get(k), (ra or {}).get(k))
        delta["duree_pct"] = _delta_pct((rb or {}).get("duree_s"), (ra or {}).get("duree_s"))
        delta["tok_s_pct"] = _delta_pct((rb or {}).get("tok_s"), (ra or {}).get("tok_s"))
        rows.append({
            "id": rid,
            "title": ref.get("title"),
            "group": ref.get("group"),
            "a": ra,
            "b": rb,
            "delta": delta,
            "in_both": bool(ra and rb),
            "html_both": bool(ra and ra.get("file") == "index.html"
                             and rb and rb.get("file") == "index.html"),
        })
    return rows


def compare_payload(run_a, run_b):
    a = load_run_details(run_a)
    b = load_run_details(run_b)
    if a is None or b is None:
        return None, {
            "error": "run introuvable ou illisible",
            "missing": [x for x, v in ((run_a, a), (run_b, b)) if v is None],
        }
    return {"a": a, "b": b, "rows": compare_rows(a, b)}, None


COMPARE_CSS = """
:root{--bg:#12141a;--card:#1b1e26;--fg:#e6e8ee;--dim:#98a0b3;--acc:#4da3ff;--ok:#2ecc71;
--warn:#f39c12;--err:#e74c3c;--line:#2a2f3a;--a:#4da3ff;--b:#b58cff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:12px 18px;background:#0f1116;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:600}
main{padding:14px 18px 40px}
.ids{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px}
.id{flex:1 1 380px;min-width:300px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.id.a{border-left:3px solid var(--a)} .id.b{border-left:3px solid var(--b)}
.id .lab{font-weight:700;font-size:13px}
.id .lab.a{color:var(--a)} .id .lab.b{color:var(--b)}
.id code{font-size:12px;color:var(--dim);word-break:break-all}
.ctrl{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:6px 0 14px;color:var(--dim);font-size:13px}
label.chk{display:flex;gap:6px;align-items:center;cursor:pointer}
button{background:#242a36;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:6px 11px;cursor:pointer;font:inherit}
button:hover{border-color:var(--acc)}
a{color:var(--acc)}
.block{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:14px}
.block>h2{font-size:14px;margin:0 0 8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.tag{font-size:11px;color:var(--dim);border:1px solid var(--line);border-radius:999px;padding:1px 7px}
.pair{display:flex;gap:12px;align-items:stretch}
.pane{flex:1 1 0;min-width:0;display:flex;flex-direction:column;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.pane.a{border-color:rgba(77,163,255,.45)} .pane.b{border-color:rgba(181,140,255,.45)}
.pane .head{display:flex;gap:8px;align-items:center;padding:6px 9px;background:#0f1116;border-bottom:1px solid var(--line);font-size:12.5px;flex-wrap:wrap}
.pane .head .lab{font-weight:700}
.pane.a .head .lab{color:var(--a)} .pane.b .head .lab{color:var(--b)}
.pane .head .num{color:var(--dim)}
.pane iframe{width:100%;height:62vh;min-height:320px;border:0;background:#fff;display:block}
.absent{padding:18px;color:var(--dim);font-size:13px;text-align:center;background:#0d0f14;height:62vh;min-height:320px;display:flex;align-items:center;justify-content:center}
.st{font-weight:600}.st.ok{color:var(--ok)}.st.no_html{color:var(--warn)}
.st.error{color:var(--err)}.st.pending,.st.skipped{color:var(--dim)}
.st.running{color:var(--acc)}.st.deja_genere{color:#b58cff}
.win{color:var(--ok)}.lose{color:var(--err)}
@media (max-width:900px){.pair{flex-direction:column}.pane iframe,.absent{height:70vh}}
"""


def _pane_html(side, run, r, pid, label):
    cls = "a" if side == "a" else "b"
    if not r:
        return (f'<div class="pane {cls}"><div class="head"><span class="lab">{label}</span>'
                '<span class="num">absent de ce run</span></div>'
                '<div class="absent">aucun résultat pour ce prompt dans ce run</div></div>')
    st = str(r.get("status") or "?")
    bits = [f'<span class="st {html.escape(st)}">{html.escape(st)}</span>']
    if r.get("duree_s") is not None:
        bits.append(f'<span class="num">{r["duree_s"]} s</span>')
    if r.get("tok_s") is not None:
        bits.append(f'<span class="num">{r["tok_s"]} tok/s</span>')
    if r.get("completion_tokens") is not None:
        bits.append(f'<span class="num">{r["completion_tokens"]} tok</span>')
    if r.get("prompt_tokens") is not None:
        bits.append(f'<span class="num">prompt {r["prompt_tokens"]} tok</span>')
    if r.get("file"):
        bits.append(f'<span class="num">{html.escape(str(r["file"]))}</span>')
    link = ""
    if r.get("open_url"):
        link = (f'<a href="{html.escape(r["open_url"])}" target="_blank" '
                'rel="noopener">plein écran ↗</a>')
    else:
        link = '<span class="num">aucun fichier</span>'
    reload_btn = f'<button onclick="reloadPane(\'f{side}{pid}\')">↻</button>'
    inner = ""
    if r.get("file") == "index.html" and r.get("open_url"):
        inner = (f'<iframe id="f{side}{pid}" src="{html.escape(r["open_url"])}" '
                 'loading="lazy" referrerpolicy="no-referrer" '
                 'sandbox="allow-scripts allow-same-origin allow-pointer-lock '
                 'allow-modals allow-downloads allow-popups"></iframe>')
    elif r.get("erreur"):
        inner = f'<div class="absent">erreur : {html.escape(str(r["erreur"])[:400])}</div>'
    else:
        txt = ("pas de HTML exploitable — " +
               (f'<a href="{html.escape(r["open_url"])}" target="_blank" rel="noopener">'
                f'ouvrir {html.escape(str(r["file"]))} ↗</a>' if r.get("open_url") else "aucun fichier"))
        inner = f'<div class="absent">{txt}</div>'
    return (f'<div class="pane {cls}"><div class="head"><span class="lab">{label}</span>'
            + " · ".join(bits) + f'<span style="flex:1"></span>{reload_btn}{link}</div>{inner}</div>')


def render_compare_page(query):
    """Page autonome : deux rendus côte à côte, prompt par prompt."""
    run_a = (query.get("a") or [""])[0]
    run_b = (query.get("b") or [""])[0]
    ids_raw = (query.get("ids") or [""])[0]
    payload, err = compare_payload(run_a, run_b)

    head = ('<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Comparaison · Prompt Duel</title>'
            f'<style>{COMPARE_CSS}</style></head><body>'
            '<header><h1>Comparaison de deux runs</h1>'
            '<span style="flex:1"></span>'
            '<a href="/benchmarks" style="margin-right:14px">benchmarks</a>'
            '<a href="/">← retour à Prompt Duel</a></header><main>')

    if err:
        missing = ", ".join(html.escape(str(m)) for m in err["missing"])
        return (head + f'<div class="block"><h2>{html.escape(err["error"])}</h2>'
                f'<p>Run(s) manquant(s) : <code>{missing}</code></p>'
                '<p>Choisis deux runs dans la carte « Comparer deux runs » de l\'UI.</p>'
                '</div></main></body></html>')

    a, b = payload["a"], payload["b"]
    rows = payload["rows"]
    want = set()
    for chunk in str(ids_raw).replace(" ", "").split(","):
        if chunk.isdigit():
            want.add(int(chunk))
    if want:
        rows = [r for r in rows if r["id"] in want]
    rows = [r for r in rows if (r["a"] and r["a"].get("file")) or (r["b"] and r["b"].get("file"))]

    def ident(side, run, label):
        cls = "a" if side == "a" else "b"
        return (f'<div class="id {cls}"><div class="lab {cls}">{label}</div>'
                f'<div><code>{html.escape(run["run_id"])}</code></div>'
                f'<div class="num" style="color:var(--dim);font-size:12.5px">'
                f'modèle {html.escape(str(run.get("model_slug") or "?"))} · '
                f'ctx {html.escape(str(run.get("n_ctx") or "?"))} · '
                f'{html.escape(run.get("params_label") or "")} · '
                f'{html.escape(str(run.get("started_at") or "?"))} · '
                f'{len(run["results"])} prompt(s) · '
                f'<a href="{html.escape(run["url"])}" target="_blank" rel="noopener">'
                'parcourir le dossier ↗</a></div></div>')

    parts = [head,
             '<div class="ids">', ident("a", a, "RUN A — référence"),
             ident("b", b, "RUN B — comparé"), '</div>',
             '<div class="ctrl">',
             f'<label class="chk"><input type="checkbox" id="onlyboth"> '
             'seulement les prompts présents des deux côtés</label>',
             f'<label class="chk"><input type="checkbox" id="onlyhtml"> '
             'seulement les prompts avec un rendu HTML des deux côtés</label>',
             '<button onclick="reloadAll()">↻ recharger les deux aperçus</button>',
             f'<span>{len(rows)} prompt(s) affiché(s)</span>',
             '</div>']

    if not rows:
        parts.append('<div class="block">Aucun prompt comparable entre ces deux runs '
                     '(aucun fichier produit de part et d\'autre).</div>')

    for r in rows:
        delta = r["delta"]
        dbits = []
        if delta.get("tok_s") is not None:
            d = delta["tok_s"]
            p = delta.get("tok_s_pct")
            cls = "win" if d > 0 else ("lose" if d < 0 else "")
            extra = f" ({p:+.1f} %)" if p is not None else ""
            dbits.append(f'<span class="{cls}">tok/s {d:+.1f}{extra}</span>')
        if delta.get("duree_s") is not None:
            d = delta["duree_s"]
            p = delta.get("duree_pct")
            cls = "win" if d < 0 else ("lose" if d > 0 else "")
            extra = f" ({p:+.1f} %)" if p is not None else ""
            dbits.append(f'<span class="{cls}">durée {d:+.1f} s{extra}</span>')
        if delta.get("completion_tokens") is not None:
            dbits.append(f'<span class="num">tokens {delta["completion_tokens"]:+.0f}</span>')
        title = html.escape(str(r.get("title") or f'prompt {r["id"]}'))
        parts.append(
            f'<div class="block" data-both="{int(r["in_both"])}" '
            f'data-html="{int(r["html_both"])}">'
            f'<h2>#{r["id"]} · {title}'
            f'<span class="tag">{html.escape(str(r.get("group") or ""))}</span>'
            f'{" ".join(dbits)}'
            f'<span style="flex:1"></span>'
            f'<a href="/runs/{html.escape(a["run_id"])}/{html.escape(str((r["a"] or {}).get("rel") or ""))}/index.html" '
            f'target="_blank" rel="noopener">A ↗</a>'
            f'<a href="/runs/{html.escape(b["run_id"])}/{html.escape(str((r["b"] or {}).get("rel") or ""))}/index.html" '
            f'target="_blank" rel="noopener">B ↗</a></h2>'
            f'<div class="pair">{_pane_html("a", a, r["a"], r["id"], "A")}'
            f'{_pane_html("b", b, r["b"], r["id"], "B")}</div></div>')

    parts.append("""
<script>
function reloadPane(id){const f=document.getElementById(id);if(f)f.src=f.src;}
function reloadAll(){document.querySelectorAll('iframe').forEach(f=>{f.src=f.src;});}
function applyFilters(){
  const both=document.getElementById('onlyboth').checked;
  const html=document.getElementById('onlyhtml').checked;
  document.querySelectorAll('.block[data-both]').forEach(b=>{
    let show=true;
    if(both && b.dataset.both!=='1') show=false;
    if(html && b.dataset.html!=='1') show=false;
    b.style.display=show?'':'none';
  });
}
document.getElementById('onlyboth').onchange=applyFilters;
document.getElementById('onlyhtml').onchange=applyFilters;
</script>""")
    parts.append("</main></body></html>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Worker d'exécution (un seul à la fois — 1 slot llama.cpp)
# --------------------------------------------------------------------------

SLOTS_DISPONIBLE = None        # None = pas encore sondé ; False = serveur sans /slots


SLOT_ESSAIS = 3                 # essais avant d'abandonner un prompt (serveur injoignable)
SLOT_ESSAI_S = 5                # délai entre deux essais


def wait_for_free_slot():
    """Boucle jusqu'à ce que le slot soit libre. False si stop demandé.

    Deux cas particuliers, tous deux mesurés sur un vrai duel :
      * serveur qui n'expose PAS `/slots` (API OpenAI seule, ex. Strata) -> 404 :
        on ne peut pas connaître l'état du slot ; les tirs étant déjà séquentiels,
        on continue sans attente (une seule trace, pas une par prompt) ;
      * serveur momentanément injoignable (relance du moteur en cours de run) :
        quelques essais espacés avant d'abandonner le prompt, au lieu de brûler
        le reste du duel.
    """
    global SLOTS_DISPONIBLE
    essais = 0
    while True:
        with STATE.lock:
            if STATE.stop_requested:
                return False
        if SLOTS_DISPONIBLE is False:
            return True
        try:
            busy = fetch_slot_busy()
        except urllib.error.HTTPError as exc:
            if exc.code not in (404, 405, 501):
                raise RuntimeError(f"lecture de /slots impossible : {exc}")
            SLOTS_DISPONIBLE = False
            LOG.add("/slots absent (404) — serveur sans routes natives : "
                    "pas d'attente de slot, les tirs restent séquentiels")
            return True
        except Exception as exc:
            essais += 1
            if essais > SLOT_ESSAIS:
                raise RuntimeError(f"lecture de /slots impossible : {exc}")
            LOG.add(f"serveur injoignable ({exc}) — essai {essais}/{SLOT_ESSAIS} "
                    f"dans {SLOT_ESSAI_S} s")
            for _ in range(int(SLOT_ESSAI_S * 10)):
                time.sleep(0.1)
                with STATE.lock:
                    if STATE.stop_requested:
                        return False
            continue
        essais = 0
        SLOTS_DISPONIBLE = True
        if not busy:
            return True
        grp = []
        with STATE.lock:
            if STATE.run:
                grp = [r["id"] for r in STATE.run["results"]
                       if r["status"] in ("pending", "running")]
        LOG.add(f"en attente du slot (is_processing=true) — en file : {grp}")
        for _ in range(int(SLOT_POLL_S * 10)):
            time.sleep(0.1)
            with STATE.lock:
                if STATE.stop_requested:
                    return False


def make_result(prompt, params=None):
    params = normalize_params(params)
    return {
        "id": prompt["id"],
        "title": prompt["title"],
        "group": prompt["group"],
        "category": prompt["category"],
        "slug": prompt["slug"],
        "max_tokens": prompt.get("max_tokens") or params["max_tokens"],
        "rel": f"{prompt['id']:02d}_{prompt['slug']}",
        "status": "pending",
        "duree_s": None,
        "tok_s": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "file": None,
        "url": None,
        "erreur": None,
        "timestamp": None,
    }


def run_worker(ids, params=None, declared_settings=None):
    """Exécute les prompts un par un, dans l'ordre croissant des ids."""
    params = normalize_params(params)
    try:
        # --- identité du LLM, figée AVANT le premier tir -------------------
        if DRY_RUN:
            model_path, model_slug, n_ctx = "dry-run", "dry-run", DRY_N_CTX
        else:
            try:
                model_path, n_ctx = fetch_model_identity()
                model_slug = model_slug_from_path(model_path)
            except Exception as exc:
                LOG.add(f"ERREUR identité LLM indisponible : {exc}")
                with STATE.lock:
                    for res in STATE.run["results"]:
                        res["status"] = "error"
                        res["erreur"] = f"LLM injoignable ({exc})"
                    STATE.run["status"] = "error"
                    STATE.run["finished_at"] = now_iso()
                write_run_json()
                return

        # réglages réellement servis, capturés au lancement (jamais supposés)
        captured = capture_settings(declared_settings)
        variant = variant_of(captured)
        suffix = run_suffix(model_slug, n_ctx, params, captured.get("variant_key"))
        LOG.add(f"réglages du run : {params_label(params)}")
        LOG.add(f"réglages serveur ({captured.get('source')}) : {captured.get('variant_label')}")
        with STATE.lock:
            reuse = find_reusable_run(suffix, ids)
            previous_started_at = None
            if reuse is not None:
                run_dir = reuse
                try:
                    previous_started_at = json.loads(
                        (run_dir / "run.json").read_text(encoding="utf-8")
                    ).get("started_at")
                except Exception:
                    previous_started_at = None
                LOG.add(f"run repris (idempotence < {RUN_REUSE_S // 60} min) : {run_dir.name}")
            else:
                base = datetime.now().strftime("%Y-%m-%d_%H%M") + suffix
                run_dir = RUNS_DIR / base
                n = 2
                while run_dir.exists():
                    run_dir = RUNS_DIR / f"{base}-{n}"
                    n += 1
                run_dir.mkdir(parents=True, exist_ok=True)
                LOG.add(f"dossier de run créé : {run_dir}")

            STATE._run_json_path = run_dir / "run.json"
            STATE.run.update({
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "run_url": f"/runs/{run_dir.name}/",
                "endpoint": LLM_BASE,
                "model_path": model_path,
                "model_slug": model_slug,
                "n_ctx": n_ctx,
                "settings": captured,          # réglages réels du serveur
                "variant": variant,            # variante (contrat §6)
                "status": "running",
                "last_run_at": now_iso(),
            })
            if previous_started_at:
                # le dossier de run garde son horodatage d'origine
                STATE.run["started_at"] = previous_started_at

        # copie des prompts utilisés dans le run (le run sélectionné uniquement),
        # au format v2 pour rester rechargeable par load_catalog()
        with STATE.lock:
            selected = [dict(p) for p in STATE.prompts if p["id"] in ids]
            sel_groups = [dict(g) for g in STATE.groups
                          if any(p["group"] == g["id"] for p in selected)]
            pversion = STATE.prompts_version
        (run_dir / "prompts.json").write_text(
            json.dumps({"version": pversion, "groups": sel_groups, "prompts": selected},
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )

        update_latest_symlink(run_dir)
        write_run_json()

        # nouveau run : le panneau « suivi en direct » ne montre plus le tir d'avant
        LIVE.clear_if_other_run(run_dir.name)

        # --- boucle séquentielle -------------------------------------------
        for pid in ids:
            with STATE.lock:
                if STATE.stop_requested:
                    LOG.add("stop demandé — les prompts restants ne sont pas lancés")
                    break
                res = next(r for r in STATE.run["results"] if r["id"] == pid)
                prompt = next(p for p in STATE.prompts if p["id"] == pid)
            eff_max_tokens = prompt.get("max_tokens") or params["max_tokens"]

            sub = run_dir / res["rel"]
            index_html = sub / "index.html"
            if index_html.exists() and index_html.stat().st_size > 0:
                info = {}
                meta_file = sub / "meta.json"
                if meta_file.exists():
                    try:
                        info = json.loads(meta_file.read_text(encoding="utf-8"))
                    except Exception:
                        info = {}
                with STATE.lock:
                    res.update({
                        "status": "deja_genere",
                        "file": "index.html",
                        "url": f"{STATE.run['run_url']}{res['rel']}/index.html",
                        "duree_s": info.get("duree_s"),
                        "tok_s": info.get("tok_s"),
                        "prompt_tokens": info.get("prompt_tokens"),
                        "completion_tokens": info.get("completion_tokens"),
                        "timestamp": info.get("timestamp"),
                    })
                LOG.add(f"[{pid:02d}] déjà généré — sauté ({index_html})")
                write_run_json()
                continue

            sub.mkdir(parents=True, exist_ok=True)
            t_start = time.time()
            with STATE.lock:
                res["status"] = "running"
            LOG.add(f"[{pid:02d}] {prompt['title']} — tir en cours")
            # le suivi en direct repart de zéro pour ce prompt
            LIVE.begin(run_dir.name, pid, prompt["title"], prompt["slug"],
                       mode=("dry" if DRY_RUN else ("stream" if STREAM else "bloc")),
                       max_tokens=eff_max_tokens, params=params)

            try:
                if not DRY_RUN:
                    LIVE.set_status("waiting")
                    if not wait_for_free_slot():
                        with STATE.lock:
                            res["status"] = "skipped"
                        LIVE.finish("skipped")
                        LOG.add(f"[{pid:02d}] arrêté avant le tir (stop demandé)")
                        continue
                    LIVE.set_status("running")
                    erase_slot_best_effort()

                if DRY_RUN:
                    data, elapsed = dry_run_completion(pid, on_delta=LIVE.append)
                else:
                    data, elapsed = chat_completion(prompt["prompt"], max_tokens=eff_max_tokens,
                                                    on_delta=LIVE.append, params=params)

                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError("réponse sans 'choices'")
                choice = choices[0] or {}
                message = choice.get("message") or {}
                content = (message.get("content") or "").strip()
                reasoning = (message.get("reasoning_content") or "").strip()
                finish_reason = choice.get("finish_reason")
                usage = data.get("usage") or {}
                timings = data.get("timings") or {}

                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                tok_s = timings.get("predicted_per_second")
                if not tok_s and completion_tokens and elapsed > 0:
                    tok_s = round(completion_tokens / elapsed, 2)

                if not content:
                    msg = "réponse vide dans content"
                    if reasoning:
                        msg += f" — {len(reasoning)} caractères dans reasoning_content (enable_thinking mal pris en compte ?)"
                    if finish_reason == "length":
                        msg += " — finish_reason=length (max_tokens atteint)"
                    raise RuntimeError(msg)

                html_doc = extract_html(content)
                if html_doc:
                    (sub / "index.html").write_text(html_doc, encoding="utf-8")
                    status, filename = "ok", "index.html"
                    LOG.add(f"[{pid:02d}] ok — {len(html_doc)} caractères HTML, {elapsed:.1f} s, {tok_s or '?'} tok/s")
                else:
                    (sub / "raw.txt").write_text(content, encoding="utf-8")
                    status, filename = "no_html", "raw.txt"
                    LOG.add(f"[{pid:02d}] pas de HTML détecté — réponse brute dans raw.txt ({len(content)} caractères)")

                if finish_reason == "length":
                    LOG.add(f"prompt {pid} : réponse tronquée (length) — HTML probablement incomplet "
                            f"(max_tokens={eff_max_tokens})")

                if reasoning:
                    # trace du raisonnement (utile quand enable_thinking est activé)
                    (sub / "reasoning.txt").write_text(reasoning, encoding="utf-8")
                    LOG.add(f"[{pid:02d}] raisonnement conservé ({len(reasoning)} caractères) "
                            "dans reasoning.txt")

                if (usage or {}).get("completion_tokens_estimes"):
                    LOG.add(f"[{pid:02d}] tokens estimés d'après les morceaux SSE "
                            "(ce build llama.cpp n'a pas renvoyé d'usage)")

                LIVE.finish(status, duree_s=round(elapsed, 2),
                            completion_tokens=completion_tokens, tok_s=tok_s,
                            finish_reason=finish_reason, erreur=None)

                meta = {
                    "id": pid,
                    "slug": prompt["slug"],
                    "title": prompt["title"],
                    "group": prompt["group"],
                    "category": prompt["category"],
                    "status": status,
                    "timestamp": now_iso(),
                    "duree_s": round(elapsed, 2),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "tok_s": tok_s,
                    "response_chars": len(content),
                    "reasoning_chars": len(reasoning),
                    "finish_reason": finish_reason,
                    "prompt": prompt["prompt"],
                    "system_prompt": SYSTEM_PROMPT,
                    "params": {**params, "max_tokens": eff_max_tokens},
                    "model_path": model_path,
                    "model_slug": model_slug,
                    "n_ctx": n_ctx,
                    "settings": captured,          # réglages réels du serveur
                    "variant": variant,
                    "fichier": filename,
                    "erreur": None,
                    "dry_run": DRY_RUN,
                }
                (sub / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

                with STATE.lock:
                    res.update({
                        "status": status,
                        "file": filename,
                        "url": f"{STATE.run['run_url']}{res['rel']}/{filename}",
                        "duree_s": round(elapsed, 2),
                        "tok_s": tok_s,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "timestamp": meta["timestamp"],
                    })

            except Exception as exc:
                with STATE.lock:
                    res["status"] = "error"
                    res["erreur"] = f"{exc.__class__.__name__}: {exc}"
                LOG.add(f"[{pid:02d}] ERREUR : {exc}")
                # le panneau live ne doit être clôturé que s'il décrit bien CE tir
                if LIVE.is_current(run_dir.name, pid):
                    LIVE.finish("error", duree_s=round(time.time() - t_start, 2),
                                erreur=f"{exc.__class__.__name__}: {exc}")
                try:
                    # garde une trace du texte déjà produit (tir interrompu, réponse vide…)
                    partial = LIVE.text_partial()
                    if partial and not (sub / "index.html").exists() and not (sub / "raw.txt").exists():
                        (sub / "partial.txt").write_text(partial, encoding="utf-8")
                        LOG.add(f"[{pid:02d}] texte partiel conservé : "
                                f"{(sub / 'partial.txt').relative_to(run_dir)} "
                                f"({len(partial)} caractères)")
                except Exception:
                    pass
                try:
                    (sub / "meta.json").write_text(json.dumps({
                        "id": pid, "slug": prompt["slug"], "title": prompt["title"],
                        "group": prompt["group"], "category": prompt["category"],
                        "status": "error", "timestamp": now_iso(), "erreur": str(exc),
                        "duree_s": round(time.time() - t_start, 2),
                        "prompt": prompt["prompt"], "model_path": model_path,
                        "model_slug": model_slug, "n_ctx": n_ctx, "fichier": None,
                        "settings": captured, "variant": variant,
                        "params": {**params, "max_tokens": eff_max_tokens},
                        "dry_run": DRY_RUN,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass

            write_run_json()

        # --- clôture --------------------------------------------------------
        with STATE.lock:
            if STATE.stop_requested:
                for r in STATE.run["results"]:
                    if r["status"] == "pending":
                        r["status"] = "skipped"
                STATE.run["status"] = "stopped"
            elif any(r["status"] == "error" for r in STATE.run["results"]):
                STATE.run["status"] = "error" if all(
                    r["status"] in ("error", "skipped") for r in STATE.run["results"]
                ) else "finished"
            else:
                STATE.run["status"] = "finished"
            STATE.run["finished_at"] = now_iso()
        write_run_json()
        with STATE.lock:
            LOG.add(f"run {STATE.run['run_id']} terminé — statut {STATE.run['status']}")
        update_latest_symlink(run_dir)

    except Exception as exc:  # filet de sécurité : jamais de traceback dans l'UI
        LOG.add(f"ERREUR worker : {exc!r}")
        with STATE.lock:
            if STATE.run:
                STATE.run["status"] = "error"
                STATE.run["finished_at"] = now_iso()
        write_run_json()
    finally:
        with STATE.lock:
            STATE.stop_requested = False
            STATE.worker = None
        probe_llm(force=True)


def start_run(ids, params=None, declared_settings=None):
    params = normalize_params(params)
    prompts = {p["id"]: p for p in STATE.prompts}
    ids = sorted(int(i) for i in ids)
    with STATE.lock:
        if STATE.worker is not None and STATE.worker.is_alive():
            return False, "un run est déjà en cours"
        if not ids:
            return False, "sélection vide"
        unknown = [i for i in ids if i not in prompts]
        if unknown:
            return False, f"ids inconnus : {unknown}"
        STATE.stop_requested = False
        STATE.run = {
            "run_id": None,
            "run_dir": None,
            "run_url": None,
            "started_at": now_iso(),
            "finished_at": None,
            "endpoint": LLM_BASE,
            "model_path": None,
            "model_slug": None,
            "n_ctx": None,
            "params": dict(params),
            "params_label": params_label(params),
            "settings": None,              # rempli par le worker (détection réelle)
            "variant": None,
            "python_version": sys.version.split()[0],
            "app_version": APP_VERSION,
            "prompts_version": STATE.prompts_version,
            "groups": [{"id": g["id"], "title": g["title"], "count": g.get("count", 0)}
                       for g in STATE.groups],
            "dry_run": DRY_RUN,
            "selected_ids": ids,
            "status": "running",
            "results": [make_result(prompts[i], params) for i in ids],
        }
        worker = threading.Thread(target=run_worker,
                                  args=(ids, params, declared_settings),
                                  name="prompt-duel-worker", daemon=True)
        STATE.worker = worker
        worker.start()
    LOG.add(f"run lancé — {len(ids)} prompt(s) : {ids} · réglages {params_label(params)}")
    return True, "run lancé"


# --------------------------------------------------------------------------
# Tests : vitesse / mémoire de contexte / intelligence (v1.5)
#
# Un test = un workload déclaré dans bench_tests.py, exécuté par le même genre de
# worker que les duels (thread unique : le serveur llama.cpp n'a qu'un seul slot).
# Les chiffres viennent du serveur (timings) ou des harnais du parc ; l'ESTIMATION
# avant lancement vient de bench_calib.json (mesures déjà faites sur ce modèle).
# --------------------------------------------------------------------------

def llm_is_local():
    """Vrai si le serveur LLM sert sur CETTE machine (scripts locaux + nvidia-smi)."""
    try:
        host = urlparse(LLM_BASE).hostname or ""
    except Exception:
        host = ""
    return host in ("127.0.0.1", "localhost", "0.0.0.0", "::1")


def bench_label(model_slug, n_ctx):
    """Étiquette des artefacts : <slug court>-<ctx>k (convention des cas du parc)."""
    slug = re.sub(r"[^a-z0-9]+", "-", (model_slug or "modele").lower()).strip("-")
    slug = re.sub(r"-gguf$", "", slug)[:40].strip("-")
    k = int(n_ctx or 0) // 1024
    return f"{slug}-{k}k" if k else slug


def bench_calib():
    return bench_tests.load_calib(str(BENCH_CALIB_FILE))


def bench_calib_entry(probe=None):
    probe = probe if probe is not None else (STATE.probe or {})
    key = bench_tests.calib_key(probe.get("model_slug"), probe.get("n_ctx"))
    return bench_calib().get(key) or {}


_HE_CACHE = {"at": 0.0, "ok": None, "err": None}


def he_remote_ok(max_age=60.0):
    """Le harnais HumanEval répond-il sur la machine d'à côté ? (mis en cache 60 s)"""
    now = time.time()
    if _HE_CACHE["ok"] is not None and (now - _HE_CACHE["at"]) < max_age:
        return _HE_CACHE["ok"], _HE_CACHE["err"]
    ok, err = False, None
    try:
        p = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", BENCH_HE_SSH,
             f"test -f {BENCH_HE_DIR}/run_he.py && echo OK"],
            capture_output=True, text=True, timeout=25)
        ok = "OK" in (p.stdout or "")
        if not ok:
            err = f"harnais HumanEval injoignable ({BENCH_HE_SSH}:{BENCH_HE_DIR}/run_he.py)"
    except Exception as exc:
        err = f"ssh {BENCH_HE_SSH} impossible ({exc.__class__.__name__})"
    _HE_CACHE.update({"at": now, "ok": ok, "err": err})
    return ok, err


_SERVER_NOTE = {"at": 0.0, "note": None}


def llm_server_note(max_age=30.0):
    """Note sur la ligne de commande du serveur local — le draft MTP fausse les scores
    HumanEval (réponses corrompues observées le 11/08 : ~15 % avec draft, 1,8 % sans)."""
    if not llm_is_local() or DRY_RUN:
        return None
    now = time.time()
    if (now - _SERVER_NOTE["at"]) < max_age:
        return _SERVER_NOTE["note"]
    note = None
    try:
        out = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True,
                             timeout=10).stdout
        for line in out.splitlines():
            if "llama-server" in line and "-m " in line:
                if "--spec-type draft" in line or "--draft" in line:
                    note = ("draft MTP actif sur le serveur : les scores HumanEval peuvent être "
                            "corrompus (~15 % historiquement) — sans draft pour un run officiel")
                break
    except Exception:
        note = None
    _SERVER_NOTE.update({"at": now, "note": note})
    return note


def bench_availability(test, probe=None):
    """(disponible, raison) — contexte réellement servi + harnais présents."""
    probe = probe if probe is not None else (STATE.probe or {})
    if DRY_RUN:
        return True, None
    need = test.get("needs_ctx") or 0
    have = probe.get("n_ctx")
    if need and have and have < need:
        return False, (f"serveur à ctx {have} — il faut ≥ {need} "
                       f"(recharge le modèle avec -c {need})")
    if test["kind"] == "batterie":
        if not llm_is_local():
            return False, f"harnais local : LLM_BASE doit être 127.0.0.1 (ici {LLM_BASE})"
        script = os.path.join(BENCH_LLM_HOME, test["script"])
        if not os.path.exists(script):
            return False, f"harnais absent : {script}"
    if test["kind"] == "humaneval":
        if not BENCH_HE_SSH or not BENCH_HE_DIR:
            return False, "harnais HumanEval non configuré (BENCH_HE_SSH / BENCH_HE_DIR)"
        ok, why = he_remote_ok()
        if not ok:
            return False, why
    return True, None


def bench_catalog(probe=None):
    """Catalogue prêt pour l'UI : estimation, provenance, disponibilité, dernier chiffre."""
    probe = probe if probe is not None else (STATE.probe or {})
    entry = bench_calib_entry(probe)
    return bench_tests.catalog(entry, availability=lambda t: bench_availability(t, probe))


def make_bench_result(test, est_s):
    return {"id": test["id"], "group": test["group"], "title": test["title"],
            "status": "pending", "est_s": est_s, "duree_s": None, "summary": None,
            "erreur": None, "metrics": {}, "log": f"{test['id']}.log", "vram_peak": None,
            "started_at": None, "finished_at": None}


def any_worker_running():
    with STATE.lock:
        return bool((STATE.worker and STATE.worker.is_alive())
                    or (STATE.bench_worker and STATE.bench_worker.is_alive()))


def write_bench_json():
    """Écrit bench.json + bench.md — appelé après CHAQUE test (un run coupé reste exploitable)."""
    path = STATE._bench_json_path
    if not path:
        return
    with STATE.lock:
        data = json.loads(json.dumps(STATE.bench or {}))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        (path.parent / "bench.md").write_text(bench_tests.render_md(data), encoding="utf-8")
    except OSError as exc:
        LOG.add(f"écriture du bench impossible : {exc}")


def bench_worker(ids, declared_settings=None):
    """Exécute les tests choisis, un par un. Un test en échec n'arrête pas la suite."""
    try:
        if DRY_RUN:
            model_path, model_slug, n_ctx = "dry-run", "dry-run", DRY_N_CTX
        else:
            try:
                model_path, n_ctx = fetch_model_identity()
                model_slug = model_slug_from_path(model_path)
            except Exception as exc:
                LOG.add(f"ERREUR tests : identité LLM indisponible ({exc})")
                with STATE.lock:
                    for r in STATE.bench["results"]:
                        r["status"] = "error"
                        r["erreur"] = f"LLM injoignable ({exc})"
                    STATE.bench["status"] = "error"
                    STATE.bench["finished_at"] = now_iso()
                write_bench_json()
                return

        label = bench_label(model_slug, n_ctx)
        base = datetime.now().strftime("%Y-%m-%d_%H%M") + f"__{label}__tests"
        run_dir = BENCH_DIR / base
        n = 2
        while run_dir.exists():
            run_dir = BENCH_DIR / f"{base}-{n}"
            n += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            port = urlparse(LLM_BASE).port or 8080
        except Exception:
            port = 8080
        # réglages réellement servis, capturés au lancement du bench
        captured = capture_settings(declared_settings)
        variant = variant_of(captured)
        LOG.add(f"réglages du bench ({captured.get('source')}) : "
                f"{captured.get('variant_label')}")
        env = bench_tests.BenchEnv(
            base=LLM_BASE, port=port, label=label, run_dir=str(run_dir),
            local=llm_is_local(), dry=DRY_RUN, ssh=BENCH_HE_SSH, he_dir=BENCH_HE_DIR,
            he_results=BENCH_HE_RESULTS, llm_home=BENCH_LLM_HOME, log_src=BENCH_LOG_SRC,
            model_slug=model_slug, model_path=model_path, n_ctx=n_ctx,
            settings=captured)
        with STATE.lock:
            STATE._bench_json_path = run_dir / "bench.json"
            STATE.bench.update({
                "run_id": run_dir.name, "run_dir": str(run_dir),
                "run_url": f"/benches/{run_dir.name}/", "label": label,
                "model_path": model_path, "model_slug": model_slug, "n_ctx": n_ctx,
                "settings": captured,          # réglages réels du serveur
                "variant": variant,
                "status": "running",
            })
        LOG.add(f"bench {run_dir.name} — modèle {model_slug} (ctx {n_ctx})")
        LIVE.clear_if_other_run(run_dir.name)
        LIVE.begin(run_dir.name, "tests", "Tests", label, mode="tests")

        for tid in ids:
            with STATE.lock:
                if STATE.bench_stop:
                    LOG.add("tests : stop demandé — les tests restants ne sont pas lancés")
                    break
                res = next((r for r in STATE.bench["results"] if r["id"] == tid), None)
                test = bench_tests.TESTS_BY_ID.get(tid)
                if res is None or test is None:
                    continue
                res["status"] = "running"
                res["started_at"] = now_iso()
                LIVE.set_status("running")

            def emit(line, _tid=tid):
                LOG.add(f"[{_tid}] {line}")
                LIVE.append("content", str(line) + "\n")

            out = bench_tests.run_test(test, env, emit, lambda: STATE.bench_stop)
            with STATE.lock:
                res.update({k: out.get(k) for k in ("status", "metrics", "summary", "erreur",
                                                    "duree_s", "log", "vram_peak")})
                res["finished_at"] = now_iso()
            calib = bench_calib()
            key = bench_tests.calib_key(model_slug, n_ctx)
            if out.get("calib"):
                bench_tests.update_calib(calib, key, out["calib"],
                                         last_summary=out.get("summary"), test_id=tid)
                bench_tests.save_calib(str(BENCH_CALIB_FILE), calib)
            write_bench_json()

        with STATE.lock:
            results = STATE.bench["results"]
            if STATE.bench_stop:
                for r in results:
                    if r["status"] == "pending":
                        r["status"] = "skipped"
                STATE.bench["status"] = "stopped"
            elif any(r["status"] == "error" for r in results):
                STATE.bench["status"] = "partial"
            else:
                STATE.bench["status"] = "finished"
            STATE.bench["finished_at"] = now_iso()
        write_bench_json()
        LOG.add(f"tests terminés — {STATE.bench['run_id']} ({STATE.bench['status']})")
        LIVE.finish("ok")

    except Exception as exc:  # filet : jamais de traceback, jamais de bench perdu
        LOG.add(f"ERREUR worker tests : {exc!r}")
        with STATE.lock:
            if STATE.bench:
                STATE.bench["status"] = "error"
                STATE.bench["finished_at"] = now_iso()
        write_bench_json()
    finally:
        with STATE.lock:
            STATE.bench_stop = False
            STATE.bench_worker = None
        probe_llm(force=True)


def start_bench(ids, declared_settings=None):
    ids = [str(i) for i in ids]
    with STATE.lock:
        if any_worker_running():
            return False, "un traitement est déjà en cours (duel ou tests)"
        tests = [bench_tests.TESTS_BY_ID[i] for i in ids if i in bench_tests.TESTS_BY_ID]
        if not tests:
            return False, "sélection vide"
        probe = dict(STATE.probe or {})
        cat = {c["id"]: c for c in bench_catalog(probe)}
        STATE.bench_stop = False
        STATE.bench = {
            "run_id": None, "run_dir": None, "run_url": None,
            "started_at": now_iso(), "finished_at": None,
            "endpoint": LLM_BASE, "model_path": None, "model_slug": None, "n_ctx": None,
            "label": None, "dry_run": DRY_RUN, "app_version": APP_VERSION,
            "python_version": sys.version.split()[0],
            "settings": None,              # rempli par le worker (détection réelle)
            "variant": None,
            "vram_note": bench_tests.vram_now() if llm_is_local() else None,
            "notes": llm_server_note(),
            "selected_ids": ids, "status": "running",
            "est_total_s": round(sum((cat.get(i) or {}).get("est_s") or 0 for i in ids), 1),
            "results": [make_bench_result(t, (cat.get(t["id"]) or {}).get("est_s"))
                        for t in tests],
        }
        worker = threading.Thread(target=bench_worker, args=(ids, declared_settings),
                                  name="prompt-duel-bench", daemon=True)
        STATE.bench_worker = worker
        worker.start()
    LOG.add(f"tests lancés — {len(ids)} test(s) : {ids}")
    return True, "tests lancés"


def list_past_benches(limit=50):
    """Sessions de tests passées (les plus récentes d'abord) — lues dans bench.json."""
    if not BENCH_DIR.exists():
        return []
    out = []
    try:
        dirs = sorted([p for p in BENCH_DIR.iterdir() if p.is_dir()], key=lambda p: p.name,
                      reverse=True)
    except OSError:
        return []
    for d in dirs[:max(1, limit)]:
        meta = d / "bench.json"
        if not meta.exists():
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        res = data.get("results") or []
        counts = {}
        for r in res:
            counts[r.get("status") or "?"] = counts.get(r.get("status") or "?", 0) + 1
        out.append({
            "run_id": data.get("run_id") or d.name,
            "url": f"/benches/{d.name}/",
            "model_slug": data.get("model_slug"),
            "n_ctx": data.get("n_ctx"),
            "label": data.get("label"),
            "started_at": data.get("started_at"),
            "finished_at": data.get("finished_at"),
            "status": data.get("status"),
            "tests": [(r.get("id"), r.get("summary")) for r in res],
            "resumes": [r.get("summary") for r in res if r.get("summary")],
            "counts": counts,
        })
    return out


def build_state(force_probe=False, params=None):
    params = normalize_params(params)
    probe = probe_llm(force=force_probe)
    # réglages détectés maintenant (mis en cache : /api/state est pollé 1×/s)
    settings_now = detect_settings(force=force_probe)
    with STATE.lock:
        run = None
        if STATE.run:
            run = json.loads(json.dumps(STATE.run))  # copie profonde
            results = run.get("results") or []
            done = sum(1 for r in results if r.get("status") in TERMINAL_STATUSES)
            run["progress"] = {"done": done, "total": len(results)}
            run["current"] = next((r["id"] for r in results if r.get("status") == "running"), None)
            run["is_running"] = bool(STATE.worker is not None and STATE.worker.is_alive())
        is_running = bool(STATE.worker is not None and STATE.worker.is_alive())
        stop_requested = STATE.stop_requested
        payload = {
            "app_version": APP_VERSION,
            "prompts_version": STATE.prompts_version,
            "dry_run": DRY_RUN,
            "stream": STREAM,
            "live_poll_ms": LIVE_POLL_MS,
            "llm_base": LLM_BASE,
            "runs_dir": str(RUNS_DIR),
            "llm": dict(probe),
            "settings": settings_now,            # réglages réellement servis (v1.6)
            "variant": variant_of(settings_now),  # variante correspondante
            "groups": json.loads(json.dumps(STATE.groups)),
            "prompts": [dict(p) for p in STATE.prompts],
            "already_done": list_already_done(model_slug=probe.get("model_slug"),
                                              n_ctx=probe.get("n_ctx"),
                                              params=params),
            "params": params,                    # réglages demandés par l'UI (badges « déjà fait »)
            "default_params": dict(DEFAULT_PARAMS),
            "run": run,
            "is_running": is_running,
            "stop_requested": stop_requested,
            "runs": list_past_runs(),
        }
    # hors du verrou : le catalogue des tests sonde la machine d'à côté (ssh, 60 s de cache)
    payload["tests"] = build_tests_state(probe)
    return payload


def build_tests_state(probe=None):
    """Bloc « tests » de /api/state : catalogue + estimation + session en cours."""
    probe = probe if probe is not None else (STATE.probe or {})
    with STATE.lock:
        bench = json.loads(json.dumps(STATE.bench)) if STATE.bench else None
        if bench:
            results = bench.get("results") or []
            done = sum(1 for r in results if r.get("status") in
                       ("ok", "partial", "error", "skipped"))
            est_total = bench.get("est_total_s") or 0
            est_done = sum((r.get("est_s") or 0) for r in results
                           if r.get("status") in ("ok", "partial", "error", "skipped"))
            real_done = sum((r.get("duree_s") or 0) for r in results)
            bench["progress"] = {"done": done, "total": len(results),
                                 "est_done_s": round(est_done, 1),
                                 "est_left_s": round(max(0.0, est_total - est_done), 1),
                                 "real_done_s": round(real_done, 1)}
            bench["current"] = next((r["id"] for r in results if r.get("status") == "running"),
                                    None)
            bench["is_running"] = bool(STATE.bench_worker and STATE.bench_worker.is_alive())
        is_running = bool(STATE.bench_worker and STATE.bench_worker.is_alive())
    return {
        "groups": [dict(g) for g in bench_tests.GROUPS],
        "catalog": bench_catalog(probe),
        "run": bench,
        "is_running": is_running,
        "stop_requested": STATE.bench_stop,
        "server_note": llm_server_note(),
        "vram_now": bench_tests.vram_now() if llm_is_local() else None,
        "benches": list_past_benches(20),
        "calib_key": bench_tests.calib_key(probe.get("model_slug"), probe.get("n_ctx")),
    }


def render_tests_html():
    """Lignes de tests rendues côté serveur (même logique que les prompts : lisibles sans JS).

    Chaque ligne porte son `data-test` : le JS ne fait que réactualiser l'estimation,
    la disponibilité (contexte réellement servi) et le statut, sans reconstruire le DOM.
    """
    catalog = bench_catalog()
    by_group = {}
    for c in catalog:
        by_group.setdefault(c["group"], []).append(c)
    parts = []
    for g in bench_tests.GROUPS:
        items = by_group.get(g["id"]) or []
        if not items:
            continue
        rows = []
        for c in items:
            dis = "" if c["available"] else " disabled"
            rows.append(
                '<label class="prompt" title="%s">'
                '<input type="checkbox" data-test="%s"%s>'
                '<div><div class="t">%s <span class="done" id="tlast-%s"></span></div>'
                '<div class="d">%s<br><span class="dim" id="test-est-%s">≈ %s (%s)</span>'
                '<span class="dim" id="test-why-%s"></span></div></div>'
                '<div class="st pending" id="tst-%s">en attente</div></label>'
                % (html.escape(c["desc"]), c["id"], dis, html.escape(c["title"]), c["id"],
                   html.escape(c["desc"]), c["id"], c["est_label"], c["est_source"],
                   c["id"], c["id"])
            )
        note = f' <span class="note">{html.escape(g["note"])}</span>' if g.get("note") else ""
        parts.append(
            f'<details class="grp" open data-tgroup="{html.escape(g["id"])}">'
            f'<summary><b>{html.escape(g["title"])}</b> · <span class="dim tcount"></span>{note}</summary>'
            '<div class="grpbar"><button data-tact="all">cocher le groupe</button>'
            '<button data-tact="none">décocher le groupe</button></div>'
            f'<div class="grprows">{"".join(rows)}</div></details>'
        )
    return "\n".join(parts)


def render_prompts_html():
    """Sections de groupes rendues côté serveur.

    Avantages : la page est lisible même sans JS, les titres de groupe sont dans le
    HTML servi, et les cellules de statut gardent exactement leurs ids st1…stN
    (le rafraîchissement JS s'appuie dessus).
    """
    by_group = {}
    for p in STATE.prompts:
        by_group.setdefault(p["group"], []).append(p)
    parts = []
    for g in STATE.groups:
        items = by_group.get(g["id"]) or []
        if not items:
            continue
        rows = []
        for p in items:
            rows.append(
                '<label class="prompt"><input type="checkbox" data-id="%d">'
                '<div><div class="t">%d. %s <span class="tag">%s</span>'
                ' <span class="done" id="done%d"></span></div>'
                '<div class="d">%s…</div></div>'
                '<div class="st pending" id="st%d">en attente</div></label>'
                % (p["id"], p["id"], html.escape(p["title"]), html.escape(p["category"]),
                   p["id"], html.escape((p["prompt"] or "")[:160]), p["id"])
            )
        note = f' <span class="note">{html.escape(g["note"])}</span>' if g.get("note") else ""
        parts.append(
            f'<details class="grp" open data-group="{html.escape(g["id"])}">'
            f'<summary><b>{html.escape(g["title"])}</b> · <span class="dim gcount"></span>{note}</summary>'
            '<div class="grpbar"><button data-gact="all">cocher le groupe</button>'
            '<button data-gact="none">décocher le groupe</button></div>'
            f'<div class="grprows">{"".join(rows)}</div></details>'
        )
    return "\n".join(parts)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prompt Duel 224</title>
<style>
:root{--bg:#12141a;--card:#1b1e26;--fg:#e6e8ee;--dim:#98a0b3;--acc:#4da3ff;--ok:#2ecc71;--warn:#f39c12;--err:#e74c3c;--line:#2a2f3a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:12px 18px;background:#0f1116;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:600}
h2{font-size:12px;margin:0 0 8px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim)}
main{max-width:1080px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}
.badge{padding:5px 10px;border-radius:999px;font-weight:600;font-size:13px;border:1px solid transparent}
.badge.ok{background:rgba(46,204,113,.14);color:var(--ok);border-color:rgba(46,204,113,.4)}
.badge.busy{background:rgba(243,156,18,.14);color:var(--warn);border-color:rgba(243,156,18,.4)}
.badge.off{background:rgba(231,76,60,.14);color:var(--err);border-color:rgba(231,76,60,.4)}
.badge.dry{background:rgba(77,163,255,.14);color:var(--acc);border-color:rgba(77,163,255,.4)}
button{background:#242a36;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:7px 12px;cursor:pointer;font:inherit}
button:hover:not(:disabled){border-color:var(--acc)}
button:disabled{opacity:.45;cursor:not-allowed}
button.primary{background:var(--acc);border-color:var(--acc);color:#08111c;font-weight:600}
button.danger{background:transparent;border-color:var(--err);color:var(--err)}
.row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.spacer{flex:1}
.dim{color:var(--dim)}
.prompt{display:grid;grid-template-columns:24px 1fr auto;gap:10px;padding:8px;border-radius:8px;align-items:start;cursor:pointer}
.prompt:hover{background:#20242e}
.prompt .t{font-weight:600}
.prompt .d{color:var(--dim);font-size:12.5px}
.prompt input{margin-top:3px}
.tag{font-size:11px;color:var(--dim);border:1px solid var(--line);border-radius:999px;padding:1px 7px}
.st{font-size:12.5px;white-space:nowrap;text-align:right}
.st.ok{color:var(--ok)}.st.no_html{color:var(--warn)}.st.error{color:var(--err)}
.st.pending,.st.skipped{color:var(--dim)}.st.running{color:var(--acc)}.st.deja_genere{color:#b58cff}
.bar{height:8px;background:#0d0f14;border-radius:99px;overflow:hidden;margin:10px 0}
#barfill{height:100%;width:0;background:var(--acc);transition:width .3s}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre#log{margin:0;max-height:260px;overflow:auto;font-size:12px;color:#c7cede;white-space:pre-wrap}
.path{background:#0d0f14;border:1px solid var(--line);border-radius:6px;padding:3px 7px;font-size:12.5px;word-break:break-all}
table{width:100%;border-collapse:collapse}
td,th{padding:5px 7px;border-bottom:1px solid var(--line);text-align:left;font-size:13px}
th{color:var(--dim);font-weight:600;font-size:12px}
a{color:var(--acc)}
#msg{color:var(--err);font-size:13px}
details.grp{border:1px solid var(--line);border-radius:8px;margin-bottom:10px;background:#181b22}
details.grp>summary{cursor:pointer;padding:9px 11px;font-size:13.5px;user-select:none}
details.grp>summary .note{color:var(--dim);font-size:12px;font-weight:400}
.grpbar{display:flex;gap:8px;padding:0 11px 8px}
.grpbar button{padding:3px 9px;font-size:12px}
.grprows{padding:0 6px 8px}
.done{color:#b58cff;font-size:11.5px}
/* --- suivi en direct (v1.2) --- */
.livebadge{font-size:11.5px;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--dim)}
.livebadge.on{color:var(--acc);border-color:rgba(77,163,255,.5);background:rgba(77,163,255,.10)}
.livebadge.fin{color:var(--ok);border-color:rgba(46,204,113,.4)}
.livebadge.err{color:var(--err);border-color:rgba(231,76,60,.4)}
pre#live{margin:0;max-height:360px;overflow:auto;font-size:12px;color:#c7cede;white-space:pre-wrap;word-break:break-word;background:#0d0f14;border:1px solid var(--line);border-radius:8px;padding:8px}
pre#live.big{max-height:none}
pre#live-reasoning{margin:0;max-height:180px;overflow:auto;font-size:12px;color:#b58cff;white-space:pre-wrap;word-break:break-word;background:#0d0f14;border:1px solid var(--line);border-radius:8px;padding:8px}
#livepreviewwrap{margin-top:10px}
#liveframe{width:100%;height:520px;border:1px solid var(--line);border-radius:8px;background:#fff}
label.chk{display:flex;align-items:center;gap:6px;font-size:12.5px;color:var(--dim);cursor:pointer}
details.rz>summary{cursor:pointer;font-size:12.5px;color:var(--dim);margin-bottom:6px}
/* --- réglages du run + comparaison (v1.3) --- */
#settings{align-items:center;margin:8px 0;font-size:12.5px}
#settings input[type=number]{width:88px;background:#0d0f14;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 6px;font:inherit}
#settings input[type=checkbox]{accent-color:var(--acc)}
select{background:#0d0f14;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 6px;font:inherit;max-width:360px}
#cmp-table table{font-size:12.5px;white-space:nowrap}
#cmp-table th{padding:4px 6px}
#cmp-table th.gA{color:var(--acc);text-align:center;border-bottom:1px solid rgba(77,163,255,.35)}
#cmp-table th.gB{color:#b58cff;text-align:center;border-bottom:1px solid rgba(181,140,255,.35)}
#cmp-table td{padding:4px 6px}
#cmp-table td.n{text-align:right;font-variant-numeric:tabular-nums}
#cmp-table tr.miss td{opacity:.55}
#cmp-table .dot{font-size:15px;line-height:1}
#cmp-table .d-win{color:var(--ok)}
#cmp-table .d-lose{color:var(--err)}
#cmp-table .d-zero{color:var(--dim)}
.cmpbtn{padding:2px 7px;font-size:11.5px}
.navlink{color:var(--acc);text-decoration:none;font-size:12.5px;border:1px solid var(--line);border-radius:6px;padding:4px 8px}
.navlink:hover{background:#171c26}
</style>
</head>
<body>
<header>
  <h1>__APP_TITLE__ <span class="dim">· lanceur de prompts vers un LLM local</span></h1>
  <div id="llm" class="badge off">état LLM…</div>
  <button id="retest">Re-tester</button>
  <div class="spacer"></div>
  <a class="navlink" href="/benchmarks">📊 Benchmarks</a>
  <span class="dim" id="ver">v__APP_VERSION__</span>
</header>
<main>
  <section class="card">
    <div class="row">
      <h2>Prompts</h2>
      <button id="all">Tout cocher</button>
      <button id="none">Tout décocher</button>
      <div class="spacer"></div>
      <span class="dim" id="count">0 / 0 sélectionné(s)</span>
    </div>
    <div id="prompts">__GROUPS_HTML__</div>
  </section>

  <section class="card">
    <div class="row">
      <button id="launch" class="primary" disabled>Lancer la sélection</button>
      <button id="stop" class="danger" disabled>Stop</button>
      <div class="spacer"></div>
      <span id="prog" class="dim">—</span>
    </div>
    <div class="row" id="settings">
      <span class="dim">Réglages du run :</span>
      <label class="chk">température <input type="number" id="set-temp" step="0.05" min="0" max="2" value="0.2"></label>
      <label class="chk">max_tokens <input type="number" id="set-mt" step="1024" min="64" value="16384"></label>
      <label class="chk" title="Raisonnement du modèle (chat_template_kwargs.enable_thinking). Sur ce build, l'activer peut envoyer la réponse dans reasoning_content : le texte part alors dans le panneau raisonnement et le HTML peut manquer.">raisonnement <input type="checkbox" id="set-think"></label>
      <button id="set-reset">défauts</button>
      <span id="set-note" class="dim"></span>
    </div>
    <div class="bar"><div id="barfill"></div></div>
    <div id="msg"></div>
    <div id="runinfo" class="dim">Aucun run pour l'instant.</div>
  </section>

  <section class="card">
    <div class="row">
      <h2>Tests — vitesse · mémoire · intelligence</h2>
      <button id="t-all">Tout cocher</button>
      <button id="t-none">Tout décocher</button>
      <div class="spacer"></div>
      <span id="t-count" class="dim">0 / 0 sélectionné(s)</span>
    </div>
    <div id="tests">__TESTS_HTML__</div>
    <div class="row" style="margin-top:10px">
      <button id="t-launch" class="primary" disabled>Lancer les tests</button>
      <button id="t-stop" class="danger" disabled>Stop</button>
      <div class="spacer"></div>
      <span id="t-est" class="dim">—</span>
    </div>
    <div class="bar"><div id="t-barfill"></div></div>
    <div id="t-msg"></div>
    <div id="t-info" class="dim">Aucun test lancé pour l'instant — coche des tests puis « Lancer les tests ».</div>
    <div id="t-results" style="margin-top:8px"><span class="dim">—</span></div>
    <details class="rz" id="t-past" style="margin-top:10px">
      <summary>tests passés (<span id="t-pastn">0</span>)</summary>
      <div id="t-pastbox"><span class="dim">—</span></div>
    </details>
  </section>

  <section class="card">
    <div class="row">
      <h2>Comparer deux runs</h2>
      <span class="dim">même prompt, deux modèles ou deux réglages</span>
      <div class="spacer"></div>
      <a id="cmp-page" class="dim" href="#" target="_blank" rel="noopener">ouvrir la page côte à côte ↗</a>
    </div>
    <div class="row">
      <label class="chk">A (référence) <select id="cmp-a"></select></label>
      <label class="chk">B (comparé) <select id="cmp-b"></select></label>
      <button id="cmp-swap">inverser A/B</button>
      <button id="cmp-cur">B = run courant</button>
      <button id="cmp-load" class="primary">Comparer</button>
      <label class="chk"><input type="checkbox" id="cmp-both"> seulement les prompts présents des deux côtés</label>
    </div>
    <div id="cmp-summary" class="dim" style="margin:8px 0">Choisis deux runs puis « Comparer ».</div>
    <div id="cmp-table" style="overflow-x:auto"><span class="dim">—</span></div>
  </section>

  <section class="card">
    <div class="row">
      <h2>Suivi en direct</h2>
      <span id="liveslug" class="dim"></span>
      <span id="livestate" class="livebadge">au repos</span>
      <div class="spacer"></div>
      <span id="livestats" class="dim">—</span>
    </div>
    <div class="row" style="margin:8px 0">
      <label class="chk"><input type="checkbox" id="autoscroll" checked> défilement auto</label>
      <label class="chk"><input type="checkbox" id="livepreview"> aperçu HTML live (recharge ~1,5 s)</label>
      <button id="livebig">agrandir</button>
      <button id="livecopy">copier le texte</button>
    </div>
    <details class="rz" id="rbox" hidden>
      <summary>raisonnement du modèle (<span id="rchars">0</span> car.)</summary>
      <pre id="live-reasoning"></pre>
    </details>
    <pre id="live">Aucun tir en cours — le texte du LLM s'affichera ici, token par token.</pre>
    <div id="livepreviewwrap" hidden>
      <iframe id="liveframe" sandbox="allow-scripts allow-pointer-lock allow-modals"></iframe>
    </div>
  </section>

  <section class="card">
    <h2>Résultats du run</h2>
    <div id="results"><span class="dim">—</span></div>
  </section>

  <section class="card">
    <h2>Runs passés</h2>
    <div id="pastruns"><span class="dim">—</span></div>
  </section>

  <section class="card">
    <h2>Journal</h2>
    <pre id="log"></pre>
  </section>
</main>

<script>
const CATALOG = __CATALOG_JSON__;
const PROMPTS = CATALOG.prompts;
const GROUPS = CATALOG.groups;
const checked = new Set();
let logCursor = 0;
let lastState = null;

const LABEL = {pending:"en attente",running:"en cours",ok:"ok",no_html:"pas de HTML",
               error:"erreur",deja_genere:"déjà généré",skipped:"arrêté"};
const TERMINAL = ["ok","no_html","error","deja_genere","skipped"];

function esc(s){return String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function num(v,d){return (v===null||v===undefined||isNaN(v))?"—":Number(v).toFixed(d);}
function paramsLabel(p){
  p=p||{};
  return "t"+(p.temperature==null?"?":p.temperature)
    +" · "+(p.enable_thinking?"think":"nothink")
    +" · mt"+(p.max_tokens==null?"?":p.max_tokens);
}

function wireList(){
  // les sections et les lignes sont rendues côté serveur : ici on ne fait que câbler
  const box=document.getElementById("prompts");
  box.addEventListener("change",e=>{
    if(e.target && e.target.type==="checkbox"){
      const id=parseInt(e.target.dataset.id,10);
      if(e.target.checked) checked.add(id); else checked.delete(id);
      updateCount();
    }
  });
  box.addEventListener("click",e=>{
    const b=e.target.closest("button[data-gact]");
    if(!b) return;
    e.preventDefault(); e.stopPropagation();
    const det=b.closest("details.grp");
    const gid=det?det.dataset.group:null;
    const items=PROMPTS.filter(p=>p.group===gid);
    if(b.dataset.gact==="all") items.forEach(p=>checked.add(p.id));
    else items.forEach(p=>checked.delete(p.id));
    items.forEach(p=>{
      const c=document.querySelector('#prompts input[data-id="'+p.id+'"]');
      if(c) c.checked=checked.has(p.id);
    });
    updateCount();
  });
  updateCount();
}

function updateCount(){
  document.getElementById("count").textContent=checked.size+" / "+PROMPTS.length+" sélectionné(s)";
  const ad=(lastState&&lastState.already_done)||{};
  document.querySelectorAll("details.grp").forEach(det=>{
    const gid=det.dataset.group;
    const items=PROMPTS.filter(p=>p.group===gid);
    const sel=items.filter(p=>checked.has(p.id)).length;
    let txt=items.length+" prompt"+(items.length>1?"s":"")+" · "+sel+"/"+items.length+" coché"+(sel>1?"s":"");
    const nd=items.filter(p=>ad[String(p.id)]).length;
    if(nd) txt+=", "+nd+" déjà fait"+(nd>1?"s":"");
    const c=det.querySelector(".gcount");
    if(c) c.textContent=txt;
  });
  if(lastState) updateButtons(lastState);
}

function updateButtons(s){
  const running = !!(s.run && s.run.is_running);
  const tbusy = !!(s.tests && s.tests.is_running);
  document.getElementById("launch").disabled = running || tbusy || checked.size===0 || !llmUsable(s);
  document.getElementById("stop").disabled = !running;
}

function llmUsable(s){
  const st=s.llm?s.llm.status:"off";
  return st==="ok" || st==="busy" || st==="dry";
}

function renderLLM(s){
  const el=document.getElementById("llm");
  const l=s.llm||{};
  if(l.status==="dry"){el.className="badge dry";el.textContent="LLM simulé — DRY RUN (aucun appel réseau)";return;}
  if(l.status==="ok"){el.className="badge ok";el.textContent="LLM prêt — "+(l.model_slug||"?")+" · ctx "+(l.n_ctx||"?");return;}
  if(l.status==="busy"){el.className="badge busy";el.textContent="LLM prêt mais slot occupé — "+(l.model_slug||"?")+" · ctx "+(l.n_ctx||"?");return;}
  el.className="badge off";
  el.textContent="LLM éteint (llama.cpp non lancé)"+(s.llm_base?" — "+s.llm_base:"");
}

function renderStatuses(s){
  PROMPTS.forEach(p=>{
    const cell=document.getElementById("st"+p.id);
    if(!cell) return;
    cell.className="st pending";
    cell.textContent=LABEL.pending;
  });
  if(s.run && s.run.results){
    s.run.results.forEach(r=>{
      const cell=document.getElementById("st"+r.id);
      if(!cell) return;
      const st=r.status||"pending";
      cell.className="st "+st;
      let txt=LABEL[st]||st;
      const bits=[];
      if(r.duree_s!==null && r.duree_s!==undefined) bits.push(num(r.duree_s,1)+" s");
      if(r.tok_s!==null && r.tok_s!==undefined) bits.push(num(r.tok_s,1)+" tok/s");
      if(r.completion_tokens) bits.push(r.completion_tokens+" tok");
      if(bits.length) txt+=" · "+bits.join(" · ");
      if(r.url) txt+=' · <a href="'+esc(r.url)+'" target="_blank">ouvrir</a>';
      cell.innerHTML=txt;
    });
  }
}

function renderRun(s){
  const run=s.run;
  const prog=document.getElementById("prog");
  const fill=document.getElementById("barfill");
  const info=document.getElementById("runinfo");
  const box=document.getElementById("results");
  if(!run){
    prog.textContent="—";fill.style.width="0%";
    info.className="dim";info.textContent="Aucun run pour l'instant.";
    box.innerHTML='<span class="dim">—</span>';
    return;
  }
  const p=run.progress||{done:0,total:(run.results||[]).length};
  prog.textContent=p.done+" / "+p.total+(run.status?"  ("+run.status+")":"");
  fill.style.width=(p.total?(100*p.done/p.total):0)+"%";
  info.className="";
  info.innerHTML='<div class="row"><span class="dim">Dossier du run :</span> <span class="path">'+esc(run.run_dir||"?")+'</span></div>'+
    '<div class="row"><span class="dim">Modèle :</span> <code>'+esc(run.model_slug||"?")+'</code>'+
    ' <span class="dim">· ctx</span> '+esc(run.n_ctx||"?")+
    ' <span class="dim">· endpoint</span> <code>'+esc(run.endpoint||"?")+'</code>'+
    ' <span class="dim">· dry_run</span> '+esc(String(run.dry_run))+
    ' <span class="dim">· réglages</span> '+esc(run.params_label||paramsLabel(run.params))+
    ' <span class="dim">· suivi live</span> '+esc(s.stream===false?"non (LLM_STREAM=0)":"streaming SSE")+'</div>'+
    '<div class="row"><span class="dim">Début :</span> '+esc(run.started_at||"?")+
    ' <span class="dim">· fin :</span> '+esc(run.finished_at||"—")+
    (run.last_run_at && run.last_run_at!==run.started_at ? ' <span class="dim">· relancé :</span> '+esc(run.last_run_at) : '')+
    ' <span class="dim">· <a href="'+esc(run.run_url||"#")+'" target="_blank">parcourir le dossier</a></span></div>';
  const gOrder=GROUPS.map(g=>g.id);
  const gTitle={}; GROUPS.forEach(g=>{gTitle[g.id]=g.title;});
  const res=(run.results||[]).slice().sort((a,b)=>{
    const ga=gOrder.indexOf(a.group), gb=gOrder.indexOf(b.group);
    const ra=(ga<0?999:ga)-(gb<0?999:gb);
    return ra!==0?ra:(a.id-b.id);
  });
  const rows=res.map(r=>'<tr><td>'+r.id+'</td><td>'+esc(gTitle[r.group]||r.group||"—")+'</td><td>'+esc(r.title)+'</td>'+
    '<td class="st '+(r.status||"")+'">'+esc(LABEL[r.status]||r.status||"")+'</td>'+
    '<td>'+num(r.duree_s,1)+'</td><td>'+num(r.tok_s,1)+'</td>'+
    '<td>'+(r.completion_tokens==null?"—":r.completion_tokens)+'</td>'+
    '<td>'+(r.url?'<a href="'+esc(r.url)+'" target="_blank">ouvrir</a>':(r.erreur?'<span class="dim">'+esc(r.erreur)+'</span>':"—"))+'</td></tr>').join("");
  box.innerHTML='<table><thead><tr><th>#</th><th>Groupe</th><th>Titre</th><th>Statut</th><th>Durée</th><th>tok/s</th><th>tokens</th><th>Fichier</th></tr></thead><tbody>'+rows+'</tbody></table>';
}

function renderPast(s){
  const box=document.getElementById("pastruns");
  const runs=s.runs||[];
  if(!runs.length){box.innerHTML='<span class="dim">—</span>';return;}
  const rows=runs.map(r=>{
    const counts=Object.keys(r.counts||{}).map(k=>k+"="+r.counts[k]).join(" ")||"—";
    return '<tr><td><a href="'+esc(r.url)+'" target="_blank">'+esc(r.run_id)+'</a></td>'+
      '<td>'+esc(r.model_slug||"?")+'</td><td>'+esc(r.n_ctx==null?"?":r.n_ctx)+'</td>'+
      '<td>'+esc(r.started_at||"?")+'</td><td>'+esc(r.status||"?")+'</td><td class="dim">'+esc(counts)+'</td></tr>';
  }).join("");
  box.innerHTML='<table><thead><tr><th>Run</th><th>Modèle</th><th>ctx</th><th>Début</th><th>Statut</th><th>Résultats</th></tr></thead><tbody>'+rows+'</tbody></table>';
}

function renderDone(s){
  const ad=(s&&s.already_done)||{};
  PROMPTS.forEach(p=>{
    const el=document.getElementById("done"+p.id);
    if(!el) return;
    const rid=ad[String(p.id)];
    el.textContent=rid?("déjà fait · "+rid):"";
  });
}

function render(s){
  lastState=s;
  if(s.default_params) Object.assign(DEFAULT_SETTINGS,s.default_params);
  renderLLM(s);
  renderStatuses(s);
  renderDone(s);
  renderRun(s);
  renderPast(s);
  renderTests(s);
  fillRunSelects(s);
  updateCount();
}

async function refresh(force){
  const st=readSettings();
  const q="?t="+encodeURIComponent(st.temperature)+"&mt="+encodeURIComponent(st.max_tokens)
          +"&think="+(st.enable_thinking?1:0)+(force?"&refresh=1":"");
  try{
    const r=await fetch("/api/state"+q,{cache:"no-store"});
    render(await r.json());
  }catch(e){
    const el=document.getElementById("llm");
    el.className="badge off";el.textContent="app injoignable ("+e+")";
  }
}

async function pollLog(){
  try{
    const r=await fetch("/api/log?since="+logCursor,{cache:"no-store"});
    const d=await r.json();
    logCursor=d.next;
    const lines=d.lines||[];
    if(!lines.length) return;
    const pre=document.getElementById("log");
    const near=pre.scrollTop+pre.clientHeight>=pre.scrollHeight-40;
    pre.textContent+=lines.map(l=>l.t+"  "+l.msg).join("\n")+"\n";
    const all=pre.textContent.split("\n");
    if(all.length>210) pre.textContent=all.slice(-200).join("\n");
    if(near) pre.scrollTop=pre.scrollHeight;
  }catch(e){}
}

/* ---------------- réglages du run (v1.3) ----------------
   Température / max_tokens / raisonnement sont choisis ici et envoyés à
   /api/run ; le serveur les range dans le dossier du run (…) et dans run.json. */
let DEFAULT_SETTINGS={temperature:0.2,max_tokens:16384,enable_thinking:false};
const SETTINGS_KEY="promptduel.settings.v1";

function readSettings(){
  const t=parseFloat(document.getElementById("set-temp").value);
  const mt=parseInt(document.getElementById("set-mt").value,10);
  return {
    temperature: isNaN(t)?DEFAULT_SETTINGS.temperature:t,
    max_tokens: isNaN(mt)?DEFAULT_SETTINGS.max_tokens:mt,
    enable_thinking: document.getElementById("set-think").checked
  };
}
function settingsNote(){
  const s=readSettings();
  const suff="__t"+s.temperature+(s.enable_thinking?"__think":"__nothink")
             +(s.max_tokens!==DEFAULT_SETTINGS.max_tokens?("__mt"+s.max_tokens):"");
  let txt="→ dossiers de run : …"+suff;
  if(s.max_tokens!==DEFAULT_SETTINGS.max_tokens) txt+=" (max_tokens non standard → run distinct)";
  if(s.enable_thinking) txt+=" ⚠️ raisonnement activé : le HTML peut partir dans reasoning_content";
  document.getElementById("set-note").textContent=txt;
}
function saveSettings(s){
  try{ localStorage.setItem(SETTINGS_KEY,JSON.stringify(s)); }catch(e){}
  settingsNote();
}
function loadSettings(){
  let v=null;
  try{ v=JSON.parse(localStorage.getItem(SETTINGS_KEY)||"null"); }catch(e){ v=null; }
  const s=Object.assign({},DEFAULT_SETTINGS,v||{});
  document.getElementById("set-temp").value=s.temperature;
  document.getElementById("set-mt").value=s.max_tokens;
  document.getElementById("set-think").checked=!!s.enable_thinking;
  settingsNote();
}

/* ---------------- comparaison de deux runs (v1.3) ---------------- */
const CMP={sig:null,data:null,runs:[]};

function shortModel(m){
  m=String(m||"?");
  return m.length>28?(m.slice(0,26)+"…"):m;
}
function runLabel(r){
  const d=String(r.started_at||r.run_id||"?").slice(0,16).replace("T"," ");
  const n=(r.selected_ids||[]).length;
  const c=r.counts||{};
  // ⚠️ Ne JAMAIS agréger no_html (ou deja_genere) dans « ok » : une page tronquée
  // (finish_reason=length) n'est pas une page réussie. C'est précisément ce qui
  // distingue deux modèles — un compteur qui les additionne efface la mesure.
  const ok=(c.ok||0);
  const nh=(c.no_html||0);
  const err=(c.error||0);
  const deja=(c.deja_genere||0);
  let s=d+" · "+shortModel(r.model_slug)+" · ctx"+(r.n_ctx==null?"?":r.n_ctx)
        +" · "+(r.params_label||"")+" · "+ok+"/"+(n||"?")+" ok";
  const extra=[];
  if(nh) extra.push(nh+" pas de HTML");
  if(err) extra.push(err+" erreur"+(err>1?"s":""));
  if(deja) extra.push(deja+" deja genere");
  if(extra.length) s+=" · "+extra.join(" · ");
  return s;
}
async function fetchRuns(){          // liste complète (au-delà des 30 du polling)
  try{
    const r=await fetch("/api/runs?limit=500",{cache:"no-store"});
    const d=await r.json();
    CMP.runs=d.runs||[];
    fillRunSelects();
  }catch(e){}
}
function fillRunSelects(s){
  const runs=CMP.runs.length?(CMP.runs):((s&&s.runs)||[]);
  const sig=runs.map(r=>r.run_id).join("|");
  if(sig===CMP.sig) return;
  CMP.sig=sig;
  const vals=runs.map(r=>r.run_id);
  [["cmp-a",1],["cmp-b",0]].forEach(([id,fallback])=>{
    const sel=document.getElementById(id);
    const old=sel.value;
    sel.innerHTML=runs.map(r=>'<option value="'+esc(r.run_id)+'">'+esc(runLabel(r))+'</option>').join("");
    if(vals.indexOf(old)>=0) sel.value=old;
    else if(vals.length) sel.value=vals[Math.min(fallback,vals.length-1)];
  });
  if(!CMP.autoLoaded && vals.length>=2){
    // première fois qu'on connaît les runs : on montre tout de suite la comparaison
    // des deux plus récents, plutôt que de laisser un tableau vide à cliquer.
    CMP.autoLoaded=true;
    loadCompare();
  }
}
function cmpDelta(v,lowerIsBetter){
  if(v===null||v===undefined||isNaN(v)) return '<span class="d-zero">—</span>';
  const cls=v===0?"d-zero":((lowerIsBetter?v<0:v>0)?"d-win":"d-lose");
  return '<span class="'+cls+'">'+(v>0?"+":"")+Number(v).toFixed(1)+'</span>';
}
function cmpDeltaPct(v,lowerIsBetter){
  if(v===null||v===undefined||isNaN(v)) return '<span class="d-zero">—</span>';
  const cls=v===0?"d-zero":((lowerIsBetter?v<0:v>0)?"d-win":"d-lose");
  return '<span class="'+cls+'">'+(v>0?"+":"")+Number(v).toFixed(1)+'%</span>';
}
function cmpCellA(r,side){
  const x=r[side];
  if(!x) return '<td class="dot" title="absent">·</td><td class="n">—</td><td class="n">—</td><td class="n">—</td>';
  const st=String(x.status||"?");
  const col={ok:"var(--ok)",no_html:"var(--warn)",error:"var(--err)",deja_genere:"#b58cff"}[st]||"var(--dim)";
  const dot='<span class="dot" style="color:'+col+'" title="'+esc(st)+(x.erreur?(" — "+esc(x.erreur)):"")+'">●</span>';
  const link=x.open_url?(' <a href="'+esc(x.open_url)+'" target="_blank" rel="noopener" title="ouvrir le fichier">↗</a>'):"";
  return '<td>'+dot+link+'</td>'
    +'<td class="n">'+(x.duree_s==null?"—":num(x.duree_s,1))+'</td>'
    +'<td class="n">'+(x.tok_s==null?"—":num(x.tok_s,1))+'</td>'
    +'<td class="n" title="prompt '+(x.prompt_tokens==null?"?":x.prompt_tokens)+' tok">'
    +(x.completion_tokens==null?"—":x.completion_tokens)+'</td>';
}
function renderCompare(){
  const d=CMP.data;
  const box=document.getElementById("cmp-table");
  if(!d){box.innerHTML='<span class="dim">—</span>';return;}
  const onlyBoth=document.getElementById("cmp-both").checked;
  const rows=(d.rows||[]).filter(r=>!onlyBoth||r.in_both);
  const bothTok=rows.filter(r=>r.a&&r.b&&r.a.tok_s!=null&&r.b.tok_s!=null);
  const dts=bothTok.map(r=>r.delta.tok_s);
  const faster=dts.filter(v=>v>0).length, slower=dts.filter(v=>v<0).length;
  const mean=dts.length?(dts.reduce((a,b)=>a+b,0)/dts.length):null;
  document.getElementById("cmp-summary").innerHTML=
    '<b>A</b> '+esc(runLabel(d.a))+' <span class="dim">vs</span> <b>B</b> '+esc(runLabel(d.b))
    +'<br>'+rows.length+' prompt(s) comparé(s) · '+bothTok.length+' avec tok/s des deux côtés'
    +(mean==null?"":" · B plus rapide sur "+faster+", plus lent sur "+slower
      +" · Δ moyen "+(mean>0?"+":"")+num(mean,1)+" tok/s");
  const head='<table><thead><tr>'
    +'<th rowspan="2">#</th><th rowspan="2">Titre</th>'
    +'<th class="gA" colspan="4">A · '+esc(shortModel(d.a.model_slug))+'</th>'
    +'<th class="gB" colspan="4">B · '+esc(shortModel(d.b.model_slug))+'</th>'
    +'<th colspan="4">écart (B−A)</th><th rowspan="2"></th></tr>'
    +'<tr><th class="gA">état</th><th class="gA">durée</th><th class="gA">tok/s</th><th class="gA">tokens</th>'
    +'<th class="gB">état</th><th class="gB">durée</th><th class="gB">tok/s</th><th class="gB">tokens</th>'
    +'<th>durée s</th><th>tok/s</th><th>durée %</th><th>tok/s %</th></tr></thead><tbody>';
  const body=rows.map(r=>{
    const t=esc(r.title||("prompt "+r.id));
    const g=esc(r.group||"");
    const link='/compare?a='+encodeURIComponent(d.a.run_id)+'&b='+encodeURIComponent(d.b.run_id)
               +'&ids='+r.id;
    return '<tr'+(r.in_both?"":' class="miss"')+'><td>'+r.id+'</td>'
      +'<td title="'+esc(g)+'">'+t+'</td>'
      +cmpCellA(r,"a")+cmpCellA(r,"b")
      +'<td class="n">'+cmpDelta(r.delta.duree_s,true)+'</td>'
      +'<td class="n">'+cmpDelta(r.delta.tok_s,false)+'</td>'
      +'<td class="n">'+cmpDeltaPct(r.delta.duree_pct,true)+'</td>'
      +'<td class="n">'+cmpDeltaPct(r.delta.tok_s_pct,false)+'</td>'
      +'<td><a class="cmpbtn" href="'+link+'" target="_blank" rel="noopener">voir les 2</a></td></tr>';
  }).join("");
  box.innerHTML=head+body+'</tbody></table>';
}
async function loadCompare(){
  const a=document.getElementById("cmp-a").value;
  const b=document.getElementById("cmp-b").value;
  const summary=document.getElementById("cmp-summary");
  document.getElementById("cmp-page").href="/compare?a="+encodeURIComponent(a)+"&b="+encodeURIComponent(b);
  if(!a||!b){summary.textContent="Choisis deux runs.";return;}
  if(a===b){summary.textContent="A et B sont le même run — choisis deux runs différents.";}
  try{
    const r=await fetch("/api/compare?a="+encodeURIComponent(a)+"&b="+encodeURIComponent(b),{cache:"no-store"});
    const d=await r.json();
    if(!r.ok||d.error){
      summary.textContent=d.error||("HTTP "+r.status);
      CMP.data=null;renderCompare();return;
    }
    CMP.data=d;renderCompare();
  }catch(e){summary.textContent="Erreur réseau : "+e;}
}

/* ---------------- tests vitesse / mémoire / intelligence (v1.5) ----------------
   Chaque test est un workload du serveur (bench_tests.py) : vitesse prefill/gen,
   needle 128k/256k, batterie FR, HumanEval. L'UI coche, annonce la DURÉE ESTIMÉE
   (mesures déjà faites sur ce modèle si elles existent, sinon valeurs par défaut)
   et affiche les chiffres mesurés + le pic VRAM. */
const TCAT=new Map();          // id -> entrée du catalogue (fourni par le serveur)
const tchecked=new Set();
const TLABEL={pending:"en attente",running:"en cours",ok:"ok",partial:"partiel",
              error:"erreur",skipped:"arrêté"};

function dur(s){
  if(s===null||s===undefined||isNaN(s)) return "—";
  s=Math.round(s);
  if(s<90) return s+" s";
  if(s<5400) return Math.round(s/60)+" min";
  return (s/3600).toFixed(1)+" h";
}

function tTotals(){
  let est=0,mes=0,def=0,unavail=0;
  tchecked.forEach(id=>{
    const c=TCAT.get(id); if(!c) return;
    if(!c.available){unavail++;return;}
    est+=c.est_s;
    if(c.est_source==="mesuré") mes++; else def++;
  });
  return {est:est,mes:mes,def:def,unavail:unavail,n:tchecked.size};
}

function tUpdateEstimate(){
  const t=tTotals();
  document.getElementById("t-count").textContent=t.n+" / "+TCAT.size+" sélectionné(s)";
  const bits=[];
  if(t.n) bits.push("sélection : "+t.n+" test"+(t.n>1?"s":""));
  if(t.est>0) bits.push("≈ "+dur(t.est));
  if(t.mes&&!t.def) bits.push("d'après des mesures sur ce modèle");
  else if(t.def) bits.push("estimation par défaut ("+t.def+" test"+(t.def>1?"s":"")+")");
  if(t.unavail) bits.push("⚠️ "+t.unavail+" indisponible(s), non compté(s)");
  document.getElementById("t-est").textContent=bits.length?bits.join(" · "):"—";
  document.querySelectorAll("#tests details.grp").forEach(det=>{
    const gid=det.dataset.tgroup;
    const items=Array.from(TCAT.values()).filter(c=>c.group===gid);
    const sel=items.filter(c=>tchecked.has(c.id));
    const est=sel.reduce((a,c)=>a+(c.available?c.est_s:0),0);
    const el=det.querySelector(".tcount");
    if(el) el.textContent=items.length+" test"+(items.length>1?"s":"")+" · "+sel.length+" coché(s)"
      +(sel.length?" · ≈ "+dur(est):"");
  });
  if(lastState) updateButtons(lastState);
}

function renderTestRun(s){
  const T=s.tests||{};
  const run=T.run;
  const running=!!T.is_running;
  document.getElementById("t-launch").disabled = running || tchecked.size===0
      || !llmUsable(s) || !!(s.run && s.run.is_running);
  document.getElementById("t-stop").disabled = !running;
  const bar=document.getElementById("t-barfill");
  const info=document.getElementById("t-info");
  const box=document.getElementById("t-results");
  if(run){
    (run.results||[]).forEach(r=>{
      const cell=document.getElementById("tst-"+r.id);
      if(!cell) return;
      const st=r.status||"pending";
      cell.className="st "+st;
      let txt=TLABEL[st]||st;
      const bits=[];
      if(r.duree_s!==null&&r.duree_s!==undefined) bits.push(num(r.duree_s,1)+" s");
      else if(r.est_s!==null&&r.est_s!==undefined) bits.push("≈ "+dur(r.est_s));
      if(bits.length) txt+=" · "+bits.join(" · ");
      cell.textContent=txt;
    });
  }
  if(!run){
    bar.style.width="0%";
    info.className="dim";
    info.textContent="Aucun test lancé pour l'instant — coche des tests puis « Lancer les tests »."
      +(T.server_note?(" ⚠️ "+T.server_note):"");
    box.innerHTML='<span class="dim">—</span>';
    return;
  }
  const p=run.progress||{done:0,total:0};
  bar.style.width=(p.total?(100*p.done/p.total):0)+"%";
  info.className="";
  info.innerHTML='<div class="row"><span class="dim">Dossier :</span> <span class="path">'+esc(run.run_dir||"?")+'</span></div>'+
    '<div class="row"><span class="dim">Modèle :</span> <code>'+esc(run.model_slug||"identification…")+'</code>'+
    ' <span class="dim">· ctx</span> '+esc(run.n_ctx==null?"?":run.n_ctx)+
    ' <span class="dim">· étiquette</span> '+esc(run.label||"?")+
    ' <span class="dim">· début</span> '+esc(run.started_at||"?")+
    ' <span class="dim">· fin</span> '+esc(run.finished_at||"—")+
    ' <span class="dim">· <a href="'+esc(run.run_url||"#")+'" target="_blank">parcourir le dossier</a></span></div>'+
    '<div class="row"><span class="dim">Avancement :</span> '+p.done+" / "+p.total+
    ' <span class="dim">· déjà écoulé</span> '+dur(p.real_done_s)+
    ' <span class="dim">· reste estimé</span> '+dur(p.est_left_s)+
    (run.vram_note?' <span class="dim">· VRAM au lancement</span> '+esc(run.vram_note):"")+'</div>'+
    (T.server_note?('<div class="row" style="color:var(--warn)">⚠️ '+esc(T.server_note)+'</div>'):'');
  const rows=(run.results||[]).map(r=>'<tr><td>'+esc(r.title)+'</td>'+
    '<td class="st '+(r.status||"")+'">'+esc(TLABEL[r.status]||r.status||"")+'</td>'+
    '<td>'+esc(dur(r.est_s))+'</td><td>'+num(r.duree_s,1)+'</td>'+
    '<td>'+esc(r.summary||r.erreur||"—")+
    (r.vram_peak?'<br><span class="dim">VRAM pic '+esc(r.vram_peak)+'</span>':'')+'</td>'+
    '<td>'+(r.log&&run.run_url?'<a href="'+esc(run.run_url+r.log)+'" target="_blank">log</a>':"—")+'</td></tr>').join("");
  box.innerHTML='<table><thead><tr><th>Test</th><th>Statut</th><th>Estimé</th><th>Réel (s)</th><th>Chiffres</th><th>Brut</th></tr></thead><tbody>'+rows+'</tbody></table>';
}

function renderTestPast(s){
  const list=(s.tests||{}).benches||[];
  document.getElementById("t-pastn").textContent=list.length;
  const box=document.getElementById("t-pastbox");
  if(!list.length){box.innerHTML='<span class="dim">—</span>';return;}
  const rows=list.map(b=>'<tr><td><a href="'+esc(b.url)+'" target="_blank">'+esc(b.run_id)+'</a></td>'+
    '<td>'+esc(b.model_slug||"?")+'</td><td>'+esc(b.n_ctx==null?"?":b.n_ctx)+'</td>'+
    '<td>'+esc(String(b.started_at||"?").slice(0,16).replace("T"," "))+'</td>'+
    '<td>'+esc(b.status||"?")+'</td><td class="dim">'+esc((b.resumes||[]).join(" · ")||"—")+'</td></tr>').join("");
  box.innerHTML='<table><thead><tr><th>Bench</th><th>Modèle</th><th>ctx</th><th>Début</th><th>Statut</th><th>Chiffres</th></tr></thead><tbody>'+rows+'</tbody></table>';
}

function renderTests(s){
  const T=s.tests;
  if(!T) return;
  TCAT.clear();
  (T.catalog||[]).forEach(c=>TCAT.set(c.id,c));
  (T.catalog||[]).forEach(c=>{
    const chk=document.querySelector('#tests input[data-test="'+c.id+'"]');
    if(chk){
      chk.disabled=!c.available;
      if(!c.available&&chk.checked){chk.checked=false;tchecked.delete(c.id);}
    }
    const est=document.getElementById("test-est-"+c.id);
    if(est) est.textContent="≈ "+c.est_label+" ("+c.est_source
      +(c.needs_ctx?" · ctx ≥ "+Math.round(c.needs_ctx/1024)+"k":"")+")";
    const why=document.getElementById("test-why-"+c.id);
    if(why) why.textContent=c.available?"":("  ⛔ "+c.reason);
    const last=document.getElementById("tlast-"+c.id);
    if(last) last.textContent=c.last?("dernier · "+c.last+" ("+String(c.last_at||"").slice(0,10)+")"):"";
  });
  tUpdateEstimate();
  renderTestRun(s);
  renderTestPast(s);
}

function wireTests(){
  const box=document.getElementById("tests");
  box.addEventListener("change",e=>{
    if(e.target&&e.target.type==="checkbox"&&e.target.dataset.test){
      const id=e.target.dataset.test;
      if(e.target.checked) tchecked.add(id); else tchecked.delete(id);
      tUpdateEstimate();
    }
  });
  box.addEventListener("click",e=>{
    const b=e.target.closest("button[data-tact]");
    if(!b) return;
    e.preventDefault(); e.stopPropagation();
    const det=b.closest("details.grp");
    const gid=det?det.dataset.tgroup:null;
    Array.from(TCAT.values()).filter(c=>c.group===gid).forEach(c=>{
      const ck=document.querySelector('#tests input[data-test="'+c.id+'"]');
      if(b.dataset.tact==="all"&&c.available){tchecked.add(c.id);if(ck) ck.checked=true;}
      else {tchecked.delete(c.id);if(ck) ck.checked=false;}
    });
    tUpdateEstimate();
  });
  tUpdateEstimate();
}

async function launchTests(){
  const ids=Array.from(tchecked);
  document.getElementById("t-msg").textContent="";
  if(!ids.length){document.getElementById("t-msg").textContent="Sélection vide.";return;}
  try{
    const r=await fetch("/api/tests/run",{method:"POST",headers:{"content-type":"application/json"},
      body:JSON.stringify({ids:ids})});
    const d=await r.json();
    if(!r.ok||d.error) document.getElementById("t-msg").textContent=d.error||("HTTP "+r.status);
    refresh(true);
  }catch(e){document.getElementById("t-msg").textContent="Erreur réseau : "+e;}
}

async function stopTests(){
  try{
    await fetch("/api/tests/stop",{method:"POST",headers:{"content-type":"application/json"},body:"{}"});
    refresh(true);
  }catch(e){}
}

document.getElementById("t-all").onclick=()=>{
  Array.from(TCAT.values()).forEach(c=>{
    const ck=document.querySelector('#tests input[data-test="'+c.id+'"]');
    if(c.available){tchecked.add(c.id);if(ck) ck.checked=true;}
  });
  tUpdateEstimate();
};
document.getElementById("t-none").onclick=()=>{
  document.querySelectorAll("#tests input[type=checkbox]").forEach(c=>c.checked=false);
  tchecked.clear();
  tUpdateEstimate();
};
document.getElementById("t-launch").onclick=launchTests;
document.getElementById("t-stop").onclick=stopTests;

/* ---------------- suivi en direct (v1.2) ----------------
   Le serveur garde le texte du tir courant ; on ne demande que la suite
   (curseurs en caractères), donc rien n'est renvoyé deux fois. */
const LIVE={key:null,text:"",rtext:"",cursor:0,rcursor:0,lastPreview:0,placeholder:true};
const LIVE_POLL_MS=800;

function livePartialHtml(t){
  const m=t.search(/<!DOCTYPE html|<html[\s>]/i);
  if(m<0) return null;
  let h=t.slice(m);
  if(!/<\/html>\s*$/i.test(h)) h+="\n</body></html>";   // HTML encore incomplet : on ferme
  return h;
}

function liveStats(d){
  const bits=[];
  if(d.id!=null) bits.push("#"+d.id+(d.title?" "+esc(d.title):""));
  if(d.mode) bits.push("mode "+esc(d.mode));
  if(d.elapsed_s!=null) bits.push(num(d.elapsed_s,1)+" s");
  if(d.running&&d.tok_s_estime!=null) bits.push("≈"+num(d.tok_s_estime,1)+" tok/s");
  if(d.tok_s!=null) bits.push(num(d.tok_s,1)+" tok/s");
  if(d.completion_tokens!=null) bits.push(d.completion_tokens+" tok");
  else if(d.chunks) bits.push("≈"+d.chunks+" tok");
  bits.push((d.chars||0)+" car.");
  if(d.max_tokens) bits.push("max "+d.max_tokens);
  if(d.params_label) bits.push(esc(d.params_label));
  if(d.finish_reason) bits.push("fin : "+esc(d.finish_reason));
  if(d.truncated) bits.push("affichage borné (tampon)");
  return bits.join(" · ");
}

function applyLive(d){
  const key=(d.id==null)?"":(d.id+"@"+(d.started_at||""));
  if(key!==LIVE.key){LIVE.key=key;LIVE.text="";LIVE.rtext="";LIVE.cursor=0;LIVE.rcursor=0;LIVE.lastPreview=0;}
  LIVE.text = d.text_reset ? (d.text||"") : (LIVE.text + (d.text||""));
  LIVE.rtext = d.reasoning_reset ? (d.reasoning||"") : (LIVE.rtext + (d.reasoning||""));
  LIVE.cursor = d.cursor||0;
  LIVE.rcursor = d.reasoning_cursor||0;

  const st=document.getElementById("livestate");
  st.className="livebadge"+(d.running?" on":(d.status==="error"?" err":(d.id!=null?" fin":"")));
  st.textContent = d.id==null ? "au repos"
    : (d.status==="waiting" ? "attente du slot libre" : (d.running ? "en cours" : (LABEL[d.status]||d.status)));
  document.getElementById("liveslug").textContent = d.slug ? ("· "+d.slug) : "";
  document.getElementById("livestats").innerHTML = d.id==null ? "—" : liveStats(d);

  const pre=document.getElementById("live");
  const auto=document.getElementById("autoscroll").checked;
  if(d.id==null){
    if(!LIVE.placeholder){pre.textContent="Aucun tir en cours — le texte du LLM s'affichera ici, token par token.";LIVE.placeholder=true;}
  }else{
    if(LIVE.placeholder){pre.textContent="";LIVE.placeholder=false;}
    const near=pre.scrollTop+pre.clientHeight>=pre.scrollHeight-40;
    pre.textContent=LIVE.text;
    if(auto&&near) pre.scrollTop=pre.scrollHeight;
  }

  const rbox=document.getElementById("rbox");
  if(LIVE.rtext){
    rbox.hidden=false;
    document.getElementById("rchars").textContent=(d.reasoning_chars||LIVE.rtext.length);
    const rp=document.getElementById("live-reasoning");
    const rnear=rp.scrollTop+rp.clientHeight>=rp.scrollHeight-40;
    rp.textContent=LIVE.rtext;
    if(auto&&rnear) rp.scrollTop=rp.scrollHeight;
  }else rbox.hidden=true;

  updateLivePreview();
}

function updateLivePreview(){
  const on=document.getElementById("livepreview").checked;
  document.getElementById("livepreviewwrap").hidden=!on;
  if(!on) return;
  const now=Date.now();
  if(now-LIVE.lastPreview<1500) return;       // un srcdoc = un rechargement complet
  const h=livePartialHtml(LIVE.text);
  if(!h) return;
  LIVE.lastPreview=now;
  document.getElementById("liveframe").srcdoc=h;
}

async function pollLive(){
  try{
    const r=await fetch("/api/live?since="+LIVE.cursor+"&rsince="+LIVE.rcursor,{cache:"no-store"});
    applyLive(await r.json());
  }catch(e){}
}

async function launch(){
  const ids=Array.from(checked).sort((a,b)=>a-b);
  document.getElementById("msg").textContent="";
  if(!ids.length){document.getElementById("msg").textContent="Sélection vide.";return;}
  const st=readSettings();
  saveSettings(st);
  try{
    const r=await fetch("/api/run",{method:"POST",headers:{"content-type":"application/json"},
      body:JSON.stringify({ids:ids,temperature:st.temperature,max_tokens:st.max_tokens,
                           enable_thinking:st.enable_thinking})});
    const d=await r.json();
    if(!r.ok||d.error) document.getElementById("msg").textContent=d.error||("HTTP "+r.status);
    refresh(true);
  }catch(e){document.getElementById("msg").textContent="Erreur réseau : "+e;}
}

async function stopRun(){
  try{ await fetch("/api/stop",{method:"POST",headers:{"content-type":"application/json"},body:"{}"}); refresh(true); }catch(e){}
}

document.getElementById("all").onclick=()=>{PROMPTS.forEach(p=>checked.add(p.id));
  document.querySelectorAll("#prompts input[type=checkbox]").forEach(c=>c.checked=true);updateCount();};
document.getElementById("none").onclick=()=>{checked.clear();
  document.querySelectorAll("#prompts input[type=checkbox]").forEach(c=>c.checked=false);updateCount();};
document.getElementById("launch").onclick=launch;
document.getElementById("stop").onclick=stopRun;
document.getElementById("retest").onclick=()=>refresh(true);

// --- réglages du run
["set-temp","set-mt","set-think"].forEach(id=>{
  document.getElementById(id).oninput=()=>saveSettings(readSettings());
  document.getElementById(id).onchange=()=>saveSettings(readSettings());
});
document.getElementById("set-reset").onclick=()=>{
  document.getElementById("set-temp").value=DEFAULT_SETTINGS.temperature;
  document.getElementById("set-mt").value=DEFAULT_SETTINGS.max_tokens;
  document.getElementById("set-think").checked=!!DEFAULT_SETTINGS.enable_thinking;
  saveSettings(readSettings());refresh(true);
};

// --- comparaison
document.getElementById("cmp-load").onclick=loadCompare;
document.getElementById("cmp-a").onchange=loadCompare;
document.getElementById("cmp-b").onchange=loadCompare;
document.getElementById("cmp-both").onchange=()=>renderCompare();
document.getElementById("cmp-swap").onclick=()=>{
  const a=document.getElementById("cmp-a"), b=document.getElementById("cmp-b");
  const t=a.value; a.value=b.value; b.value=t; loadCompare();
};
document.getElementById("cmp-cur").onclick=()=>{
  const r=lastState&&lastState.run;
  const summary=document.getElementById("cmp-summary");
  if(!r||!r.run_id){summary.textContent="Aucun run courant à mettre en B.";return;}
  const b=document.getElementById("cmp-b");
  const has=Array.prototype.some.call(b.options,o=>o.value===r.run_id);
  if(!has){summary.textContent="Le run courant n'est pas (encore) dans la liste — réessaie dans une seconde.";return;}
  b.value=r.run_id;
  loadCompare();
};

// --- suivi en direct
document.getElementById("livepreview").onchange=()=>{LIVE.lastPreview=0;updateLivePreview();};
document.getElementById("livebig").onclick=()=>{
  const p=document.getElementById("live");
  p.classList.toggle("big");
  const big=p.classList.contains("big");
  document.getElementById("livebig").textContent=big?"réduire":"agrandir";
  if(big) p.scrollTop=p.scrollHeight;
};
document.getElementById("livecopy").onclick=async()=>{
  const btn=document.getElementById("livecopy");
  const done=(txt)=>{btn.textContent=txt;setTimeout(()=>{btn.textContent="copier le texte";},1500);};
  try{ await navigator.clipboard.writeText(LIVE.text); done("copié ✓"); }
  catch(e){
    // http:// non sécurisé : l'API clipboard est indisponible → repli classique
    const ta=document.createElement("textarea");
    ta.value=LIVE.text; ta.style.position="fixed"; ta.style.opacity="0";
    document.body.appendChild(ta); ta.select();
    let ok=false;
    try{ ok=document.execCommand("copy"); }catch(e2){ ok=false; }
    ta.remove();
    done(ok?"copié ✓":"copie impossible");
  }
};

wireList();
wireTests();
loadSettings();
refresh(true);
fetchRuns();
pollLog();
pollLive();
setInterval(()=>refresh(false),1000);
setInterval(pollLog,1000);
setInterval(pollLive,LIVE_POLL_MS);
setInterval(fetchRuns,15000);       // la liste des runs pour comparer bouge rarement
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# Serveur HTTP
# --------------------------------------------------------------------------

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "prompt-duel/" + APP_VERSION
    protocol_version = "HTTP/1.1"

    # -- plomberie ---------------------------------------------------------
    def log_message(self, fmt, *args):  # silence : notre propre journal suffit
        pass

    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8", head=False, extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if not head and body:
            self.wfile.write(body)

    def _json(self, code, obj, head=False):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", head)

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self):
        self._route("GET")

    def do_HEAD(self):
        self._route("HEAD")

    def do_POST(self):
        self._route("POST")

    # -- routage -----------------------------------------------------------
    def _route(self, method):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        head = (method == "HEAD")
        try:
            if path in ("/", "/index.html") and method in ("GET", "HEAD"):
                catalog = {
                    "version": STATE.prompts_version,
                    "groups": STATE.groups,
                    "prompts": STATE.prompts,
                }
                page = (INDEX_HTML
                        .replace("__GROUPS_HTML__", render_prompts_html())
                        .replace("__TESTS_HTML__", render_tests_html())
                        # "</" échappé : un prompt contient littéralement </script> (import map
                        # three.js) et fermerait la balise <script> avant l'heure. "<\/" est
                        # équivalent à "/" une fois le JSON analysé, donc aucune perte.
                        .replace("__CATALOG_JSON__",
                                 json.dumps(catalog, ensure_ascii=False).replace("</", "<\\/"))
                        .replace("__APP_VERSION__", APP_VERSION)
                        .replace("__APP_TITLE__", html.escape(APP_TITLE)))
                return self._send(200, page, "text/html; charset=utf-8", head)

            if path in ("/benchmarks", "/benchmarks/") and method in ("GET", "HEAD"):
                # index unifie des tests LLM (duels, vitesse, batteries, HumanEval, contexte).
                # etat de visibilite + requete transmis : les liens « afficher les
                # resultats masques » fonctionnent donc aussi SANS JavaScript.
                return self._send(200, render_benchmarks_page(visibility_state(), query),
                                  "text/html; charset=utf-8", head)

            if path in ("/graph", "/graph/") and method in ("GET", "HEAD"):
                # page du graphique (v1.6) — rendue par page_graph.py (lot parallèle).
                if page_graph is None:
                    return self._send(
                        503,
                        "Page /graph indisponible : le module page_graph.py n'est pas "
                        "présent sur cette instance (il est livré séparément).\n",
                        "text/plain; charset=utf-8", head)
                index = load_bench_index()
                state = visibility_state()
                try:
                    page = page_graph.render_graph_page(index, state, query)
                except Exception as exc:
                    LOG.add(f"ERREUR rendu /graph : {exc!r}")
                    return self._send(500, f"Erreur de rendu de la page /graph : {exc}\n",
                                      "text/plain; charset=utf-8", head)
                return self._send(200, page, "text/html; charset=utf-8", head)

            if path == "/api/graph" and method in ("GET", "HEAD"):
                # données du graphique : index unifié -> entrées normalisées ->
                # visibilité (masqués/supprimés retirés) -> une série par métrique.
                index = load_bench_index()
                entries = index.get("entries") if isinstance(index, dict) else []
                entries = entries if isinstance(entries, list) else []
                normalized = metrics.normalize_entries(entries)
                state = visibility_state()
                visible = visibility.apply(state, normalized)
                charts = []
                # `?partiels=1` : tracer aussi les resultats partiels (duel qui a perdu
                # des prompts). Par defaut ils restent dans l'index mais hors courbes.
                inclure_partiels = _query_flag(self.path, "partiels")
                for kind, metric_key in GRAPH_METRICS:
                    data = metrics.series_for_chart(visible, metric_key=metric_key,
                                                    kind=kind,
                                                    include_partial=inclure_partiels)
                    data["kind"] = kind
                    charts.append(data)
                by_kind = {}
                for entry in visible:
                    k = entry.get("kind") or "?"
                    by_kind[k] = by_kind.get(k, 0) + 1
                hidden_ids = sorted(set((state.get("hidden") or [])
                                        + (state.get("deleted") or [])))
                payload = {
                    "charts": charts,
                    "hidden": hidden_ids,
                    "counts": {
                        "entries": len(entries),
                        "visible": len(visible),
                        "hidden": len(hidden_ids),
                        "by_kind": by_kind,
                        "index": (index.get("counts") or {}) if isinstance(index, dict) else {},
                    },
                    "source": (index.get("_source") if isinstance(index, dict) else None),
                }
                return self._json(200, payload, head)

            if path == "/api/state" and method in ("GET", "HEAD"):
                force = str((query.get("refresh") or ["0"])[0]).lower() in ("1", "true", "yes")
                # l'UI envoie les réglages qu'elle a sous les yeux : les badges
                # « déjà fait » correspondent alors vraiment à ce qui serait relancé
                wanted = {
                    "temperature": (query.get("t") or [None])[0],
                    "max_tokens": (query.get("mt") or [None])[0],
                    "enable_thinking": (query.get("think") or [None])[0],
                }
                return self._json(200, build_state(force_probe=force, params=wanted), head)

            if path == "/api/runs" and method in ("GET", "HEAD"):
                # liste complète des runs pour les sélecteurs de comparaison
                try:
                    limit = int((query.get("limit") or [str(RUNS_COMPARE_LIMIT)])[0])
                except ValueError:
                    limit = RUNS_COMPARE_LIMIT
                limit = max(1, min(limit, 2000))
                return self._json(200, {"limit": limit, "runs": list_past_runs_cached(limit)}, head)

            if path == "/api/compare" and method in ("GET", "HEAD"):
                run_a = (query.get("a") or [""])[0]
                run_b = (query.get("b") or [""])[0]
                payload, err = compare_payload(run_a, run_b)
                if err:
                    return self._json(404, err, head)
                return self._json(200, payload, head)

            if path in ("/compare", "/compare.html") and method in ("GET", "HEAD"):
                return self._send(200, render_compare_page(query), "text/html; charset=utf-8", head)

            if path == "/api/log" and method in ("GET", "HEAD"):
                try:
                    since = int((query.get("since") or ["0"])[0])
                except ValueError:
                    since = 0
                lines, nxt = LOG.since(since)
                return self._json(200, {"since": since, "next": nxt, "lines": lines}, head)

            if path == "/api/live" and method in ("GET", "HEAD"):
                # suivi en direct : le client envoie ses curseurs (en caractères) et
                # ne reçoit que la suite. since=0 → tout le tampon courant.
                def _cursor(name):
                    try:
                        return max(0, int((query.get(name) or ["0"])[0]))
                    except ValueError:
                        return 0
                return self._json(200, LIVE.snapshot(_cursor("since"), _cursor("rsince")), head)

            if path == "/api/run" and method == "POST":
                body = self._read_body()
                ids = body.get("ids") or []
                if not isinstance(ids, list):
                    return self._json(400, {"ok": False, "error": "ids doit être une liste"})
                try:
                    ids = [int(i) for i in ids]
                except Exception:
                    return self._json(400, {"ok": False, "error": "ids doit contenir des entiers"})
                if not ids:
                    return self._json(400, {"ok": False, "error": "sélection vide"})
                ids = sorted(set(ids))
                with STATE.lock:
                    known_ids = {p["id"] for p in STATE.prompts}
                    running = any_worker_running()
                unknown = [i for i in ids if i not in known_ids]
                if unknown:
                    return self._json(400, {"ok": False,
                                            "error": f"ids inconnus : {unknown} "
                                                     f"(ids valides : 1→{len(known_ids)})"})
                if running:
                    return self._json(409, {"ok": False, "error": "un run est déjà en cours"})
                probe = probe_llm(force=True)
                if probe["status"] == "off":
                    return self._json(503, {
                        "ok": False,
                        "error": f"LLM éteint ou injoignable sur {LLM_BASE} — "
                                 "démarre ton serveur llama.cpp puis réessaie "
                                 f"({probe.get('error') or 'injoignable'})",
                    })
                declared = body.get("settings") if isinstance(body.get("settings"), dict) else None
                ok, msg = start_run(ids, params=normalize_params(body),
                                    declared_settings=declared)
                return self._json(200 if ok else 409, {"ok": ok, "error": None if ok else msg,
                                                       "message": msg,
                                                       "params": normalize_params(body)})

            if path == "/api/tests/run" and method == "POST":
                body = self._read_body()
                ids = body.get("ids") or []
                if not isinstance(ids, list) or not ids:
                    return self._json(400, {"ok": False, "error": "sélection vide"})
                ids = [str(i) for i in ids]
                unknown = [i for i in ids if i not in bench_tests.TESTS_BY_ID]
                if unknown:
                    return self._json(400, {"ok": False,
                                            "error": f"tests inconnus : {unknown}"})
                if any_worker_running():
                    return self._json(409, {"ok": False,
                                            "error": "un traitement est déjà en cours "
                                                     "(duel ou tests)"})
                probe = probe_llm(force=True)
                if probe["status"] == "off":
                    return self._json(503, {
                        "ok": False,
                        "error": f"LLM éteint ou injoignable sur {LLM_BASE} — les tests de "
                                 f"vitesse et de contexte ont besoin du serveur "
                                 f"({probe.get('error') or 'injoignable'})",
                    })
                # disponibilité : le contexte réellement servi doit couvrir le test
                refused = []
                for tid in ids:
                    ok, why = bench_availability(bench_tests.TESTS_BY_ID[tid], probe)
                    if not ok:
                        refused.append(f"{tid} : {why}")
                if refused:
                    return self._json(409, {"ok": False, "error": "test(s) indisponible(s) — "
                                                               + " | ".join(refused)})
                ok, msg = start_bench(ids, declared_settings=body.get("settings")
                                      if isinstance(body.get("settings"), dict) else None)
                return self._json(200 if ok else 409, {"ok": ok, "message": msg,
                                                       "error": None if ok else msg})

            if path == "/api/visibility" and method == "POST":
                # masquer / réafficher / supprimer / restaurer des résultats (v1.6)
                body = self._read_body()
                ok, payload = apply_visibility_action(body.get("action"), body.get("ids"))
                return self._json(200 if ok else 400, payload)

            if path == "/api/tests/stop" and method == "POST":
                with STATE.lock:
                    if STATE.bench_worker is None or not STATE.bench_worker.is_alive():
                        return self._json(409, {"ok": False, "error": "aucun test en cours"})
                    STATE.bench_stop = True
                LOG.add("stop demandé — le test en cours est interrompu, le suivant ne part pas")
                return self._json(200, {"ok": True, "message": "arrêt demandé"})

            if path == "/api/benches" and method in ("GET", "HEAD"):
                try:
                    limit = int((query.get("limit") or ["20"])[0])
                except ValueError:
                    limit = 20
                return self._json(200, {"benches": list_past_benches(max(1, min(limit, 200)))},
                                  head)

            if path == "/api/stop" and method == "POST":
                with STATE.lock:
                    if STATE.worker is None or not STATE.worker.is_alive():
                        return self._json(409, {"ok": False, "error": "aucun run en cours"})
                    STATE.stop_requested = True
                LOG.add("stop demandé — le prompt en cours va au bout, le suivant ne sera pas lancé")
                return self._json(200, {"ok": True, "message": "arrêt demandé"})

            if path.startswith("/benches/"):
                return self._serve_static(path, "/benches/", BENCH_DIR, head)

            if path.startswith("/runs/"):
                return self._serve_runs(path, head)

            return self._json(404, {"error": f"route inconnue : {path}"})
        except Exception as exc:
            LOG.add(f"ERREUR serveur {method} {path} : {exc!r}")
            try:
                self._json(500, {"error": f"{exc.__class__.__name__}: {exc}"})
            except Exception:
                pass

    # -- fichiers de run / de bench ----------------------------------------
    def _serve_runs(self, path, head):
        return self._serve_static(path, "/runs/", RUNS_DIR, head)

    def _serve_static(self, path, prefix, root_dir, head):
        rel = path[len(prefix):]
        target = (root_dir / rel)
        try:
            resolved = target.resolve()
            root = root_dir.resolve()
            if root != resolved and root not in resolved.parents:
                return self._send(403, "403 interdit\n")
        except Exception:
            return self._send(404, "404 introuvable\n")

        if not resolved.exists():
            return self._send(404, "404 introuvable\n")

        if resolved.is_dir():
            entries = sorted(resolved.iterdir(), key=lambda p: (p.is_file(), p.name))
            links = []
            if resolved != root_dir.resolve():
                links.append('<li><a href="../">../</a></li>')
            for e in entries:
                if e.name.startswith(".tmp"):
                    continue
                name = e.name + ("/" if e.is_dir() else "")
                links.append(f'<li><a href="{html.escape(name)}">{html.escape(name)}</a></li>')
            page = ("<!DOCTYPE html><html lang=\"fr\"><head><meta charset=\"utf-8\">"
                    f"<title>{html.escape(rel)}</title></head><body style=\"background:#12141a;color:#e6e8ee;"
                    "font-family:system-ui,sans-serif;padding:16px\">"
                    f"<h2>{html.escape(str(resolved))}</h2><ul>" + "".join(links) + "</ul></body></html>")
            return self._send(200, page, "text/html; charset=utf-8", head)

        try:
            body = resolved.read_bytes()
        except Exception as exc:
            return self._send(500, f"500 {exc}\n")
        ctype = CONTENT_TYPES.get(resolved.suffix.lower(), "application/octet-stream")
        return self._send(200, body, ctype, head)


def main():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    STATE.prompts, STATE.groups, STATE.prompts_version = load_catalog()
    LOG.add(f"prompt-duel v{APP_VERSION} (python {sys.version.split()[0]})")
    n_v1 = sum(1 for p in STATE.prompts if p["group"] == "canvas-2d")
    extra = f" · {len(STATE.prompts) - n_v1} nouveaux" if n_v1 else ""
    LOG.add(f"{len(STATE.prompts)} prompts chargés · {len(STATE.groups)} groupes{extra}")
    for g in STATE.groups:
        LOG.add(f"  groupe {g['id']} — {g['title']} : {g['count']} prompt(s)")
    LOG.add(f"format prompts.json : v{STATE.prompts_version} · max_tokens par défaut : {MAX_TOKENS}")
    LOG.add(f"réglages par défaut : {params_label(DEFAULT_PARAMS)} "
            "(modifiables dans l'UI, pour chaque run)")
    LOG.add(f"endpoint LLM : {LLM_BASE}")
    LOG.add(f"benches : {BENCH_DIR} — {len(bench_tests.TESTS)} tests cochables "
            f"({len(bench_tests.GROUPS)} familles : "
            f"{', '.join(g['id'] for g in bench_tests.GROUPS)})")
    if DRY_RUN:
        LOG.add("MODE LLM_DRY_RUN=1 — aucun appel réseau vers llama.cpp")
    LOG.add(f"runs : {RUNS_DIR}")
    probe = probe_llm(force=True)
    if probe["status"] == "off":
        LOG.add(f"LLM éteint / injoignable ({probe.get('error')}) — l'app reste utilisable")
    else:
        LOG.add(f"LLM {probe['status']} — modèle {probe['model_slug']} · ctx {probe['n_ctx']}")
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as exc:
        # pas de traceback pour une erreur d'exploitation courante : message utile
        if exc.errno == errno.EADDRINUSE:
            print()
            print(f"  ERREUR : le port {PORT} est déjà utilisé — une autre instance de prompt-duel "
                  f"tourne déjà (ou un autre service écoute sur ce port).")
            print()
            print("  Donc l'app est probablement DÉJÀ accessible, inutile de la relancer :")
            print(f"    → ouvrir http://127.0.0.1:{PORT}   (ou http://<ip-de-la-machine>:{PORT})")
            print("    → état du service :  systemctl --user status prompt-duel")
            print()
            print("  Trois solutions :")
            print(f"    1. garder le service et juste ouvrir l'UI (rien à faire)")
            print(f"    2. lancer à la main :  systemctl --user stop prompt-duel  puis  python3 app.py")
            print(f"    3. utiliser un autre port :  PORT={PORT + 1} python3 app.py")
            print()
            sys.exit(1)
        print(f"ERREUR : impossible d'écouter sur {HOST}:{PORT} — {exc}")
        sys.exit(1)
    server.daemon_threads = True
    LOG.add(f"UI : http://{HOST}:{PORT}  (depuis une autre machine : http://<ip-de-cette-machine>:{PORT})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.add("arrêt demandé (Ctrl-C)")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
