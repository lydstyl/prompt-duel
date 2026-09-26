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
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

APP_VERSION = "1.2.0"

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

REQUEST_TIMEOUT_S = 1800         # un tir peut être très long
SLOT_POLL_S = 5                  # poll de /slots en attente de slot libre
HEALTH_TIMEOUT_S = 3
PROBE_CACHE_S = 3.0              # l'UI polle toutes les 1 s : on lisse les sondes
LOG_MAX = 2000
RUNS_LIST_LIMIT = 30
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
        self._t0 = None
        self.updated_at = None

    # -- écriture (worker) -------------------------------------------------
    def begin(self, run_id, pid, title, slug, mode="stream", max_tokens=None):
        with self._lock:
            self._clear_locked()
            self.run_id = run_id
            self.pid, self.title, self.slug = pid, title, slug
            self.mode, self.max_tokens = mode, max_tokens
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


def _chat_payload(prompt, max_tokens, stream=False):
    payload = {
        "model": "local",  # ignoré par llama.cpp (serveur mono-modèle)
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": int(max_tokens or MAX_TOKENS),
        "cache_prompt": CACHE_PROMPT,
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
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


def chat_completion_stream(prompt, max_tokens=None, on_delta=None):
    """POST /v1/chat/completions en streaming SSE (llama.cpp).

    Renvoie (payload, duree_s) avec EXACTEMENT la même forme que
    chat_completion() : le reste du worker ne voit pas la différence.
    on_delta(kind, morceau) est appelé au fil de l'eau ("content"|"reasoning").
    """
    payload = _chat_payload(prompt, max_tokens, stream=True)
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


def chat_completion(prompt, max_tokens=None, on_delta=None):
    """POST /v1/chat/completions. Renvoie (payload, duree_s).

    Si `on_delta` est fourni et que le streaming est actif (LLM_STREAM=1, défaut),
    le tir se fait en SSE et le texte arrive au fur et à mesure. Si le streaming
    échoue AVANT le moindre morceau, on retombe proprement sur le tir en un bloc.
    """
    if on_delta is not None and STREAM:
        try:
            data, elapsed = chat_completion_stream(prompt, max_tokens=max_tokens, on_delta=on_delta)
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

    payload = _chat_payload(prompt, max_tokens, stream=False)
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


def list_already_done(model_slug=None, n_ctx=None, max_age=5.0):
    """{ "<id>": "<run_id>" } des prompts déjà générés avec le MÊME modèle + ctx + température.

    Lecture seule, tolérante à l'absence/illisibilité des fichiers, jamais bloquante.
    Résultat mémorisé quelques secondes (l'UI polle /api/state toutes les secondes).
    """
    key = (model_slug, n_ctx)
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
        if (data.get("params") or {}).get("temperature") != TEMPERATURE:
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
# Gestion des runs sur disque
# --------------------------------------------------------------------------

def run_suffix(model_slug, n_ctx):
    ctx = n_ctx if n_ctx else 0
    think = "think" if ENABLE_THINKING else "nothink"
    return f"__{model_slug}__ctx{ctx}__t{TEMPERATURE:g}__{think}"


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
                counts = {}
                for res in data.get("results") or []:
                    st = res.get("status") or "?"
                    counts[st] = counts.get(st, 0) + 1
                info["counts"] = counts
            except Exception:
                pass
        out.append(info)
    return out


# --------------------------------------------------------------------------
# Worker d'exécution (un seul à la fois — 1 slot llama.cpp)
# --------------------------------------------------------------------------

def wait_for_free_slot():
    """Boucle jusqu'à ce que le slot soit libre. False si stop demandé."""
    while True:
        with STATE.lock:
            if STATE.stop_requested:
                return False
        try:
            busy = fetch_slot_busy()
        except Exception as exc:
            raise RuntimeError(f"lecture de /slots impossible : {exc}")
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


def make_result(prompt):
    return {
        "id": prompt["id"],
        "title": prompt["title"],
        "group": prompt["group"],
        "category": prompt["category"],
        "slug": prompt["slug"],
        "max_tokens": prompt.get("max_tokens") or MAX_TOKENS,
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


def run_worker(ids):
    """Exécute les prompts un par un, dans l'ordre croissant des ids."""
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

        suffix = run_suffix(model_slug, n_ctx)
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
            eff_max_tokens = prompt.get("max_tokens") or MAX_TOKENS

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
                       max_tokens=eff_max_tokens)

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
                                                    on_delta=LIVE.append)

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
                    "finish_reason": finish_reason,
                    "prompt": prompt["prompt"],
                    "system_prompt": SYSTEM_PROMPT,
                    "params": {
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "max_tokens": eff_max_tokens,
                        "enable_thinking": ENABLE_THINKING,
                        "cache_prompt": CACHE_PROMPT,
                    },
                    "model_path": model_path,
                    "model_slug": model_slug,
                    "n_ctx": n_ctx,
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
                        "params": {"temperature": TEMPERATURE, "top_p": TOP_P,
                                   "max_tokens": eff_max_tokens,
                                   "enable_thinking": ENABLE_THINKING,
                                   "cache_prompt": CACHE_PROMPT},
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


def start_run(ids):
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
            "params": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
                "enable_thinking": ENABLE_THINKING,
                "cache_prompt": CACHE_PROMPT,
            },
            "python_version": sys.version.split()[0],
            "app_version": APP_VERSION,
            "prompts_version": STATE.prompts_version,
            "groups": [{"id": g["id"], "title": g["title"], "count": g.get("count", 0)}
                       for g in STATE.groups],
            "dry_run": DRY_RUN,
            "selected_ids": ids,
            "status": "running",
            "results": [make_result(prompts[i]) for i in ids],
        }
        worker = threading.Thread(target=run_worker, args=(ids,), name="prompt-duel-worker", daemon=True)
        STATE.worker = worker
        worker.start()
    LOG.add(f"run lancé — {len(ids)} prompt(s) : {ids}")
    return True, "run lancé"


def build_state(force_probe=False):
    probe = probe_llm(force=force_probe)
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
        return {
            "app_version": APP_VERSION,
            "prompts_version": STATE.prompts_version,
            "dry_run": DRY_RUN,
            "stream": STREAM,
            "live_poll_ms": LIVE_POLL_MS,
            "llm_base": LLM_BASE,
            "runs_dir": str(RUNS_DIR),
            "llm": dict(probe),
            "groups": json.loads(json.dumps(STATE.groups)),
            "prompts": [dict(p) for p in STATE.prompts],
            "already_done": list_already_done(model_slug=probe.get("model_slug"),
                                              n_ctx=probe.get("n_ctx")),
            "run": run,
            "is_running": is_running,
            "stop_requested": stop_requested,
            "runs": list_past_runs(),
        }


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
</style>
</head>
<body>
<header>
  <h1>__APP_TITLE__ <span class="dim">· lanceur de prompts vers un LLM local</span></h1>
  <div id="llm" class="badge off">état LLM…</div>
  <button id="retest">Re-tester</button>
  <div class="spacer"></div>
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
    <div class="bar"><div id="barfill"></div></div>
    <div id="msg"></div>
    <div id="runinfo" class="dim">Aucun run pour l'instant.</div>
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
  document.getElementById("launch").disabled = running || checked.size===0 || !llmUsable(s);
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
  renderLLM(s);
  renderStatuses(s);
  renderDone(s);
  renderRun(s);
  renderPast(s);
  updateCount();
}

async function refresh(force){
  try{
    const r=await fetch("/api/state"+(force?"?refresh=1":""),{cache:"no-store"});
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
  try{
    const r=await fetch("/api/run",{method:"POST",headers:{"content-type":"application/json"},
      body:JSON.stringify({ids:ids})});
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
refresh(true);
pollLog();
pollLive();
setInterval(()=>refresh(false),1000);
setInterval(pollLog,1000);
setInterval(pollLive,LIVE_POLL_MS);
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
                        # "</" échappé : un prompt contient littéralement </script> (import map
                        # three.js) et fermerait la balise <script> avant l'heure. "<\/" est
                        # équivalent à "/" une fois le JSON analysé, donc aucune perte.
                        .replace("__CATALOG_JSON__",
                                 json.dumps(catalog, ensure_ascii=False).replace("</", "<\\/"))
                        .replace("__APP_VERSION__", APP_VERSION)
                        .replace("__APP_TITLE__", html.escape(APP_TITLE)))
                return self._send(200, page, "text/html; charset=utf-8", head)

            if path == "/api/state" and method in ("GET", "HEAD"):
                force = str((query.get("refresh") or ["0"])[0]).lower() in ("1", "true", "yes")
                return self._json(200, build_state(force_probe=force), head)

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
                    running = STATE.worker is not None and STATE.worker.is_alive()
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
                ok, msg = start_run(ids)
                return self._json(200 if ok else 409, {"ok": ok, "error": None if ok else msg, "message": msg})

            if path == "/api/stop" and method == "POST":
                with STATE.lock:
                    if STATE.worker is None or not STATE.worker.is_alive():
                        return self._json(409, {"ok": False, "error": "aucun run en cours"})
                    STATE.stop_requested = True
                LOG.add("stop demandé — le prompt en cours va au bout, le suivant ne sera pas lancé")
                return self._json(200, {"ok": True, "message": "arrêt demandé"})

            if path.startswith("/runs/"):
                return self._serve_runs(path, head)

            return self._json(404, {"error": f"route inconnue : {path}"})
        except Exception as exc:
            LOG.add(f"ERREUR serveur {method} {path} : {exc!r}")
            try:
                self._json(500, {"error": f"{exc.__class__.__name__}: {exc}"})
            except Exception:
                pass

    # -- fichiers de run ---------------------------------------------------
    def _serve_runs(self, path, head):
        rel = path[len("/runs/"):]
        target = (RUNS_DIR / rel)
        try:
            resolved = target.resolve()
            root = RUNS_DIR.resolve()
            if root != resolved and root not in resolved.parents:
                return self._send(403, "403 interdit\n")
        except Exception:
            return self._send(404, "404 introuvable\n")

        if not resolved.exists():
            return self._send(404, "404 introuvable\n")

        if resolved.is_dir():
            entries = sorted(resolved.iterdir(), key=lambda p: (p.is_file(), p.name))
            links = []
            if resolved != RUNS_DIR.resolve():
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
    STATE.prompts, STATE.groups, STATE.prompts_version = load_catalog()
    LOG.add(f"prompt-duel v{APP_VERSION} (python {sys.version.split()[0]})")
    n_v1 = sum(1 for p in STATE.prompts if p["group"] == "canvas-2d")
    extra = f" · {len(STATE.prompts) - n_v1} nouveaux" if n_v1 else ""
    LOG.add(f"{len(STATE.prompts)} prompts chargés · {len(STATE.groups)} groupes{extra}")
    for g in STATE.groups:
        LOG.add(f"  groupe {g['id']} — {g['title']} : {g['count']} prompt(s)")
    LOG.add(f"format prompts.json : v{STATE.prompts_version} · max_tokens par défaut : {MAX_TOKENS}")
    LOG.add(f"endpoint LLM : {LLM_BASE}")
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
