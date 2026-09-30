#!/usr/bin/env python3
"""visibility.py — etat persiste des resultats masques / supprimes et filtres.

Un seul fichier JSON, par defaut a cote de ce module (`visibility.json`) :

    {"hidden": ["kind:source", ...],     # masques : retires des vues, recuperables
     "deleted": ["kind:source", ...],    # tombstones : le resultat d'origine n'est
                                         # PAS supprime par ce module, il est
                                         # seulement exclu et garde sa trace
     "filters": {"kinds": [], "models": [], "variants": [], "series": []}}

Contraintes :
  * l'app est multi-threads (ThreadingHTTPServer) -> l'ecriture est ATOMIQUE
    (fichier temporaire + `os.replace`) et serialisee par un verrou : un lecteur
    voit toujours l'ancien fichier complet ou le nouveau, jamais un demi-fichier ;
  * fichier absent, tronque ou illisible -> etat VIDE, jamais d'exception ;
  * `apply()` ne modifie JAMAIS les entrees qu'on lui passe (copies).

CLI de debug :
    python3 visibility.py --show            # etat courant
    python3 visibility.py --hide <id> ...   # masque des identifiants
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.environ.get("VISIBILITY_PATH") or os.path.join(APP_DIR, "visibility.json")

# Cles de filtres acceptees (pluriel = forme canonique, singulier tolere).
FILTER_ALIASES = {
    "kind": "kinds", "kinds": "kinds",
    "model": "models", "models": "models",
    "variant": "variants", "variants": "variants",
    "serie": "series", "series": "series",
}
FILTER_KEYS = ("kinds", "models", "variants", "series")

# `metrics.py` (meme dossier) fournit les identifiants et les variantes ; son
# absence ne doit pas empecher l'app de tourner (repli local ci-dessous).
try:
    import metrics
except ImportError:  # pragma: no cover - depend de l'environnement
    metrics = None

_LOCK = threading.RLock()


# --------------------------------------------------------------- etat
def empty_state():
    """Etat vide : jamais None, toujours copiable et serialisable."""
    return {"hidden": [], "deleted": [], "filters": {k: [] for k in FILTER_KEYS}}


def normalize_state(state):
    """Etat propre : listes de chaines sans doublon, filtres connus uniquement."""
    clean = empty_state()
    if not isinstance(state, dict):
        return clean
    for key in ("hidden", "deleted"):
        values = state.get(key) or []
        if isinstance(values, (str, bytes)):
            values = [values]
        seen = []
        for value in values:
            text = str(value)
            if text and text not in seen:
                seen.append(text)
        clean[key] = seen
    filters = state.get("filters") or {}
    if isinstance(filters, dict):
        for raw_key, values in filters.items():
            key = FILTER_ALIASES.get(str(raw_key).strip().lower())
            if not key:
                continue
            if isinstance(values, (str, bytes)):
                values = [values]
            elif not isinstance(values, (list, tuple, set)):
                continue
            for value in values:
                text = str(value)
                if text and text not in clean["filters"][key]:
                    clean["filters"][key].append(text)
    return clean


def load(path=None):
    """Etat persiste (fichier absent / vide / illisible -> etat vide)."""
    target = path or DEFAULT_PATH
    with _LOCK:
        try:
            with open(target, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError, UnicodeDecodeError):
            return empty_state()
    return normalize_state(data)


def save(state, path=None):
    """Ecriture ATOMIQUE de l'etat (tmp + os.replace). Ne leve jamais d'OSError."""
    target = path or DEFAULT_PATH
    payload = json.dumps(normalize_state(state), ensure_ascii=False, indent=1) + "\n"
    directory = os.path.dirname(os.path.abspath(target)) or "."
    tmp = None
    with _LOCK:
        try:
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".visibility-", suffix=".tmp", dir=directory)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
            tmp = None
        except OSError as exc:
            print(f"! visibility.save({target}) impossible : {exc}", file=sys.stderr)
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass


# --------------------------------------------------------------- identifiants
def _as_ids(ids):
    if ids is None:
        return []
    if isinstance(ids, (str, bytes)):
        ids = [ids]
    out = []
    for value in ids:
        text = str(value)
        if text and text not in out:
            out.append(text)
    return out


def entry_id(entry):
    """Identifiant d'une entree, avec `metrics.entry_id` en priorite."""
    if not isinstance(entry, dict):
        return ""
    if metrics is not None:
        return metrics.entry_id(entry)
    if entry.get("entry_id"):
        return str(entry["entry_id"])
    source = entry.get("source") or entry.get("run_id") or ""
    return f"{entry.get('kind') or '?'}:{source}"


def variant_key(entry):
    """Cle de variante d'une entree (via `metrics`), ou None."""
    if not isinstance(entry, dict):
        return None
    if entry.get("variant") and isinstance(entry["variant"], dict):
        return entry["variant"].get("key")
    if metrics is not None:
        try:
            return metrics.entry_variant(entry).get("key")
        except (TypeError, ValueError):
            return None
    return None


# --------------------------------------------------------------- consultation
def is_hidden(state, entry_id):
    """Masque OU supprime : dans les deux cas, l'entree ne doit pas s'afficher."""
    if not isinstance(state, dict):
        return False
    return (str(entry_id) in (state.get("hidden") or [])
            or str(entry_id) in (state.get("deleted") or []))


def is_deleted(state, entry_id):
    """Supprime (tombstone) : exclu definitivement, la source reste intacte."""
    return isinstance(state, dict) and str(entry_id) in (state.get("deleted") or [])


# --------------------------------------------------------------- mutations
def _mutate(state, ids, key):
    state = state if isinstance(state, dict) else {}
    state.setdefault("hidden", [])
    state.setdefault("deleted", [])
    state.setdefault("filters", {k: [] for k in FILTER_KEYS})
    for value in _as_ids(ids):
        if value not in state[key]:
            state[key].append(value)
    return state


def _unmutate(state, ids, key):
    state = state if isinstance(state, dict) else {}
    values = _as_ids(ids)
    state[key] = [v for v in (state.get(key) or []) if v not in values]
    return state


def hide(state, ids):
    """Masque des entrees (retirees des vues, recuperables). Retourne l'etat."""
    with _LOCK:
        return _mutate(state, ids, "hidden")


def unhide(state, ids):
    """Affiche de nouveau des entrees masquees."""
    with _LOCK:
        return _unmutate(state, ids, "hidden")


def delete(state, ids):
    """Tombstone : l'entree est exclue (la source n'est pas supprimee ici)."""
    with _LOCK:
        state = _mutate(state, ids, "deleted")
        _unmutate(state, ids, "hidden")   # supprime = plus la peine de le masquer
        return state


def restore(state, ids):
    """Retire les tombstones : l'entree reapparait."""
    with _LOCK:
        return _unmutate(state, ids, "deleted")


def set_filter(state, name, values):
    """Remplace un filtre de serie ('kind', 'model', 'variant', 'series')."""
    key = FILTER_ALIASES.get(str(name or "").strip().lower())
    if not key:
        return state
    state = normalize_state(state) if not isinstance(state, dict) else state
    state.setdefault("filters", {k: [] for k in FILTER_KEYS})
    state["filters"][key] = _as_ids(values)
    return state


# --------------------------------------------------------------- application
def _passes_filters(entry, filters):
    """Filtres = listes blanches : vide = pas de filtre. Toutes les listes
    renseignees doivent matcher (ET logique)."""
    def _match(values, *candidates):
        wanted = [str(v).lower() for v in values]
        for candidate in candidates:
            if candidate is None:
                continue
            if str(candidate).lower() in wanted:
                return True
        return False

    if filters["kinds"] and not _match(filters["kinds"], entry.get("kind")):
        return False
    if filters["models"] and not _match(filters["models"], entry.get("model"),
                                        entry.get("model_slug")):
        return False
    if filters["variants"] or filters["series"]:
        key = variant_key(entry)
        if filters["variants"] and not _match(filters["variants"], key):
            return False
        if filters["series"] and not _match(filters["series"], key):
            return False
    return True


def apply(state, entries):
    """Entrees visibles : masquees et supprimees retirees, filtres appliques.

    Retourne des COPIES superficielles : les entrees recues ne sont jamais
    modifiees, et l'ordre d'origine est conserve.
    """
    state = state if isinstance(state, dict) else {}
    hidden = set(state.get("hidden") or [])
    deleted = set(state.get("deleted") or [])
    raw_filters = state.get("filters") or {}
    filters = {}
    for key in FILTER_KEYS:
        values = raw_filters.get(key) or []
        if isinstance(values, (str, bytes)):
            values = [values]
        filters[key] = [str(v) for v in values]

    out = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        eid = entry.get("entry_id") or entry_id(entry)
        if eid in hidden or eid in deleted:
            continue
        if any(filters.values()) and not _passes_filters(entry, filters):
            continue
        out.append(dict(entry))
    return out


# --------------------------------------------------------------- CLI de debug
def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="Resultats masques / supprimes (prompt-duel)")
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--show", action="store_true", help="affiche l'etat")
    ap.add_argument("--hide", nargs="*", default=None, metavar="ID")
    ap.add_argument("--unhide", nargs="*", default=None, metavar="ID")
    ap.add_argument("--delete", nargs="*", default=None, metavar="ID")
    ap.add_argument("--restore", nargs="*", default=None, metavar="ID")
    args = ap.parse_args(argv)

    state = load(args.path)
    changed = False
    for ids, fn in ((args.hide, hide), (args.unhide, unhide),
                    (args.delete, delete), (args.restore, restore)):
        if ids:
            fn(state, ids)
            changed = True
    if changed:
        save(state, args.path)
        print(f"-> ecrit {args.path}")
    print(json.dumps(normalize_state(state), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
