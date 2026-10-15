#!/usr/bin/env bash
# scripts/check-app-locale.sh — test HTTP REEL de l'app lancee en local.
#
# Pourquoi ce script : la suite unitaire rend les pages EN MEMOIRE, et la suite e2e
# Playwright ne couvre qu'une partie des routes. Un cablage manquant (route /llms
# absente du routeur, module non importe, variable d'environnement non lue) ne se
# voit nulle part ailleurs. Ici : l'app est REELLEMENT lancee sur un port de test,
# avec un index et des donnees TEMPORAIRES (jamais le vault, jamais visibility.json
# du depot), et chaque route est appelee en HTTP comme le ferait un navigateur.
#
# Usage :  bash scripts/check-app-locale.sh            # port 8788
#          PORT_TEST=8790 bash scripts/check-app-locale.sh
#
# Codes de sortie : 0 = toutes les verifications passent ; 1 = echec (le detail est
# imprime) ; 2 = port deja utilise ; 3 = l'app n'a pas repondu (journal imprime).
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT_TEST="${PORT_TEST:-8788}"

log() { printf '%s\n' "$*"; }
hr()  { printf '%s\n' "============================================================"; }

port_busy() {
  python3 - "$1" <<'PY'
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
try:
    rc = s.connect_ex(("127.0.0.1", int(sys.argv[1])))
finally:
    s.close()
sys.exit(0 if rc == 0 else 1)
PY
}

wait_ready() {
  python3 - "$1" <<'PY'
import sys, time, urllib.request
url = f"http://127.0.0.1:{int(sys.argv[1])}/"
deadline = time.time() + 20.0
last = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            if r.status == 200:
                sys.exit(0)
    except Exception as exc:
        last = exc
    time.sleep(0.2)
print(f"timeout : l'app n'a pas repondu sur {url} (derniere erreur : {last!r})", file=sys.stderr)
sys.exit(1)
PY
}

hr; log "TEST HTTP REEL — app lancee sur 127.0.0.1:$PORT_TEST (donnees temporaires)"; hr

if port_busy "$PORT_TEST"; then
  log "ERREUR : le port $PORT_TEST est deja utilise (autre instance ?)."
  log "        choisis un autre port :  PORT_TEST=8790 bash scripts/check-app-locale.sh"
  exit 2
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/prompt-duel-check.XXXXXX")"
APP_PID=""
cleanup() {
  if [ -n "$APP_PID" ] && kill -0 "$APP_PID" 2>/dev/null; then
    kill "$APP_PID" 2>/dev/null
    wait "$APP_PID" 2>/dev/null
  fi
}
trap cleanup EXIT

# Donnees de test : fixtures e2e si presentes, sinon l'index local du depot.
if [ -d "$ROOT_DIR/e2e/fixtures" ]; then
  cp -R "$ROOT_DIR/e2e/fixtures/runs" "$TMP/runs" 2>/dev/null
  cp -R "$ROOT_DIR/e2e/fixtures/benches" "$TMP/benches" 2>/dev/null
  INDEX="$ROOT_DIR/e2e/fixtures/bench_index.json"
else
  mkdir -p "$TMP/runs" "$TMP/benches"
  INDEX="$ROOT_DIR/bench_index.json"
fi
[ -f "$INDEX" ] || { log "ERREUR : index de test introuvable ($INDEX)"; exit 1; }
mkdir -p "$TMP/runs" "$TMP/benches"

PORT="$PORT_TEST" HOST=127.0.0.1 \
RUNS_DIR="$TMP/runs" BENCH_DIR="$TMP/benches" \
BENCH_INDEX="$INDEX" VISIBILITY_PATH="$TMP/visibility.json" \
LLM_DRY_RUN=1 LLM_BASE="http://127.0.0.1:9" APP_TITLE="Prompt Duel (check local)" \
python3 "$ROOT_DIR/app.py" >"$TMP/app.log" 2>&1 &
APP_PID=$!
log "app lancee (pid $APP_PID) — donnees : $TMP"

if ! wait_ready "$PORT_TEST"; then
  log "ERREUR : l'app n'a pas repondu — journal :"
  tail -n 30 "$TMP/app.log" | while IFS= read -r line; do log "  | $line"; done
  exit 3
fi

python3 - "$PORT_TEST" "$INDEX" "$ROOT_DIR" <<'PY'
"""Verifications HTTP reelles, cles LUES dans l'index (aucune donnee codee en dur)."""
import json
import sys
import urllib.error
import urllib.request

port, index_path, root = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, root)
import llms        # noqa: E402
import metrics     # noqa: E402

base = f"http://127.0.0.1:{port}"
with open(index_path, encoding="utf-8") as fh:
    index = json.load(fh)
norm = metrics.normalize_entries(index.get("entries") or [])
familles = [f for f in llms.catalogue_llm(norm) if f["identifie"]]
variantes = [item["key"] for item in metrics.series_catalogue(norm)]
if len(familles) < 2:
    print("ECHEC : moins de 2 LLM identifiables dans l'index de test")
    sys.exit(1)
if len(variantes) < 2:
    print("ECHEC : moins de 2 reglages tracables dans l'index de test")
    sys.exit(1)
k1, k2 = familles[0]["key"], familles[1]["key"]
l1, l2 = familles[0]["label"], familles[1]["label"]
v1, v2 = variantes[0], variantes[1]
print(f"index de test : {len(norm)} entree(s) · LLM {k1} / {k2} · reglages {v1[:24]}…")

erreurs = []


def appel(chemin, attendu=200, methode="GET"):
    url = base + chemin
    requete = urllib.request.Request(url, method=methode)
    try:
        with urllib.request.urlopen(requete, timeout=20) as r:
            corps = r.read().decode("utf-8", "replace")
            return r.status, corps
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:
        return 0, f"<exception {exc!r}>"


def verifie(nom, condition, detail=""):
    etat = "OK " if condition else "ECHEC"
    print(f"  [{etat}] {nom}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        erreurs.append(f"{nom}{(' — ' + detail) if detail else ''}")


# --- pages, en HTTP reel ---------------------------------------------------
statut, page = appel("/")
verifie("GET / : 200", statut == 200, f"HTTP {statut}")
verifie("GET / : liens /llms et /graph dans l'en-tete",
        'href="/llms"' in page and 'href="/graph"' in page)

statut, page = appel("/llms")
verifie("GET /llms : 200", statut == 200, f"HTTP {statut}")
verifie("GET /llms : catalogue avec selection multiple",
        'name="llm"' in page and "multiple" in page)
verifie("GET /llms : les LLM de l'index sont proposes", f'value="{k1}"' in page)

statut, page = appel(f"/llms?llm={k1}&llm={k2}")
verifie("GET /llms?llm=&llm= : 200", statut == 200, f"HTTP {statut}")
verifie("GET /llms?llm=&llm= : tableau comparatif", page.count("<table") >= 3,
        f"{page.count('<table')} tableau(x)")
verifie("GET /llms?llm=&llm= : les deux LLM sont nommes",
        l1 in page and l2 in page)

statut, page = appel(f"/llms?llm={k1}&llm=cle-qui-nexiste-pas")
verifie("GET /llms : cle inconnue signalee (page quand meme rendue)",
        statut == 200 and "cle-qui-nexiste-pas" in page, f"HTTP {statut}")

statut, page = appel("/graph")
verifie("GET /graph (etape 1) : 200", statut == 200, f"HTTP {statut}")
verifie("GET /graph : catalogue des LLM present", 'id="selection-llm"' in page)
verifie("GET /graph (etape 1) : aucun graphique", "<svg" not in page)

statut, page = appel(f"/graph?llm={k1}&llm={k2}")
verifie("GET /graph?llm=&llm= : 200", statut == 200, f"HTTP {statut}")
verifie("GET /graph?llm=&llm= : une courbe par LLM (SVG inline)", page.count("<svg") >= 1,
        f"{page.count('<svg')} svg")
verifie("GET /graph?llm=&llm= : aucune cle inconnue", "inconnu" not in page)
verifie("GET /graph?llm=&llm= : les deux LLM en legende",
        l1 in page and l2 in page)

statut, page = appel(f"/graph?sel={v1}&sel={v2}")
verifie("GET /graph?sel=&sel= (non-regression) : 200", statut == 200, f"HTTP {statut}")
verifie("GET /graph?sel=&sel= : graphiques rendus", page.count("<svg") >= 1,
        f"{page.count('<svg')} svg")

statut, page = appel(f"/graph?llm={k1}")
verifie("GET /graph avec 1 seul LLM : rien trace, raison dite",
        statut == 200 and "<svg" not in page and "au moins 2" in page, f"HTTP {statut}")

statut, page = appel(f"/graph?llm={v1}&llm={v2}")
verifie("GET /graph?llm=<cle de variante> (ancien lien) : fonctionne",
        statut == 200 and page.count("<svg") >= 1, f"HTTP {statut}")

statut, page = appel("/benchmarks")
verifie("GET /benchmarks : 200 + lien /llms", statut == 200 and 'href="/llms"' in page,
        f"HTTP {statut}")

statut, page = appel("/compare")
verifie("GET /compare : 200", statut == 200, f"HTTP {statut}")

statut, page = appel("/llms", methode="HEAD")
verifie("HEAD /llms : 200", statut == 200, f"HTTP {statut}")

# --- routes JSON -----------------------------------------------------------
for chemin in ("/api/state", "/api/graph", "/api/runs", "/api/benches", "/api/log"):
    statut, corps = appel(chemin)
    ok_json = False
    if statut == 200:
        try:
            json.loads(corps)
            ok_json = True
        except ValueError:
            ok_json = False
    verifie(f"GET {chemin} : 200 + JSON", statut == 200 and ok_json, f"HTTP {statut}")

# --- l'index de test n'a pas bouge ----------------------------------------
import os  # noqa: E402
avant = os.stat(index_path).st_mtime
statut, _ = appel("/")
verifie("l'index de test n'est jamais modifie", os.stat(index_path).st_mtime == avant)

print()
if erreurs:
    print(f"ECHEC — {len(erreurs)} verification(s) en echec :")
    for erreur in erreurs:
        print(f"  · {erreur}")
    sys.exit(1)
print("OK — toutes les verifications HTTP passent")
PY
rc=$?

hr
if [ "$rc" -eq 0 ]; then
  log "TEST HTTP REEL : OK"
  rm -rf "$TMP"
else
  log "TEST HTTP REEL : ECHEC (code $rc) — journal de l'app :"
  tail -n 30 "$TMP/app.log" | while IFS= read -r line; do log "  | $line"; done
  log "donnees conservees : $TMP"
fi
hr
exit "$rc"
