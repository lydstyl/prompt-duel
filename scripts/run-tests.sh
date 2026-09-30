#!/usr/bin/env bash
# scripts/run-tests.sh — harnais de tests de prompt-duel.
#
# Usage :
#   bash scripts/run-tests.sh            # tout (unitaires puis e2e)
#   bash scripts/run-tests.sh unit       # unitaires uniquement (python3 -m unittest)
#   bash scripts/run-tests.sh e2e        # end-to-end uniquement (Playwright)
#   bash scripts/run-tests.sh all        # equivaut a l'appel sans argument
#
# E2E : l'app est lancee REELLEMENT en LLM_DRY_RUN=1 (aucun appel reseau vers
# llama.cpp), sur un port de test (E2E_PORT, defaut 8799), avec RUNS_DIR/BENCH_DIR
# pointes sur un dossier TEMPORAIRE alimente par e2e/fixtures/. Jamais de sleep
# aveugle : on poll la racine tant que l'app ne repond pas (timeout ~20 s). Le
# process est tue quoi qu'il arrive (trap EXIT).
#
# Resolution de Playwright : on prefere e2e/node_modules, puis le paquet global
# (npm root -g), puis le cache npx, puis NODE_PATH. On ne depend donc pas d'un
# `npm install` reussi (le reseau npm peut etre indisponible).
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
E2E_DIR="$ROOT_DIR/e2e"
E2E_PORT="${E2E_PORT:-8799}"
PW_DIR=""

log() { printf '%s\n' "$*"; }
hr()  { printf '%s\n' "============================================================"; }

usage() {
  log "usage : bash scripts/run-tests.sh [unit|e2e|all]"
  log "  (sans argument = all ; E2E_PORT surcharge le port de test, defaut $E2E_PORT)"
}

# --------------------------------------------------------------------------
# 1) Unitaires — python3 -m unittest discover -s tests
# --------------------------------------------------------------------------
run_unit() {
  hr; log "UNITAIRES — python3 -m unittest discover -s tests"; hr
  local tests_dir="$ROOT_DIR/tests"
  if [ ! -d "$tests_dir" ]; then
    log "aucun dossier tests/ — etape ignoree (OK)"
    return 0
  fi
  local files=()
  while IFS= read -r -d '' f; do files+=("$f"); done \
    < <(find "$tests_dir" -maxdepth 3 -name 'test*.py' -print0 2>/dev/null)
  if [ "${#files[@]}" -eq 0 ]; then
    log "aucun test unitaire pour l'instant (tests/ vide) — etape ignoree (OK)"
    return 0
  fi
  log "${#files[@]} fichier(s) de test trouve(s) dans tests/"
  ( cd "$ROOT_DIR" && python3 -m unittest discover -s tests )
}

# --------------------------------------------------------------------------
# Helpers e2e
# --------------------------------------------------------------------------
port_busy() {
  python3 - "$1" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
try:
    rc = s.connect_ex(("127.0.0.1", port))
finally:
    s.close()
sys.exit(0 if rc == 0 else 1)   # 0 = quelqu'un ecoute deja
PY
}

# Poll sur '/', timeout 20 s, pas de sleep aveugle (0,2 s entre deux essais).
wait_ready() {
  python3 - "$1" <<'PY'
import sys, time, urllib.request
port = int(sys.argv[1])
url = f"http://127.0.0.1:{port}/"
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

# Le paquet playwright attend une REVISION de navigateur precise. On prefere un
# paquet dont le navigateur est deja present dans le cache, pour ne jamais
# dependre d'un telechargement (reseau souvent indisponible).
pw_browser_ok() {
  node - "$1" <<'JS'
const fs = require('fs'), path = require('path');
const dir = process.argv[2];
try {
  const b = JSON.parse(fs.readFileSync(path.join(dir, 'playwright-core', 'browsers.json'), 'utf8'));
  const root = process.env.PLAYWRIGHT_BROWSERS_PATH
    || path.join(process.env.HOME || '', '.cache', 'ms-playwright');
  const chromium = (b.browsers || [])
    .filter((x) => x.name.startsWith('chromium') && !x.name.includes('tip-of-tree'));
  process.exit(chromium.some((x) => fs.existsSync(path.join(root, x.name + '-' + x.revision))) ? 0 : 1);
} catch (e) {
  process.exit(1);
}
JS
}

# Cherche un dossier node_modules contenant le paquet playwright.
find_playwright() {
  PW_DIR=""
  local candidates=() cand

  [ -d "$E2E_DIR/node_modules/playwright" ] && candidates+=("$E2E_DIR/node_modules")

  cand="$(npm root -g 2>/dev/null || true)"
  [ -n "$cand" ] && [ -d "$cand/playwright" ] && candidates+=("$cand")

  if [ -d "$HOME/.npm/_npx" ]; then
    for cand in "$HOME"/.npm/_npx/*/node_modules; do
      [ -d "$cand/playwright" ] && candidates+=("$cand")
    done
  fi

  if [ -n "${NODE_PATH:-}" ]; then
    local IFS_OLD="$IFS"; IFS=':'
    for cand in $NODE_PATH; do
      [ -d "$cand/playwright" ] && candidates+=("$cand")
    done
    IFS="$IFS_OLD"
  fi

  if [ "${#candidates[@]}" -gt 0 ]; then
    # 1) un paquet dont les navigateurs sont deja installes
    for cand in "${candidates[@]}"; do
      if pw_browser_ok "$cand"; then PW_DIR="$cand"; return 0; fi
    done
    # 2) sinon le premier candidat, en dernier recours
    PW_DIR="${candidates[0]}"
    return 0
  fi
  return 1
}

# --------------------------------------------------------------------------
# 2) End-to-end — demarre l'app en dry run puis lance Playwright
# --------------------------------------------------------------------------
run_e2e() {
  hr; log "E2E — app reelle en LLM_DRY_RUN=1, port de test $E2E_PORT"; hr

  if port_busy "$E2E_PORT"; then
    log "ERREUR : le port $E2E_PORT est deja utilise (un autre service ou une instance ecoute deja)."
    log "        choisis un autre port :  E2E_PORT=8801 bash scripts/run-tests.sh e2e"
    return 2
  fi

  if ! find_playwright; then
    log "paquet 'playwright' introuvable — tentative d'installation locale dans e2e/…"
    if command -v timeout >/dev/null 2>&1; then
      ( cd "$E2E_DIR" && timeout 120 npm install --no-audit --no-fund --prefer-offline --loglevel=error )
    else
      ( cd "$E2E_DIR" && npm install --no-audit --no-fund --prefer-offline --loglevel=error )
    fi
    if ! find_playwright; then
      log "ERREUR : 'playwright' reste introuvable. Installe-le :  cd e2e && npm install"
      return 3
    fi
  fi
  log "playwright : $PW_DIR/playwright"
  export NODE_PATH="$PW_DIR${NODE_PATH:+:$NODE_PATH}"

  local tmp app_pid=""
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/prompt-duel-e2e.XXXXXX")"

  cleanup() {
    if [ -n "$app_pid" ] && kill -0 "$app_pid" 2>/dev/null; then
      kill "$app_pid" 2>/dev/null
      wait "$app_pid" 2>/dev/null
    fi
  }
  trap cleanup EXIT

  # donnees de test : copie des fixtures dans le dossier temporaire
  cp -R "$E2E_DIR/fixtures/runs" "$tmp/runs"
  cp -R "$E2E_DIR/fixtures/benches" "$tmp/benches"

  PORT="$E2E_PORT" HOST=127.0.0.1 RUNS_DIR="$tmp/runs" BENCH_DIR="$tmp/benches" \
  LLM_DRY_RUN=1 LLM_BASE="http://127.0.0.1:9" APP_TITLE="Prompt Duel (e2e)" \
  BENCH_INDEX="$E2E_DIR/fixtures/bench_index.json" \
  BENCH_HE_SSH="e2e@127.0.0.1" BENCH_LLM_HOME="$tmp" \
  python3 "$ROOT_DIR/app.py" >"$tmp/app.log" 2>&1 &
  app_pid=$!
  log "app lancee (pid $app_pid) — donnees : $tmp"

  if ! wait_ready "$E2E_PORT"; then
    log "ERREUR : l'app n'a pas repondu — journal :"
    tail -n 30 "$tmp/app.log" | while IFS= read -r line; do log "  | $line"; done
    cleanup; trap - EXIT
    return 4
  fi
  log "app prete sur http://127.0.0.1:$E2E_PORT/ — lancement de Playwright"
  log "(durée attendue : ~15-25 s)"

  export E2E_PORT
  export E2E_BASE_URL="http://127.0.0.1:$E2E_PORT"

  local rc=0
  ( cd "$E2E_DIR" && node "$PW_DIR/playwright/cli.js" test --reporter=list ) || rc=$?

  cleanup
  trap - EXIT

  if [ "$rc" -eq 0 ]; then
    log "Playwright : OK (code de sortie 0)"
    rm -rf "$tmp"
  else
    log "Playwright : ECHEC (code de sortie $rc)"
    log "artefacts conserves : $tmp"
    log "dernieres lignes du journal de l'app :"
    tail -n 40 "$tmp/app.log" | while IFS= read -r line; do log "  | $line"; done
  fi
  return "$rc"
}

# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------
cmd="${1:-all}"
case "$cmd" in
  unit)
    run_unit; exit $?
    ;;
  e2e)
    run_e2e; exit $?
    ;;
  all)
    run_unit; u=$?
    run_e2e;  e=$?
    hr
    u_txt="OK"; [ "$u" -ne 0 ] && u_txt="ECHEC (code $u)"
    e_txt="OK"; [ "$e" -ne 0 ] && e_txt="ECHEC (code $e)"
    log "RESULTAT — unitaires : $u_txt   |   e2e : $e_txt"
    hr
    if [ "$u" -eq 0 ] && [ "$e" -eq 0 ]; then exit 0; else exit 1; fi
    ;;
  -h|--help|help)
    usage; exit 0
    ;;
  *)
    log "sous-commande inconnue : $cmd"
    usage; exit 2
    ;;
esac
