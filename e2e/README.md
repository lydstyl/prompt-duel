# Tests de prompt-duel

Deux niveaux, un seul point d'entrée : `scripts/run-tests.sh`.

```bash
bash scripts/run-tests.sh          # unitaires + end-to-end (défaut)
bash scripts/run-tests.sh unit     # unitaires seulement
bash scripts/run-tests.sh e2e      # end-to-end seulement
```

Le script se lance **depuis la racine du dépôt** et ne dépend d'aucune variable
d'environnement préalable. `E2E_PORT` (défaut `8799`) surcharge le port de test.

## Unitaires

`run-tests.sh unit` exécute simplement :

```
python3 -m unittest discover -s tests
```

Stdlib uniquement, pas de pytest (contrainte du projet). Si `tests/` est absent
ou n'a aucun fichier `test*.py`, l'étape est ignorée sans erreur — le harnais
reste vert tant que les tests unitaires n'existent pas encore.

## End-to-end (Playwright)

`run-tests.sh e2e` :

1. refuse de démarrer si le port de test est déjà occupé (message clair,
   suggestion `E2E_PORT=8801 …`) ;
2. lance **l'app réelle** en `LLM_DRY_RUN=1` (aucun appel réseau vers llama.cpp),
   `RUNS_DIR` / `BENCH_DIR` pointés sur un **dossier temporaire** alimenté par
   `e2e/fixtures/` — les dossiers `runs/` et `benches/` du dépôt ne sont jamais
   touchés ;
3. attend que `GET /` réponde (poll toutes les 0,2 s, timeout ~20 s, **pas de
   `sleep` aveugle**), en échouant vite si le process meurt (journal de l'app) ;
4. lance Playwright sur `e2e/*.spec.js` ;
5. **tue l'app quoi qu'il arrive** (`trap EXIT`) et affiche le code de sortie.
   Les artefacts (journal de l'app, dossier temporaire) sont conservés en cas
   d'échec, supprimés en cas de succès.

### Couverture

| spec | vérifie |
|---|---|
| `home.spec.js` | page d'accueil : formulaire chargé, badge LLM « DRY RUN », catalogue complet (autant de cases que de prompts, groupés en `<details>`), section tests rendue ; zéro erreur console |
| `benchmarks.spec.js` | `/benchmarks` : titre, compte « N entrees » conforme à la fixture, tableaux de l'index affichés ; zéro erreur console |
| `compare.spec.js` | `/compare?a=…&b=…` accessible et rend deux runs réels côte à côte ; zéro erreur console |
| `duel-run.spec.js` | duel réel en dry run : coche → lance → le run apparaît dans l'UI et dans `/api/runs` → statut terminal (`finished`) ; zéro erreur console |

Les specs utilisent des sélecteurs sémantiques (texte, `role`, `#id`, `data-*`)
et `expect(...).toBeVisible()` / `expect.poll` — jamais de `waitForTimeout`.
Une seule écriture disque possible (le duel) et elle vise le dossier temporaire.

Les erreurs console sont collectées via `page.on('console')` et
`page.on('pageerror')` (`helpers.js`) ; le bruit `favicon.ico` de Chromium est
filtré, c'est le seul faux positif écarté.

## Playwright : résolution du paquet

Le script ne dépend **pas** d'un `npm install` réussi (le réseau npm peut être
indisponible). Il cherche un paquet `playwright` dans cet ordre :

1. `e2e/node_modules/playwright` (installation locale) ;
2. `$(npm root -g)/playwright` (paquet global) ;
3. `~/.npm/_npx/*/node_modules/playwright` (cache npx) ;
4. `NODE_PATH`.

Parmi les candidats, il **préfère celui dont les navigateurs sont déjà
installés** dans `~/.cache/ms-playwright` (ou `PLAYWRIGHT_BROWSERS_PATH`) :
Playwright exige une révision de navigateur précise par version. Si aucun paquet
n'est trouvé, il tente `npm install` dans `e2e/` avant d'échouer.

Version pinée dans `package.json` : **`playwright@1.62.0`**, qui correspond aux
navigateurs présents sur cette machine (`chromium_headless_shell-1234`). Sur une
autre machine, soit lancer `npx playwright install` pour la version épinglée,
soit ajuster la version dans `e2e/package.json`.

## Ajouter une spec

Créer `e2e/<nom>.spec.js` (le runner les découvre automatiquement), utiliser
`baseURL` (l'app est déjà lancée) et `attachConsoleGuard(page)` de `helpers.js`
pour la vérification « zéro erreur console ».
