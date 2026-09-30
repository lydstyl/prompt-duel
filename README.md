# Prompt Duel 224

Web app locale qui envoie des prompts à un LLM local (llama.cpp) et génère
**un fichier HTML autonome par prompt**. Les sorties sont rangées dans
`runs/<horodatage>__<slug du modèle>__ctx<contexte>__t<température>__nothink/`,
pour qu'on sache toujours **quel modèle et quels réglages** ont produit quel fichier.

Python 3 **stdlib uniquement** — aucune dépendance, aucun `pip install`, aucun `npm install`.

**v1.1.0** : catalogue porté à **32 prompts répartis en 4 groupes** (canvas 2D facile, canvas 2D
avancé, 3D/WebGL, logique & jeu), affichés par sections dans l'UI, et `max_tokens` porté à
**16384** — les prompts 3D et jeu produisent 8k→15k tokens et étaient tronqués à 8192.
**+2 prompts importés** (groupe `importes`, ids 33→34) : voir « Prompts importés » ci-dessous.

**v1.2.0** : **suivi en direct pendant un tir** — les tirs passent en **streaming SSE**
(`stream: true` vers llama.cpp), chaque morceau est poussé dans un tampon mémoire lu par
l'UI via `/api/live` : on voit le texte arriver token par token, la vitesse, le
raisonnement, et on peut afficher un **aperçu HTML live** de la page en cours d'écriture.
Voir « Suivi en direct » ci-dessous.

**v1.3.0** : **réglages par run** (température, `max_tokens`, raisonnement choisis dans
l'UI) et **comparaison de deux runs** — un tableau prompt par prompt avec durée / tok/s /
tokens des deux côtés, écarts colorés, et une page `/compare` qui affiche les **deux rendus
HTML côte à côte** dans l'app. Voir « Comparer deux runs » ci-dessous.

**v1.4.0** : **index unifié des tests** — `bench_index.py` agrège les duels,
les mesures de vitesse, les batteries de petites tâches, HumanEval et les
remplissages de contexte dans un `index.json` rangé dans le **vault Obsidian**
(source de vérité), et la page **`/benchmarks`** les affiche dans l'app.
Voir « Index des tests et page /benchmarks » ci-dessous.

**v1.4.1** : **`reasoning_effort` au premier niveau** du body, en plus de
`chat_template_kwargs.enable_thinking` (valeur `none` quand le run est demandé sans
raisonnement, `medium` avec). Certains modèles — le **Ternary Bonsai 2** de PrismML — ignorent
`enable_thinking` et raisonnent par défaut : sans ce champ, un run « nothink » était en
réalité un run « think » (durée multipliée, comparaison faussée). Les runtimes qui ne
connaissent pas le champ l'ignorent, donc aucun effet de bord sur Qwen3.6/3.8 ni Swift.

---

## Démarrage

```bash
cd ~/apps/prompt-duel
python3 app.py                 # bind 0.0.0.0:8791
```

UI : **http://<ip-de-la-machine>:8791**

Endpoint du LLM : par défaut `http://127.0.0.1:8080` (llama.cpp local).
Pour un serveur sur une autre machine : `LLM_BASE=http://192.168.1.50:8080 python3 app.py`.

Variantes :

```bash
PORT=8899 python3 app.py            # autre port
HOST=127.0.0.1 python3 app.py       # écoute locale seulement
LLM_DRY_RUN=1 python3 app.py        # aucun appel à llama.cpp (tests / démo)
LLM_STREAM=0 python3 app.py         # pas de streaming (tir en un bloc, comme en v1.1)
LIVE_MAX_CHARS=400000 python3 app.py  # taille max du tampon de suivi en direct
RUNS_DIR=/tmp/runs-test python3 app.py  # runs ailleurs (instance de test isolée)
LLM_BASE=http://autre-pc:8080 python3 app.py   # autre endpoint serveur
```

Le LLM lui-même n'est **pas** lancé par cette app : c'est à l'utilisateur de démarrer
son serveur llama.cpp. Si llama.cpp est éteint, l'app le détecte,
affiche une pastille rouge « LLM éteint » et reste utilisable (le journal
explique l'erreur, pas de traceback dans l'UI).

## Arrêt

`Ctrl-C` dans le terminal, ou, si le service systemd est actif :
`systemctl --user stop prompt-duel`.

## Service systemd (optionnel)

Exemple d'unité utilisateur — à adapter, `%h` désigne le home de l'utilisateur :

```bash
systemctl --user status prompt-duel     # doit être « active (running) »
systemctl --user restart prompt-duel    # après une modification de app.py
systemctl --user stop prompt-duel
journalctl --user -u prompt-duel -n 50  # journal systemd
```

Pour le (re)créer à l'identique :

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/prompt-duel.service <<'EOF'
[Unit]
Description=Prompt Duel 224 (LLM local -> HTML)
After=network.target

[Service]
ExecStart=/usr/bin/python3 %h/apps/prompt-duel/app.py
WorkingDirectory=%h/apps/prompt-duel
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now prompt-duel
```

Si `loginctl enable-linger $USER` demandait sudo : ne pas insister, `python3 app.py`
dans un terminal suffit parfaitement.

## Catalogue des prompts (`prompts.json`)

Depuis la v1.1, `prompts.json` est un **objet** (format v2) : les prompts sont regroupés,
et `groups` ordonne l'affichage.

```json
{
  "version": 2,
  "groups": [
    { "id": "canvas-2d",        "title": "Canvas 2D — niveau facile (v1)", "note": "…" },
    { "id": "canvas-2d-avance", "title": "Canvas 2D — niveau avancé",      "note": "…" },
    { "id": "webgl-3d",         "title": "3D / WebGL (three.js r160 CDN)", "note": "…" },
    { "id": "logique-jeu",      "title": "Logique & jeu",                  "note": "…" }
  ],
  "prompts": [
    { "id": 1, "slug": "lorenz-attractor", "title": "…", "group": "canvas-2d",
      "category": "physique", "prompt": "…", "criteria": ["…", "…", "…"] }
  ]
}
```

| Groupe | Prompts | Contenu |
|---|---|---|
| `canvas-2d` | 10 (ids 1→10) | v1 — canvas 2D, JS vanilla, zéro librairie |
| `canvas-2d-avance` | 6 (ids 11→16) | Stable Fluids, Gray-Scott, ray tracer CPU, labyrinthe + A*, ondes 2D, métaballs |
| `webgl-3d` | 8 (ids 17→24) | Mandelbulb, scène 3D, planète procédurale, terrain infini, tesseract, système solaire, trou noir, galaxie GPU |
| `logique-jeu` | 8 (ids 25→32) | Go 9×9, Puissance 4, 2048, démineur, billard, mini-golf, Sokoban, Tetris |
| `importes` | 2 (ids 33→34) | Physique du sable (automate cellulaire), Donjon procédural + brouillard de guerre |

- Chaque prompt porte `group` (id de groupe) **et** `category` (domaine fin, affiché en étiquette).
- **Rétrocompatibilité v1** : un ancien `prompts.json` (tableau racine) reste accepté ; tous ses
  prompts tombent alors dans le groupe de secours `divers`. Un `group` absent ou inconnu →
  `divers`, avec une ligne dans le journal. Jamais de plantage, jamais de prompt perdu.
- ⚠️ Les 8 prompts du groupe `webgl-3d` chargent **three.js r160 depuis un CDN épinglé**
  (`https://cdn.jsdelivr.net/npm/three@0.160.0/…`) : **connexion Internet requise pour afficher
  ces 8 pages**. C'est le seul groupe autorisé à charger une librairie externe.

### Prompts importés (groupe `importes`)

Deux prompts récupérés dans l'historique du dépôt public **`lukesdevlab/youtube`**, d'où ils ont
été **supprimés le 08/09/2026** (commit `a92c9930`, « clear out promots ») :

| id | slug | version récupérée | origine |
|---|---|---|---|
| 33 | `sand-physics` | `a5398076` (12/08/2026) | `prompts/sand-physics.txt` |
| 34 | `dungeon-game` | `3288d191` (03/09/2026) | `prompts/dungeon-game.txt` |

- Textes **verbatim** (aucune reformulation) ; leur `prompt` s'adresse à un agent de code mais
  demande explicitement « a single HTML file » sans librairie externe, donc compatible avec le
  message système de l'app. L'extraction HTML tolère la checklist de vérification qui suit le code.
- Champ `source` conservé dans `prompts.json`, dans `/api/state` et dans la copie
  `prompts.json` de chaque run (traçabilité).

### `max_tokens`

- Défaut **16384**, surchargeable globalement : `MAX_TOKENS=8192 python3 app.py`.
- Surcharge **par prompt** : un champ optionnel `"max_tokens"` dans une entrée de `prompts.json`
  s'applique à ce prompt uniquement (aucun des 34 prompts fournis n'en a besoin — porte de sortie).
- Le `max_tokens` **effectif** apparaît dans `meta.json` (`params.max_tokens`) et dans `run.json`.
  Le nommage des dossiers de run ne change pas (`…__t0.2__nothink`).
- Réponse terminée en `finish_reason: "length"` → le journal le signale
  (`prompt N : réponse tronquée (length) — HTML probablement incomplet`) : c'est un signal de
  qualité, pas un crash, et le fichier est écrit normalement.

## Où sont les runs

```
~/apps/prompt-duel/runs/
├── latest -> 2026-09-25_2310__qwen3.8-27b-ud-q4_k_xl__ctx262144__t0.2__nothink   (symlink)
└── 2026-09-25_2310__qwen3.8-27b-ud-q4_k_xl__ctx262144__t0.2__nothink/
    ├── run.json          # identité du LLM + réglages + résultats
    ├── prompts.json      # copie des prompts DU RUN (format v2, run sélectionné uniquement)
    ├── 01_lorenz-attractor/
    │   ├── index.html    # LE livrable
    │   ├── meta.json     # prompt envoyé, tokens, tok/s, durée, statut, réglages
    │   ├── reasoning.txt # SEULEMENT si le modèle a produit du raisonnement
    │   ├── partial.txt   # SEULEMENT si le tir a été coupé en cours de route
    │   └── raw.txt       # SEULEMENT si l'extraction HTML a échoué
    └── 02_mandelbrot-zoom/…
```

- `runs/latest` pointe sur le dernier run (recréé à chaque run).
- Le dossier de run reprend **le nom du `.gguf` réellement chargé** (`GET /v1/models`)
  et le contexte réel (`GET /props` → `default_generation_settings.n_ctx`).
  Exemple réellement observé le 25/09/2026 : `Qwen3.8-27B-UD-Q2_K_XL.gguf`, ctx 131072
  → `runs/2026-09-25_2315__qwen3.8-27b-ud-q2_k_xl__ctx131072__t0.2__nothink/`.
- Depuis la v1.3, le nom porte aussi **`__mt<n>`** quand le `max_tokens` choisi diffère du
  défaut (`16384`) : `…__t0.5__nothink__mt8192`. C'est ce qui évite qu'un run avec d'autres
  réglages écrase ou réutilise le run par défaut. Le `max_tokens` par défaut n'apparaît pas,
  donc les dossiers produits par les v1.1/v1.2 gardent exactement le même nom.
- Depuis l'UI, les liens « ouvrir » passent par `/runs/<run>/…`.

## API

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/` | UI (français, dark, sans dépendance) |
| GET | `/api/state` | état complet (LLM, `prompts_version`, `groups`, prompts, `already_done`, run en cours, progression, résultats, runs passés, `params`, `default_params`) |
| GET | `/api/state?refresh=1` | idem en forçant la re-sonde du LLM (bouton « Re-tester ») |
| GET | `/api/state?t=…&mt=…&think=…` | idem, mais les badges « déjà fait » sont calculés pour **ces réglages** (c'est ce que l'UI envoie à chaque poll) |
| POST | `/api/run` | body `{"ids":[1,4,7],"temperature":0.2,"max_tokens":16384,"enable_thinking":false}` → run séquentiel (**jusqu'à 34 ids**, aucun plafond) ; les réglages sont optionnels (défauts sinon) ; 400 si sélection vide **ou id inconnu**, 409 si un run tourne déjà, 503 si le LLM est éteint |
| POST | `/api/stop` | arrêt demandé : le prompt en cours va au bout, le suivant n'est pas lancé |
| GET | `/api/log?since=N` | lignes de journal depuis l'index N |
| GET | `/api/live?since=C&rsince=R` | suivi en direct : uniquement la suite du texte (`C`) et du raisonnement (`R`) depuis ces curseurs **en caractères** |
| GET | `/api/runs?limit=N` | liste des runs (défaut 500, max 2000), utilisée par les sélecteurs de comparaison — au-delà des 30 du polling `/api/state` |
| GET | `/api/compare?a=<run>&b=<run>` | comparaison de deux runs : identité + réglages des deux, une ligne par prompt (union) avec `status`, `duree_s`, `tok_s`, tokens, `open_url`, et les écarts `delta` (b − a) ; 404 si un run est introuvable |
| GET | `/compare?a=<run>&b=<run>[&ids=26,29]` | page de comparaison visuelle : les deux rendus HTML en `<iframe>` côte à côte, prompt par prompt |
| GET | `/runs/…` | fichiers générés (+ listing de dossier) |
| GET | `/benchmarks` | index unifié des tests LLM (duels, vitesse/VRAM, batteries, HumanEval, contexte) — lu dans le vault, copie locale en secours |

- `groups` = `[{id, title, note, count, prompt_ids}]`, dans l'ordre du fichier.
- `already_done` = `{"<id>": "<run_id>"}` : prompts déjà générés avec **le même modèle**
  (`model_slug` + `n_ctx` + mêmes réglages). L'UI affiche « déjà fait · \<run_id\> ».

## Réglages du run (v1.3)

Dans la carte de lancement : **température**, **max_tokens** et **raisonnement**
(`chat_template_kwargs.enable_thinking`). Valeurs par défaut : `t0.2`, `mt16384`,
raisonnement **off** — réinitialisables par le bouton « défauts ». Les réglages sont
gardés dans le `localStorage` du navigateur et rappelés au prochain chargement.

- Le résumé à droite des champs montre le **suffixe de dossier** qui sera appliqué
  (`…__t0.5__nothink__mt8192`) : on voit donc *avant* de lancer si le run ira dans un
  nouveau dossier ou rejoindra un dossier existant (idempotence).
- Les badges « déjà fait » sont recalculés pour **les réglages affichés** (l'UI envoie
  `?t=…&mt=…&think=…` à `/api/state`) : changer la température fait donc réapparaître
  les prompts comme à faire.
- ⚠️ Activer le raisonnement sur ce build peut envoyer la réponse dans
  `reasoning_content` : le texte s'affiche alors dans le bloc violet du suivi en direct
  et `content` peut être vide (statut `error`, raisonnement conservé dans
  `reasoning.txt`). Le `meta.json` de chaque prompt garde les réglages utilisés.

## Comparer deux runs (v1.3)

Ouvrir la carte **« Comparer deux runs »** : deux listes déroulantes (A = référence,
B = comparé, la plus récente en tête), un bouton « B = run courant » pour opposer le
dernier run à un autre, et le tableau se charge tout seul avec les deux runs les plus
récents.

Le tableau (alimenté par `/api/compare`) montre, **sur la même ligne**, le même prompt
dans les deux runs :

| colonne | contenu |
|---|---|
| `#` / Titre | id et titre du prompt (groupe en infobulle) |
| A · état | pastille verte `ok`, orange `no_html`, rouge `error`, violet `deja_genere` + lien `↗` vers le fichier |
| A · durée / tok/s / tokens | les chiffres du run A (tokens = `completion_tokens`, survoler donne les tokens du prompt) |
| B · … | idem pour le run B |
| écart (B−A) | Δ durée en secondes et Δ tok/s, **verts quand B est meilleur**, rouges sinon |
| `voir les 2` | ouvre `/compare?a=…&b=…&ids=<id>` sur ce prompt |

- Ligne estompée = prompt présent d'un seul côté (`—` de l'autre côté) ; la case
  « seulement les prompts présents des deux côtés » filtre ces lignes.
- Un prompt absent des deux côtés n'apparaît pas.

### Page `/compare` — les rendus côte à côte

`/compare?a=<runA>&b=<runB>[&ids=26,29]` est une page autonome (même style sombre) :

- en tête, l'identité des deux runs (modèle, ctx, réglages, date, nombre de prompts,
  lien vers le dossier) et les cases de filtrage ;
- pour chaque prompt, les **deux rendus HTML dans deux `<iframe>` côte à côte**, avec
  au-dessus les chiffres de chaque côté, un bouton `↻` pour recharger un aperçu et un
  lien **plein écran ↗** (ouvre le fichier seul dans un onglet) ;
- les aperçus sont en `loading="lazy"` : ils se chargent quand on arrive sur la ligne
  (utile quand on compare 10 prompts canvas/WebGL) ;
- les iframes portent `sandbox="allow-scripts allow-same-origin allow-pointer-lock
  allow-modals allow-downloads allow-popups"` : les pages générées s'exécutent
  normalement (c'est le même rendu qu'en plein écran), mais elles restent dans leur cadre.

## Index des tests et page `/benchmarks` (v1.4.0)

Les tests LLM d'un même parc vivent éparpillés : duels HTML dans `runs/`, mesures de
vitesse dans `~/llm/case-*.log`, batteries de petites tâches dans
`~/llm/smalltasks/results_*.json`, remplissages de contexte dans `~/llm/fill-*.out`.
`bench_index.py` les **regroupe dans un seul index** et **copie les artefacts dans le
vault Obsidian**, qui devient la source de vérité :

```
documents/llm-benchmarks/
  index.json                  <- l'index (généré)
  <campagne>/vitesse/         <- case-*.log (miroir de ~/llm)
  <campagne>/contexte/        <- fill-*.out
  <campagne>/batteries/       <- results_*.json
  duels/<run_id>/run.json     <- métadonnées de chaque run de duel
  humaneval/<test>/humaneval-summary.json
```

```bash
python3 bench_index.py              # miroir + parsing + écriture de l'index
python3 bench_index.py --no-mirror  # ne lit que ce qui est déjà dans le vault
python3 bench_index.py --print      # affiche l'index au lieu de l'écrire
```

Variables d'environnement : `BENCH_VAULT_ROOT` (défaut
`/home/gab/NAS/AgentsMirror/vaults/personnel`), `BENCH_DOCS`, `BENCH_WIKI`, `BENCH_LOG_SRC`
(défaut `~/llm`), `BENCH_MACHINE`, `BENCH_CAMPAIGN`, `RUNS_DIR`.

La page **`/benchmarks`** (lien « 📊 Benchmarks » dans l'en-tête de l'UI, et lien depuis
`/compare`) rend cet index : duels HTML (durée, tok/s, prompts aboutis), vitesse & VRAM,
batteries de 13 petites tâches (score, latence médiane, échecs), HumanEval et remplissages
de contexte. Elle lit `index.json` dans le vault et retombe sur la copie locale
`bench_index.json` si le NAS n'est pas monté.

**Méthode des chiffres de vitesse** : le bench envoie **3 fois le même prompt** (489 tokens
en entrée, 128 en sortie) ; on garde la moyenne des `eval time` du log serveur, seule
façon d'obtenir des valeurs comparables entre modèles. Un log sans mesure de prefill
(HumanEval, remplissage de contexte) est ignoré : ce n'est pas un bench de vitesse.

## Suivi en direct

Pendant un tir, l'UI affiche dans la carte **« Suivi en direct »** :

- le **texte** produit par le LLM, token par token (défilement auto, bouton *agrandir*) ;
- le **raisonnement** (`reasoning_content`) s'il y en a — vide en principe, puisque
  `enable_thinking=false` ; le bloc n'apparaît que s'il y a du contenu ;
- une ligne d'état : `#id titre · mode · durée · ≈tok/s · ≈tokens · caractères · max_tokens · fin: stop` ;
- un **aperçu HTML live** (case à cocher, désactivée par défaut) : la page en cours
  d'écriture est injectée dans une `<iframe sandbox>` et rechargée au plus une fois
  toutes les 1,5 s. Le HTML partiel est refermé à la volée (`</body></html>`) — les
  prompts canvas/WebGL commencent donc à s'animer **pendant** la génération ;
- **copier le texte** (utile pour récupérer une réponse qui ne finit jamais).

Côté serveur : le tir part avec `"stream": true` + `stream_options.include_usage`
(pour garder les vrais compteurs de tokens), les morceaux sont accumulés dans un
`LiveStore` borné (`LIVE_MAX_CHARS`, défaut 400 000 caractères) et l'UI ne demande
que les incréments depuis ses curseurs — elle ne reçoit jamais deux fois le même texte.

Cas particuliers :

- llama.cpp absent / streaming refusé **avant** le premier morceau → repli automatique
  sur un tir en un bloc (le journal le dit), le prompt n'est pas perdu ;
- streaming coupé **après** le premier morceau → statut `error`, mais le texte déjà
  produit est conservé dans `<id>_<slug>/partial.txt` et reste affiché ;
- `LLM_STREAM=0` → comportement v1.1 (un bloc à la fin) ; le panneau se remplit alors
  d'un coup au lieu de progresser.

## Le cœur : comment un prompt est tiré

1. **Attente d'un slot libre** — boucle sur `GET /slots` tant que `is_processing`
   est vrai (poll 5 s), avec « en attente du slot » dans l'UI. Un seul serveur,
   **un seul slot : jamais deux inférences en parallèle** (un seul worker, séquentiel).
2. **Remise à zéro du contexte** (les trois mesures à chaque tir) :
   - `messages` = exactement 1 message système + 1 message utilisateur, jamais d'historique ;
   - `"cache_prompt": false` ;
   - `POST /slots/0?action=erase` en best-effort ; si 404/405, le journal note
     `slot erase non supporté — stateless par construction` et le run continue.
3. **POST `/v1/chat/completions`** (streaming SSE depuis la v1.2 — `LLM_STREAM=0` pour
   revenir au tir en un bloc — timeout 1800 s) avec les **réglages du run**
   (`temperature`, `top_p`, `max_tokens`, `enable_thinking` — voir « Réglages du run »),
   `chat_template_kwargs.enable_thinking=false` par défaut — **obligatoire**, sinon la réponse
   part dans `reasoning_content` et `content` peut revenir vide.
   Les morceaux reçus alimentent le suivi en direct (voir « Suivi en direct ») ; le
   `content` final reconstitué est strictement celui du tir en un bloc.
4. **Extraction HTML** : fences ``` retirées, du premier `<!DOCTYPE html` / `<html`
   au dernier `</html>`. Trouvé → `index.html` ; sinon `raw.txt` + statut `no_html`
   (le run continue).
5. **`meta.json`** : prompt envoyé, statut, tokens, tok/s
   (`timings.predicted_per_second`, sinon `completion_tokens / durée`), durée,
   timestamp, longueur de la réponse.
6. **Prompt suivant.** Erreur réseau / HTTP ≠ 200 / timeout → statut `error`
   enregistré, **la file continue**.

**Idempotence** : si `runs/<run_dir>/<id>_<slug>/index.html` existe déjà, le prompt
est marqué `déjà généré` et sauté — rien n'est réécrit. Un run identique
(même modèle, même contexte, mêmes réglages, mêmes ids) relancé dans les
30 minutes reprend le **même** dossier de run, ce qui rend le « relancer »
réellement idempotent ; passé ce délai, un nouveau dossier est créé.

## Mode `LLM_DRY_RUN=1`

Aucun contact avec llama.cpp : après 3 s, l'app renvoie un HTML bidon valide
(`<h1>DRY RUN prompt N</h1>`) plus un `timings` factice, avec le slug `dry-run`.
Le streaming est **simulé** (12 morceaux espacés) : le suivi en direct est donc
testable sans llama.cpp. Tout le reste du pipeline est exercé (nommage, écriture,
`meta.json`, progression, UI). C'est le mode utilisé par les tests automatisés.

## Dépannage

| Symptôme | Cause / solution |
|---|---|
| Pastille rouge « LLM éteint » | llama.cpp n'est pas lancé sur l'endpoint configuré. Démarrer le serveur, puis `curl http://127.0.0.1:8080/health` doit répondre `{"status":"ok"}`. |
| Pastille orange « slot occupé » | Une autre inférence tourne sur le serveur (1 seul slot). L'app attend automatiquement. |
| Le run semble figé | Normal pendant l'attente de slot ou pendant un long tir (timeout 1800 s). Vérifier le journal. |
| `content` vide, `finish_reason: length` | Le modèle a raisonné : `enable_thinking` doit être **dans** `chat_template_kwargs` (c'est le cas ici) ou `max_tokens` est trop bas. |
| Erreur « identité LLM indisponible » | `/v1/models` ou `/props` a échoué au démarrage du run : relancer une fois le serveur stable. |
| Le nom du modèle dans le dossier ne correspond pas | Normal : il vient de `GET /v1/models` **au moment du run**, jamais d'une supposition. |
| Journal : `slot erase non supporté — stateless par construction (HTTP 501)` | Information, pas une erreur : ce build de llama.cpp ne supporte pas l'effacement de slot. L'app est stateless par construction (`cache_prompt:false` + 2 messages), donc le tir est propre quand même. |
| Port déjà utilisé | `PORT=8899 python3 app.py`. |
| `OSError: [Errno 98] Address already in use` au lancement | Le service `prompt-duel` tourne **déjà** sur 8791 : l'app est donc déjà en ligne sur `http://127.0.0.1:8791`, il n'y a rien à relancer. Pour la lancer à la main : `systemctl --user stop prompt-duel` puis `python3 app.py`. Ou `PORT=8792 python3 app.py`. |
| Repartir de zéro sur un run | Supprimer le dossier de run concerné (ou juste les `index.html` à régénérer). |

## Fichiers

- `app.py` — serveur HTTP + API + worker d'exécution (stdlib seule).
- `prompts.json` — les 34 prompts en 5 groupes (format v2) : prompt en anglais, titres et
  catégories en français, 3 critères vérifiables par prompt.
- `README.md` — ce fichier.
- `runs/` — sorties générées.
- `benchmarks.py` — page `/benchmarks` : rend l'index unifié (stdlib seule).
- `bench_index.py` — construit l'index : scanne les runs, miroite les artefacts
  dans le vault, écrit `index.json` (vault + copie locale).
- `bench_index.json` — copie locale de l'index (généré ; versionné pour garder
  l'historique des résultats dans git).
