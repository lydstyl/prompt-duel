#!/usr/bin/env python3
"""benchmarks.py — page /benchmarks de Prompt Duel : rend l'index unifie des tests.

Lit l'index JSON produit par bench_index.py (vault en priorite, copie locale en secours)
et affiche une page HTML autonome : duels HTML, vitesse/VRAM, batteries de taches,
HumanEval, remplissages de contexte.

v1.6 (etape E2) :
  * colonne « variante » (les reglages reellement utilises) quand elle existe ;
  * bouton « Masquer » par ligne (POST /api/visibility, amelioration progressive) ;
  * filtre « afficher les resultats masques » (lien GET, + bascule JavaScript) ;
  * liste des resultats masques / supprimes avec « Reafficher » / « Restaurer ».

Aucune dependance : stdlib uniquement, comme le reste de l'app.
"""
import html
import json
import os
from datetime import datetime
from urllib.parse import parse_qs

# metrics / visibility (meme dossier). Leur absence ne doit pas empecher la page
# de s'afficher : on degrade proprement (pas de variante, pas de masquage).
try:
    import metrics
except ImportError:  # pragma: no cover - depend de l'environnement
    metrics = None
try:
    import visibility
except ImportError:  # pragma: no cover - depend de l'environnement
    visibility = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Ordre de priorite des sources d'index
INDEX_CANDIDATES = [
    os.environ.get("BENCH_INDEX") or "",
    "/home/gab/NAS/AgentsMirror/vaults/personnel/documents/llm-benchmarks/index.json",
    os.path.join(APP_DIR, "bench_index.json"),
]
EMPTY_HINT = ("Aucun index trouve. Genere-le depuis ~/apps/prompt-duel :<br>"
              "<code>python3 bench_index.py</code>")

CSS = """
:root{--bg:#0d0f14;--bg2:#141821;--fg:#e6e9ef;--dim:#8b93a5;--line:#242a36;--acc:#4da3ff;--ok:#4ddb8b;--err:#ff6b6b;--warn:#ffc85c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:14px 18px;border-bottom:1px solid var(--line);background:var(--bg2);position:sticky;top:0;z-index:5;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:600}
h2{font-size:14px;margin:0}
.dim{color:var(--dim);font-size:12.5px}
a{color:var(--acc);text-decoration:none}
a:hover{text-decoration:underline}
.navlink{color:var(--acc);text-decoration:none;font-size:12.5px;border:1px solid var(--line);border-radius:6px;padding:4px 8px}
main{padding:18px;max-width:1500px;margin:0 auto}
section.card{background:var(--bg2);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:16px}
.card>h2{margin-bottom:10px;display:flex;gap:10px;align-items:baseline}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:6px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--dim);font-weight:500}
td.n{text-align:right;font-variant-numeric:tabular-nums}
tr:hover td{background:#171c26}
.tag{display:inline-block;border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:11px;color:var(--dim)}
.ok{color:var(--ok)}.err{color:var(--err)}.warn{color:var(--warn)}
.scroll{overflow-x:auto}
/* --- v1.6 : variante, masquage, filtres --- */
tr.res-masque{display:none}
td.variant{white-space:normal;max-width:260px;color:var(--dim);font-size:11.5px}
button{background:#1b2130;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:2px 8px;font:inherit;font-size:11.5px;cursor:pointer}
button:hover{background:#232b3d}
button[disabled]{opacity:.5;cursor:default}
button.danger:hover{border-color:var(--err);color:var(--err)}
input[type=checkbox]{accent-color:var(--acc);margin:0;cursor:pointer}
.filtre-bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:8px 0 2px;font-size:12.5px;color:var(--dim)}
.filtre-bar label{display:inline-flex;gap:6px;align-items:center;cursor:pointer}
code{background:#0a0c11;border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:12px}
footer{padding:14px 18px;color:var(--dim);font-size:12px;border-top:1px solid var(--line)}
"""


def _load_index():
    for path in INDEX_CANDIDATES:
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                data["_source"] = path
                return data
            except Exception as exc:
                return {"error": f"{path} illisible : {exc}"}
    return {"error": None}


def _esc(v):
    return html.escape("" if v is None else str(v))


def _num(v, dec=1, suffix=""):
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _esc(v)
    return f"{f:,.{dec}f}".replace(",", " ") + suffix


def _date(v):
    if not v:
        return "-"
    s = str(v)
    try:
        return datetime.fromisoformat(s).strftime("%d/%m %H:%M")
    except ValueError:
        return s[:16]


def _rows(entries, kind, model_keys, columns, row_fn):
    return [e for e in entries if e.get("kind") == kind]


# --------------------------------------------------------------- v1.6 : variante / masquage
def _state_of(state):
    """Etat de visibilite utilisable (charge le fichier si non fourni)."""
    if isinstance(state, dict):
        return visibility.normalize_state(state) if visibility else state
    if visibility is not None:
        try:
            return visibility.load()
        except Exception:  # etat illisible : on continue sans masquage
            return {"hidden": [], "deleted": [], "filters": {}}
    return {"hidden": [], "deleted": [], "filters": {}}


def _entry_id(entry):
    """Identifiant stable d'une entree (via metrics, sinon repli local)."""
    if not isinstance(entry, dict):
        return ""
    if entry.get("entry_id"):
        return str(entry["entry_id"])
    if metrics is not None:
        try:
            return str(metrics.entry_id(entry))
        except Exception:
            return ""
    source = entry.get("source") or entry.get("run_id") or ""
    return f"{entry.get('kind') or '?'}:{source}"


def _variant_label(entry):
    """Libelle de la variante (reglages) d'une entree, ou « standard »."""
    if not isinstance(entry, dict):
        return "standard"
    variant = entry.get("variant")
    if isinstance(variant, dict) and (variant.get("label") or variant.get("key")):
        return str(variant.get("label") or variant.get("key"))
    if metrics is not None:
        try:
            variant = metrics.entry_variant(entry)
            return str(variant.get("label") or variant.get("key") or "standard")
        except Exception:
            return "standard"
    return "standard"


def _avec_variante(famille):
    """Cette famille porte-t-elle de vraies variantes (autres que « standard ») ?"""
    return any(_variant_label(e) not in ("", "standard") for e in famille)


def _vis_sets(state, show_hidden):
    """(identifiants a marquer, masques, supprimes) selon l'etat de visibilite.

    Quand les masques sont affiches (`show_hidden`), plus rien n'est marque : les
    lignes redeviennent des lignes normales, mais les actions restent proposees.
    """
    hidden = {str(i) for i in (state.get("hidden") or [])}
    deleted = {str(i) for i in (state.get("deleted") or [])}
    marques = set() if show_hidden else (hidden | deleted)
    return marques, hidden, deleted


def _attrs(entry, masques):
    """Attributs du <tr> : identifiant d'entree + marque « masquee » (CSS)."""
    eid = _entry_id(entry)
    attrs = f' data-entry="{_esc(eid)}"'
    if eid and eid in masques:
        attrs += ' class="res-masque"'
    return attrs


def _action_cell(entry, hidden, deleted):
    """Bouton d'action d'une ligne : Masquer, Reafficher ou Restaurer."""
    eid = _entry_id(entry)
    if eid in deleted:
        return (f'<td><button data-vis="restore" data-id="{_esc(eid)}">'
                f'Restaurer</button></td>')
    if eid in hidden:
        return (f'<td><button data-vis="unhide" data-id="{_esc(eid)}">'
                f'R&eacute;afficher</button></td>')
    return (f'<td><button class="danger" data-vis="hide" data-id="{_esc(eid)}">'
            f'Masquer</button></td>')


def _variant_cell(entry, avec):
    """Cellule variante (les reglages reellement utilises)."""
    if not avec:
        return []
    label = _variant_label(entry)
    return [f'<td class="variant">{_esc(label)}</td>']


def _table(headers, rows, cls=""):
    """Tableau HTML. Une ligne = liste de cellules, ou (attributs, cellules)."""
    if not rows:
        return '<p class="dim">aucun resultat</p>'
    th = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    parts = []
    for row in rows:
        # liste de cellules, ou couple (attributs du <tr>, cellules)
        if isinstance(row, tuple) and len(row) == 2 and isinstance(row[0], str):
            attrs, cells = row
        else:
            attrs, cells = "", row
        parts.append(f"<tr{attrs}>" + "".join(cells) + "</tr>")
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{"".join(parts)}</tbody></table></div>'


def _kind_of_score(score, total):
    try:
        if float(score) >= float(total):
            return '<span class="ok">complet</span>'
    except (TypeError, ValueError):
        pass
    return '<span class="warn">partiel</span>'


def render_body(data, state=None, show_hidden=False, liens_requete=True):
    if data.get("error"):
        return f'<section class="card"><h2>Index</h2><p class="err">{_esc(data["error"])}</p></section>'
    entries = data.get("entries") or []
    if not entries:
        return f'<section class="card"><h2>Index</h2><p class="dim">{EMPTY_HINT}</p></section>'

    etat = _state_of(state)
    masques, hidden_ids, deleted_ids = _vis_sets(etat, show_hidden)
    n_masques = len(etat.get("hidden") or []) + len(etat.get("deleted") or [])
    n_visibles = sum(1 for e in entries if _entry_id(e) not in masques)

    out = []
    gen = _date(data.get("generated_at"))
    # Sans requete transmise par app.py, un lien « ?masques=1 » ne servirait a rien :
    # on n'affiche alors que la bascule JavaScript (fonctionnelle) et la mention /graph.
    if liens_requete:
        bascule = ('<span>sans JavaScript ? '
                   '<a href="/benchmarks?masques=1">afficher les masques</a> '
                   '&middot; <a href="/benchmarks">les cacher</a></span>')
    else:
        bascule = ('<span class="dim">sans JavaScript, les resultats masques restent '
                   'caches : ils sont listes en bas de page (identifiants) et '
                   'restaurables depuis <a href="/graph">/graph</a>.</span>')
    out.append(f'<section class="card"><h2>Index unifie des tests LLM</h2>'
               f'<p class="dim">genere le {gen} par <code>bench_index.py</code> &middot; '
               f'{len(entries)} entrees &middot; source : <code>{_esc(data.get("_source"))}</code></p>'
               f'<div class="filtre-bar"><label><input type="checkbox" id="voir-masques"'
               f'{" checked" if show_hidden else ""}> afficher les resultats masques</label>'
               f'<span>visibles : {n_visibles} &middot; masques / supprimes : {n_masques}</span>'
               f'{bascule}'
               f'<span><a href="/graph">voir les graphiques</a></span></div>'
               f'<p class="dim">Le vault est la source de verite (notes + <code>documents/llm-benchmarks/</code>) ; '
               f'cette page en est la vue visuelle. Un « run » de duel = un modele, un contexte, un jeu de reglages.</p>'
               f'</section>')

    # 1) Duels HTML (prompt-duel)
    duels = sorted([e for e in entries if e.get("kind") == "duel"],
                   key=lambda e: e.get("date") or "", reverse=True)
    if duels:
        avec = _avec_variante(duels)
        rows = []
        for e in duels:
            cells = [
                f'<td>{_date(e.get("date"))}</td>',
                f'<td>{_esc(e.get("model"))}</td>',
                *_variant_cell(e, avec),
                f'<td class="n">{_num(e.get("ctx"),0)}</td>',
                f'<td class="n">{_num(e.get("temperature"),1)}</td>',
                f'<td>{"oui" if e.get("thinking") else "non"}</td>',
                f'<td class="n">{_esc(e.get("prompts_ok"))}/{_esc(e.get("prompts_total"))}</td>',
                f'<td class="n">{_num(e.get("duration_s"),0," s")}</td>',
                f'<td class="n">{_num(e.get("tokens_out"),0)}</td>',
                f'<td class="n">{_num(e.get("tok_s"))}</td>',
                (f'<td class="n">{_esc(e.get("vram_total_gb"))} Go<br>'
                 f'<span class="dim">pic duel {_num(e.get("vram_peak_duel_mib"),0," MiB")}</span></td>'
                 if e.get("vram_total_gb") else '<td class="dim">-</td>'),
                f'<td><a href="/compare">comparer</a> &middot; <a href="{_esc(e.get("run_url"))}">fichiers</a></td>',
            ]
            rows.append((_attrs(e, masques), cells + [_action_cell(e, hidden_ids, deleted_ids)]))
        out.append('<section class="card"><h2>Duels HTML <span class="dim">'
                   '32+ prompts canvas/3D/jeu &rarr; une page HTML par prompt</span></h2>'
                   + _table(["date", "modele"] + (["variante"] if avec else [])
                            + ["ctx", "temp", "reasoning", "prompts", "duree",
                               "tokens sortie", "tok/s", "VRAM", "", ""], rows) + '</section>')

    # 1b) Empreinte VRAM (sampler 0,5 s + jalons de phase)
    vrams = sorted([e for e in entries if e.get("kind") == "vram"],
                   key=lambda e: (e.get("model") or "", -(e.get("ctx") or 0)))
    if vrams:
        avec = _avec_variante(vrams)
        rows = []
        for e in vrams:
            fits8 = ('<span class="ok">oui</span>' if e.get("fits_8gb") else
                     '<span class="err">non</span>')
            fits16 = ('<span class="ok">oui</span>' if e.get("fits_16gb") else
                      '<span class="err">non</span>')
            cells = [
                f'<td>{_esc(e.get("model"))}<br><span class="dim">{_esc(e.get("quant") or "")} '
                f'&middot; {_esc(e.get("label"))}</span></td>',
                *_variant_cell(e, avec),
                f'<td class="n">{_num(e.get("ctx"),0)}</td>',
                f'<td class="dim">{_esc(e.get("gpu"))}</td>',
                f'<td class="n">{_num(e.get("loaded_mib"),0)}</td>',
                f'<td class="n">{_num(e.get("peak_gen_mib"),0)}</td>',
                f'<td class="n">{_num(e.get("peak_prefill_mib"),0)}</td>',
                f'<td class="n">{_num(e.get("peak_duel_mib"),0)}</td>',
                f'<td class="n"><b>{_num(e.get("total_gb"),2)} Go</b></td>',
                f'<td>{fits8}</td>',
                f'<td>{fits16}</td>',
                f'<td class="n">{_num(e.get("tg"))}</td>',
                f'<td class="dim">{_esc(e.get("note") or "")}</td>',
            ]
            rows.append((_attrs(e, masques), cells + [_action_cell(e, hidden_ids, deleted_ids)]))
        out.append('<section class="card"><h2>Empreinte VRAM <span class="dim">'
                   'echantillonnage 0,5 s sur une seule carte &middot; pic = maximum mesure '
                   '(prefill et duel inclus)</span></h2>'
                   + _table(["modele / quant"] + (["variante"] if avec else [])
                            + ["ctx", "GPU", "charge MiB", "pic gen", "pic prefill",
                               "pic duel", "total", "8 Go ?", "16 Go ?", "TG t/s", "note", ""], rows)
                   + '<p class="dim">« charge » = mediane au repos apres <code>/health</code> ; '
                     '« pic » = maximum des echantillons sur la fenetre de phase. '
                     'Tient dans 8 Go = pic total &le; 7 800 MiB ; 16 Go = pic &le; 15 800 MiB '
                     '(marge volontairement large : le bureau occupe la carte d\'affichage).</p></section>')

    # 2) Vitesse & VRAM
    speeds = sorted([e for e in entries if e.get("kind") == "speed"],
                    key=lambda e: (e.get("model") or "", e.get("ctx") or 0))
    if speeds:
        avec = _avec_variante(speeds)
        rows = []
        for e in speeds:
            cells = [
                f'<td>{_esc(e.get("model"))}<br><span class="dim">{_esc(e.get("label"))}</span></td>',
                *_variant_cell(e, avec),
                f'<td class="n">{_num(e.get("ctx"),0)}</td>',
                f'<td class="n">{_num(e.get("tg"))}</td>',
                f'<td class="n">{_num(e.get("pp"),0)}</td>',
                f'<td class="n">{_esc(e.get("vram"))}</td>',
                f'<td class="dim">{_esc(e.get("note") or "")}</td>',
            ]
            rows.append((_attrs(e, masques), cells + [_action_cell(e, hidden_ids, deleted_ids)]))
        out.append('<section class="card"><h2>Vitesse &amp; VRAM <span class="dim">'
                   'llama-server .224 &middot; PP = prefill, TG = generation (KV q8_0)</span></h2>'
                   + _table(["modele"] + (["variante"] if avec else [])
                            + ["ctx", "TG t/s", "PP t/s", "VRAM 5060Ti/5080", "note", ""], rows)
                   + '</section>')

    # 3) Batteries de petites taches
    bat = sorted([e for e in entries if e.get("kind") == "battery"],
                 key=lambda e: (e.get("model") or "", e.get("mode") or ""))
    if bat:
        avec = _avec_variante(bat)
        rows = []
        for e in bat:
            fails = e.get("fails") or []
            cells = [
                f'<td>{_esc(e.get("model"))}<br><span class="dim">{_esc(e.get("label"))}</span></td>',
                *_variant_cell(e, avec),
                f'<td class="tag">{_esc(e.get("mode"))}</td>',
                f'<td class="n">{_esc(e.get("score"))}/{_esc(e.get("tasks"))}</td>',
                f'<td>{_kind_of_score(e.get("score"), e.get("tasks"))}</td>',
                f'<td class="n">{_num(e.get("lat_median"),2," s")}</td>',
                f'<td class="n">{_num(e.get("lat_max"),1," s")}</td>',
                f'<td class="dim">{_esc(", ".join(fails)) if fails else "aucun"}</td>',
            ]
            rows.append((_attrs(e, masques), cells + [_action_cell(e, hidden_ids, deleted_ids)]))
        out.append('<section class="card"><h2>Batteries de petites taches <span class="dim">'
                   '13 taches FR reelles, scoring verifie par code</span></h2>'
                   + _table(["modele"] + (["variante"] if avec else [])
                            + ["mode", "score", "", "latence med", "latence max", "echecs", ""], rows)
                   + '<p class="dim">« mode » = reasoning coupe (rapide) ou reasoning medium. '
                     'Les 2 echecs communs en mode rapide (prorata, jour de la semaine) sont une limite du '
                     'mode sans reflexion, pas d\'un modele.</p></section>')

    # 4) HumanEval
    he = sorted([e for e in entries if e.get("kind") == "humaneval"],
                key=lambda e: (-(e.get("total") or 0), -(e.get("score_pct") or 0)))
    if he:
        avec = _avec_variante(he)
        rows = []
        for e in he:
            cells = [
                f'<td>{_esc(e.get("model"))}</td>',
                *_variant_cell(e, avec),
                f'<td class="dim">{_esc(e.get("config") or "")}</td>',
                f'<td class="n">{_esc(e.get("passed"))}/{_esc(e.get("total"))}</td>',
                f'<td class="n">{_num(e.get("score_pct"))} %</td>',
                f'<td class="n">{_num(e.get("avg_latency_s"),2," s")}</td>',
                f'<td class="n">{_num(e.get("wall_s"),0," s")}</td>',
                f'<td class="dim">{_date(e.get("date"))}</td>',
            ]
            rows.append((_attrs(e, masques), cells + [_action_cell(e, hidden_ids, deleted_ids)]))
        out.append('<section class="card"><h2>HumanEval <span class="dim">'
                   'sous-ensemble 0-49, extraction validee 5/5 avant le run, reasoning coupe</span></h2>'
                   + _table(["modele"] + (["variante"] if avec else [])
                            + ["config", "passes", "score", "latence/probleme", "temps mural", "date", ""], rows)
                   + '<p class="dim">Tri : jeux complets (164 problemes) d\'abord, puis score. '
                     'Les sous-ensembles courts (5, 23 ou 38 problemes) ne se comparent pas aux runs complets.</p>'
                   + '</section>')

    # 5) Contexte (remplissages reels)
    ctxs = sorted([e for e in entries if e.get("kind") == "context"],
                  key=lambda e: -(e.get("ctx") or 0))
    if ctxs:
        avec = _avec_variante(ctxs)
        rows = []
        for e in ctxs:
            ok = e.get("needle")
            verdict = ('<span class="ok">valide</span>' if ok else
                       '<span class="err">echec</span>' if e.get("status") == "error" else
                       '<span class="warn">-</span>')
            cells = [
                f'<td>{_esc(e.get("model"))}</td>',
                *_variant_cell(e, avec),
                f'<td class="n">{_num(e.get("ctx"),0)}</td>',
                f'<td class="n">{_num(e.get("tokens"),0)}</td>',
                f'<td>{verdict}</td>',
                f'<td class="n">{_num(e.get("prefill_tps"),0)}</td>',
                f'<td class="n">{_esc(e.get("peak_vram"))}</td>',
                f'<td class="dim">{_esc(e.get("note") or "")}</td>',
            ]
            rows.append((_attrs(e, masques), cells + [_action_cell(e, hidden_ids, deleted_ids)]))
        out.append('<section class="card"><h2>Contexte long <span class="dim">'
                   'remplissage reel + needle a 85 % de profondeur (jamais un simple /health)</span></h2>'
                   + _table(["modele"] + (["variante"] if avec else [])
                            + ["ctx", "tokens ingeres", "needle", "prefill t/s", "pic VRAM", "note", ""], rows)
                   + '</section>')

    # 6) Notes du vault (liens sortants) et artefacts
    notes = data.get("vault_notes") or []
    if notes:
        items = "".join(f'<li><code>{_esc(n.get("path"))}</code> — {_esc(n.get("title"))}</li>' for n in notes)
        out.append(f'<section class="card"><h2>Notes du vault</h2><ul class="dim" style="margin:6px 0 0 0">{items}</ul>'
                   f'<p class="dim">Chemin du vault sur le .224 : '
                   f'<code>/home/gab/NAS/AgentsMirror/vaults/personnel/wiki/casquettes/developpeur/llm-benchmarks/</code></p>'
                   f'</section>')

    # 7) Resultats masques / supprimes (v1.6) : recuperables ici et depuis /graph
    out.append(_masques_bloc(etat, entries))
    return "".join(out)


def _masques_bloc(etat, entries):
    """Liste des resultats masques / supprimes, avec les boutons de restauration.

    « Masquee » = retiree des vues mais recuperable ; « supprimee » = tombstone
    (l'entree d'origine n'est pas touchee). Les boutons passent par
    `POST /api/visibility` (JavaScript) ; sans JavaScript, l'identifiant reste
    lisible et la restauration se fait depuis /graph ou en ligne de commande.
    """
    hidden = [str(i) for i in (etat.get("hidden") or [])]
    deleted = [str(i) for i in (etat.get("deleted") or [])]
    if not hidden and not deleted:
        return ""
    lookup = {}
    for entry in entries:
        eid = _entry_id(entry)
        if eid and eid not in lookup:
            lookup[eid] = entry
    lignes = []
    for ids, etat_txt, action, libelle in ((hidden, "masque", "unhide", "R&eacute;afficher"),
                                           (deleted, "supprime", "restore", "Restaurer")):
        for eid in ids:
            entry = lookup.get(eid)
            if entry is not None:
                desc = (f'{_esc(entry.get("model") or entry.get("model_slug") or "-")} '
                        f'<span class="dim">{_esc(entry.get("kind"))} &middot; '
                        f'{_esc(_date(entry.get("date")))} &middot; {_esc(_variant_label(entry))}</span>')
            else:
                desc = '<span class="dim">introuvable dans l\'index</span>'
            lignes.append(
                f'<tr data-entry="{_esc(eid)}"><td>{etat_txt}</td><td>{desc}</td>'
                f'<td class="dim"><code>{_esc(eid)}</code></td>'
                f'<td><button data-vis="{action}" data-id="{_esc(eid)}">{libelle}</button></td></tr>')
    return ('<section class="card" id="masques"><h2>Resultats masques <span class="dim">'
            f'{len(hidden)} masque(s) &middot; {len(deleted)} supprime(s) &middot; '
            'masque = retire des vues mais recuperable ; supprime = exclu (la source '
            "d'origine n'est pas touchee)</span></h2>"
            '<div class="scroll"><table><thead><tr><th>etat</th><th>resultat</th>'
            '<th>identifiant</th><th></th></tr></thead><tbody>'
            + "".join(lignes) + '</tbody></table></div>'
            '<p class="dim">Les graphiques et cette page excluent ces resultats ; '
            '<a href="/graph">/graph</a> permet de les reafficher ou de les restaurer.</p>'
            '<p id="vis-msg" class="dim"></p></section>')


def _show_hidden_from(query):
    """« afficher les resultats masques » : parametre `masques` (dict ou chaine)."""
    if isinstance(query, str):
        query = parse_qs(query.lstrip("?"))
    valeurs = []
    if isinstance(query, dict):
        for cle in ("masques", "masque", "hidden", "show_hidden"):
            brut = query.get(cle)
            if brut is None:
                continue
            valeurs += list(brut) if isinstance(brut, (list, tuple)) else [brut]
    for valeur in valeurs:
        if str(valeur).lower() not in ("0", "non", "false", "off", "no"):
            return True
    return False


# JavaScript d'amelioration progressive (facultatif) : bascule des lignes masquees
# et actions de visibilite. Sans JavaScript, la page reste lisible telle quelle.
SCRIPT = """
<script>
(function () {
  var box = document.getElementById('voir-masques');
  if (box) {
    box.addEventListener('change', function () {
      var lignes = document.querySelectorAll('tr.res-masque');
      for (var i = 0; i < lignes.length; i++) {
        lignes[i].style.display = box.checked ? 'table-row' : 'none';
      }
    });
  }
  document.addEventListener('click', function (ev) {
    var noeud = ev.target;
    var bouton = noeud && noeud.closest ? noeud.closest('[data-vis]') : null;
    if (!bouton) { return; }
    ev.preventDefault();
    var ids = (bouton.getAttribute('data-id') || '').split(',').filter(Boolean);
    var action = bouton.getAttribute('data-vis');
    if (!ids.length || !action) { return; }
    bouton.disabled = true;
    fetch('/api/visibility', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: action, ids: ids })
    }).then(function (rep) {
      if (!rep.ok) { throw new Error('HTTP ' + rep.status); }
      return rep.json();
    }).then(function () {
      location.reload();
    }).catch(function (err) {
      bouton.disabled = false;
      var zone = document.getElementById('vis-msg');
      if (zone) {
        zone.textContent = 'echec de « ' + action + ' » : ' + err.message;
        zone.className = 'err';
      } else {
        console.log('echec visibilite : ' + err.message);
      }
    });
  });
})();
</script>
"""


def render_benchmarks_page(state=None, query=None):
    """Page /benchmarks. `state`/`query` sont optionnels (appeles par app.py).

    Sans argument, l'etat de visibilite est lu sur disque et les resultats
    masques restent caches (avec la bascule JavaScript disponible dans la page).
    """
    data = _load_index()
    if not data.get("error") and not data.get("entries"):
        data = dict(data)
    show_hidden = _show_hidden_from(query)
    body = render_body(data, state=state, show_hidden=show_hidden,
                       liens_requete=query is not None)
    gen = _date(data.get("generated_at"))
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmarks LLM — Prompt Duel</title>
<style>{CSS}</style></head><body>
<header>
  <h1>Benchmarks LLM <span class="dim">&middot; index unifie des tests locaux</span></h1>
  <div class="spacer" style="flex:1"></div>
  <a class="navlink" href="/">&#8592; Prompt Duel</a>
  <a class="navlink" href="/graph">Graphique</a>
  <a class="navlink" href="/compare">Comparaison de runs</a>
  <span class="dim">index {gen}</span>
</header>
<main>{body}</main>
<footer>Prompt Duel 224 &middot; page <code>/benchmarks</code> &middot; index regenere par
<code>python3 bench_index.py</code> (vault = source de verite, copie locale en secours) &middot;
« Masquer » retire un resultat des vues ; il reste recuperable en bas de page ou sur
<code>/graph</code>.</footer>
{SCRIPT}
</body></html>""".encode("utf-8")


if __name__ == "__main__":
    import sys
    sys.stdout.write(render_benchmarks_page().decode("utf-8"))
