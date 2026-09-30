#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""page_graph.py — page /graph de Prompt Duel : tout ce qui a ete teste, en graphiques.

La page est rendue a partir de trois choses :
  * l'index unifie des tests (`bench_index.py`) : dict ``{"entries": [...], ...}``
    ou directement la liste des entrees ;
  * l'etat de visibilite (`visibility.load()`) : resultats masques / supprimes ;
  * les parametres de requete (`kind`, `variant`, `masques`) : filtres par test et
    par variante, affichage des resultats masques.

Regles tenues ici :
  * **stdlib uniquement**, comme le reste de l'app (pas de framework, pas de build) ;
  * la page reste **lisible sans JavaScript** : graphiques SVG inline (`charts.py`),
    tableaux de valeurs, filtres = formulaire GET, liste des masques en clair ;
  * le JavaScript present n'est qu'une **amelioration progressive** : masquer /
    reafficher / restaurer via ``POST /api/visibility`` puis rechargement, et
    cases a cocher de la legende (masquage cote client, non persiste) ;
  * **aucune ecriture** : cette page ne touche ni au vault ni a l'etat de visibilite ;
  * **jamais d'exception** : index vide, absent, liste au lieu d'un dict, entrees
    abimees -> la page reste valide.

Usage en ligne de commande (debug, rendu sur disque) :
    python3 page_graph.py --index /chemin/index.json --out /tmp/graph.html
"""
from urllib.parse import parse_qs

import html
import json
import os
import sys

import charts       # rendu SVG inline (stdlib pure)
import metrics      # registre des familles + agregats pour graphique
import visibility   # etat masque / supprime (lecture seule ici)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
EMPTY_HINT = ("Aucun index trouve. Genere-le depuis ~/apps/prompt-duel :<br>"
              "<code>python3 bench_index.py</code>")

# Metriques PRINCIPALES montrees en graphique, par famille (2 maximum par famille).
# Les familles absentes de l'index ne produisent simplement aucun graphique.
PRIMARY_METRICS = {
    "vitesse": ("pp_tps", "tg_tps"),
    "speed": ("pp", "tg"),
    "memoire": ("prefill_tps", "needle"),
    "context": ("prefill_tps", "tokens"),
    "intelligence": ("score_avg", "score_pct"),
    "battery": ("score_avg", "lat_median"),
    "humaneval": ("score_pct", "avg_latency_s"),
    "duel": ("tok_s", "prompts_ok"),
    "vram": ("total_gb", "peak_mib"),
}

MAX_CHARTS = 10      # garde-fou : au-dela, la page devient un mur de graphiques
MAX_TABLE_ROWS = 400  # garde-fou des tableaux de valeurs
MAX_RES_ROWS = 600    # garde-fou du tableau de resultats

# Habillage : memes variables que /benchmarks et l'accueil (theme sombre).
CSS = """
:root{--bg:#0d0f14;--bg2:#141821;--fg:#e6e9ef;--dim:#8b93a5;--line:#242a36;--acc:#4da3ff;--ok:#4ddb8b;--err:#ff6b6b;--warn:#ffc85c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:14px 18px;border-bottom:1px solid var(--line);background:var(--bg2);position:sticky;top:0;z-index:5;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:600}
h2{font-size:14px;margin:0}
h3{font-size:13px;margin:14px 0 4px}
.dim{color:var(--dim);font-size:12.5px}
a{color:var(--acc);text-decoration:none}
a:hover{text-decoration:underline}
.tabs{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.navlink{color:var(--acc);text-decoration:none;font-size:12.5px;border:1px solid var(--line);border-radius:6px;padding:4px 8px}
.navlink:hover{background:#171c26}
.navlink[aria-current=page]{background:#171c26;border-color:var(--acc);color:var(--fg)}
main{padding:18px;max-width:1500px;margin:0 auto}
section.card{background:var(--bg2);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:16px}
section.card>h2{margin-bottom:8px;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:6px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--dim);font-weight:500}
td.n{text-align:right;font-variant-numeric:tabular-nums}
tr:hover td{background:#171c26}
tr.masque td{opacity:.55}
tr.masque td:first-child{text-decoration:line-through}
.tag{display:inline-block;border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:11px;color:var(--dim)}
.ok{color:var(--ok)}.err{color:var(--err)}.warn{color:var(--warn)}
.scroll{overflow-x:auto}
.chart-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;padding:6px;background:#0f131b}
.chart-values{margin-top:8px}
button{background:#1b2130;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:3px 9px;font:inherit;font-size:12px;cursor:pointer}
button:hover{background:#232b3d}
button[disabled]{opacity:.5;cursor:default}
button.danger:hover{border-color:var(--err);color:var(--err)}
input[type=checkbox]{accent-color:var(--acc);margin:0;cursor:pointer}
fieldset{border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:0;min-width:190px}
legend{color:var(--dim);font-size:12px;padding:0 5px}
.filtres{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start}
.filtre-list{max-height:180px;overflow:auto;display:flex;flex-direction:column;gap:3px;font-size:12.5px;padding-right:4px}
.filtre-list label{display:flex;gap:6px;align-items:center;cursor:pointer}
.actions{display:flex;gap:6px;align-items:center}
.actions .dim{font-size:11px}
code{background:#0a0c11;border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:12px}
footer{padding:14px 18px;color:var(--dim);font-size:12px;border-top:1px solid var(--line)}
"""


# ---------------------------------------------------------------------------
# Petits utilitaires de rendu
# ---------------------------------------------------------------------------

def _esc(value):
    """Echappe tout texte injecte dans la page."""
    return html.escape("" if value is None else str(value))


def _num(value, dec=1, suffix=""):
    """Nombre lisible (« 12,3 t/s ») ; rien d'exploitable -> « - »."""
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    return f"{number:,.{dec}f}".replace(",", " ").replace(".", ",") + suffix


def _date(value):
    """Date courte (JJ/MM HH:MM) ; la valeur brute est rendue si illisible."""
    if not value:
        return "-"
    text = str(value)
    return text[8:10] + "/" + text[5:7] + " " + text[11:16] if len(text) >= 16 and text[4] == "-" else text[:16]


def _kind_label(kind):
    """Libelle FR d'une famille de test."""
    canon = metrics.canon_kind(kind)
    spec = metrics.KINDS.get(canon) or {}
    return str(spec.get("label") or canon or "autre")


def _entries_of(index):
    """Liste des entrees d'un index (dict, liste, ou rien)."""
    if isinstance(index, dict):
        raw = index.get("entries")
        return [e for e in raw if isinstance(e, dict)] if isinstance(raw, list) else []
    if isinstance(index, (list, tuple)):
        return [e for e in index if isinstance(e, dict)]
    return []


def _source_of(index):
    """Chemin de l'index (pour l'en-tete), si connu."""
    if isinstance(index, dict):
        return str(index.get("_source") or index.get("source") or "")
    return ""


def _variant_of(entry):
    """Variante (reglages) d'une entree, jamais None."""
    variant = entry.get("variant") if isinstance(entry, dict) else None
    if isinstance(variant, dict):
        return variant
    try:
        return metrics.entry_variant(entry)
    except Exception:  # entree abimee : jamais d'exception sur une page
        return {}


def _variant_label(entry):
    variant = _variant_of(entry)
    return str(variant.get("label") or variant.get("key") or "standard")


def _group_label(entry):
    """Libelle du test (axe X) d'une entree."""
    try:
        return str(metrics.group_of(entry)[1])
    except Exception:
        return str(entry.get("test_id") or entry.get("kind") or "autre")


def _entry_id(entry):
    if entry.get("entry_id"):
        return str(entry["entry_id"])
    try:
        return str(metrics.entry_id(entry))
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Parametres de requete (filtres)
# ---------------------------------------------------------------------------

def parse_query(query):
    """Normalise les parametres : ``{"kinds", "variants", "show_hidden", "raw"}``.

    Accepte ``None``, une chaine de requete (« ?kind=duel&masques=1 ») ou le dict
    de ``urllib.parse.parse_qs`` (valeurs en listes). Les alias FR sont toleres.
    """
    raw = {}
    if isinstance(query, str):
        raw = parse_qs(query.lstrip("?"), keep_blank_values=True)
    elif isinstance(query, dict):
        for key, value in query.items():
            values = value if isinstance(value, (list, tuple)) else [value]
            raw[str(key)] = ["" if v is None else str(v) for v in values]
    kinds, variants = [], []
    show_hidden = False
    for key, values in raw.items():
        low = key.strip().lower()
        for value in values:
            text = value.strip()
            if low in ("kind", "kinds", "test", "tests", "famille", "familles"):
                if text and text not in kinds:
                    kinds.append(text)
            elif low in ("variant", "variants", "variante", "variantes", "reglages"):
                if text and text not in variants:
                    variants.append(text)
            elif low in ("masques", "masque", "hidden", "masquee"):
                # case a cocher : « 1 », « on », « true » ou valeur vide = vrai
                show_hidden = text.lower() not in ("0", "non", "false", "off", "no")
    return {"kinds": kinds, "variants": variants, "show_hidden": show_hidden, "raw": raw}


def _passe_filtres(entry, kinds, variants):
    """L'entree passe-t-elle les filtres par test et par variante ?"""
    if kinds:
        canon = metrics.canon_kind(entry.get("kind"))
        allowed = {metrics.canon_kind(k) for k in kinds}
        if canon not in allowed and str(entry.get("kind")) not in kinds:
            return False
    if variants:
        key = str(_variant_of(entry).get("key") or "standard")
        if key not in variants:
            return False
    return True


# ---------------------------------------------------------------------------
# Vue calculee (separee du HTML : testable seule)
# ---------------------------------------------------------------------------

def _has_points(data):
    for serie in (data or {}).get("series") or []:
        if serie.get("points"):
            return True
    return False


def _familles_presentes(entries):
    """Familles reellement presentes, dans l'ordre d'affichage du registre."""
    vues = {}
    for entry in entries:
        canon = metrics.canon_kind(entry.get("kind"))
        vues.setdefault(canon, 0)
        vues[canon] += 1
    return sorted(vues, key=lambda k: (metrics.KIND_RANK.get(k, 9), str(k)))


def _charts_for(entries, familles):
    """Un graphique par metrique principale presente (au plus MAX_CHARTS)."""
    out = []
    for kind in familles:
        for key in PRIMARY_METRICS.get(kind, ()):
            if len(out) >= MAX_CHARTS:
                return out
            try:
                data = metrics.series_for_chart(entries, metric_key=key, kind=kind)
            except Exception:  # donnee abimee : on saute ce graphique, pas la page
                continue
            if not _has_points(data):
                continue
            svg = charts.grouped_bars(data)
            if not svg:
                continue
            out.append({
                "kind": kind,
                "kind_label": _kind_label(kind),
                "metric": data.get("metric") or {"key": key, "label": key, "unit": ""},
                "data": data,
                "svg": svg,
                "legend": charts.legend(data),
                "points": sum(len(s.get("points") or []) for s in data.get("series") or []),
            })
    return out


def _duration_data(entries, familles, charts_list):
    """Donnees du graphique des durees : le premier lot qui porte des durees."""
    candidates = [c["data"] for c in charts_list]
    for kind in familles:
        for key in (PRIMARY_METRICS.get(kind, ()) or ()):
            try:
                candidates.append(metrics.series_for_chart(entries, metric_key=key, kind=kind))
            except Exception:
                continue
    for data in candidates:
        for serie in data.get("series") or []:
            for point in serie.get("points") or []:
                if point.get("duration_s") is not None:
                    return data
    return None


def build_view(index, state=None, query=None):
    """Vue complete de la page /graph (donnees + SVG), sans une ligne de HTML.

    -> {"entries","visibles","masques","supprimes","familles","variantes",
        "charts","duree_svg","filtres","counts","source","index_error"}
    """
    filtres = parse_query(query)
    brut = _entries_of(index)
    index_error = index.get("error") if isinstance(index, dict) else None
    try:
        norm = metrics.normalize_entries(brut)
    except Exception:  # index abime : on prefere une page vide a une page morte
        norm = []
        index_error = index_error or "index illisible"

    if state is None:
        etat = visibility.load()
    else:
        etat = visibility.normalize_state(state) if isinstance(state, dict) else {}
    etat = etat if isinstance(etat, dict) else {}
    # Les filtres de SERIES persistes ne s'appliquent pas ici : sur cette page, les
    # filtres viennent de l'URL (visibles, modifiables sans JavaScript).
    etat_visible = {"hidden": etat.get("hidden") or [], "deleted": etat.get("deleted") or [],
                    "filters": {}}

    if filtres["show_hidden"]:
        gardees = list(norm)
    else:
        gardees = visibility.apply(etat_visible, norm)
    visibles = [e for e in gardees if _passe_filtres(e, filtres["kinds"], filtres["variants"])]

    familles = _familles_presentes(visibles)
    listes = _charts_for(visibles, familles)

    # Variantes presentes (filtre) : tous les libelles, tries par cle.
    variantes = {}
    for entry in norm:
        variant = _variant_of(entry)
        key = str(variant.get("key") or "standard")
        variantes.setdefault(key, str(variant.get("label") or key))
    variantes = dict(sorted(variantes.items()))

    # Familles presentes dans l'index COMPLET (le filtre doit rester utilisable
    # meme si le filtre courant ne laisse rien passer).
    familles_filtre = {}
    for entry in norm:
        canon = metrics.canon_kind(entry.get("kind"))
        familles_filtre.setdefault(canon, 0)
        familles_filtre[canon] += 1
    familles_filtre = sorted(familles_filtre.items(),
                             key=lambda kv: (metrics.KIND_RANK.get(kv[0], 9), str(kv[0])))

    lookup = {}
    for entry in norm:
        eid = _entry_id(entry)
        if eid and eid not in lookup:
            lookup[eid] = entry

    def _liste(ids):
        """(id, entree|None) dans l'ordre de l'etat."""
        out = []
        for eid in ids or []:
            text = str(eid)
            out.append({"id": text, "entry": lookup.get(text)})
        return out

    masques = _liste(etat.get("hidden"))
    supprimes = _liste(etat.get("deleted"))

    duree_data = _duration_data(visibles, familles, listes)
    duree_svg = charts.duration_chart(duree_data) if duree_data else ""
    duree_legend = charts.legend(duree_data) if duree_data else ""

    return {
        "entries": norm,
        "visibles": visibles,
        "familles": familles,
        "familles_filtre": familles_filtre,
        "variantes": variantes,
        "charts": listes,
        "duree_svg": duree_svg,
        "duree_legend": duree_legend,
        "masques": masques,
        "supprimes": supprimes,
        "filtres": filtres,
        "source": _source_of(index),
        "index_error": index_error,
        "counts": {
            "total": len(norm),
            "visibles": len(visibles),
            "masques": len(masques),
            "supprimes": len(supprimes),
        },
    }


# ---------------------------------------------------------------------------
# Fragments HTML
# ---------------------------------------------------------------------------

def _filtre_liste(nom, entrees):
    """Liste de cases a cocher (formulaire GET) : (value, libelle, compteur)."""
    return f'<div class="filtre-list">{"".join(entrees)}</div>' if entrees else '<p class="dim">aucun</p>'


def _filtres_form(view):
    """Formulaire GET : filtres par test et par variante, + resultats masques."""
    filtres = view["filtres"]
    coches = set(filtres["kinds"])
    cases_familles = []
    for kind, count in view["familles_filtre"]:
        cases_familles.append(
            f'<label><input type="checkbox" name="kind" value="{_esc(kind)}"'
            f'{" checked" if kind in coches else ""}> {_esc(_kind_label(kind))} '
            f'<span class="dim">({count})</span></label>'
        )
    coches_v = set(filtres["variants"])
    cases_variantes = []
    for key, label in view["variantes"].items():
        cases_variantes.append(
            f'<label><input type="checkbox" name="variant" value="{_esc(key)}"'
            f'{" checked" if key in coches_v else ""}> {_esc(label)}</label>'
        )
    masques_coche = ' checked' if filtres["show_hidden"] else ''
    return (
        '<section class="card"><h2>Filtres <span class="dim">'
        'par test et par variante &middot; fonctionnent sans JavaScript (formulaire GET)</span></h2>'
        '<form class="filtres" method="get" action="/graph">'
        '<fieldset><legend>Test (famille)</legend>'
        f'{_filtre_liste("kind", cases_familles)}</fieldset>'
        '<fieldset><legend>Variante (r&eacute;glages)</legend>'
        f'{_filtre_liste("variant", cases_variantes)}</fieldset>'
        '<fieldset><legend>Visibilit&eacute;</legend>'
        f'<label class="dim"><input type="checkbox" name="masques" value="1"{masques_coche}> '
        'afficher les r&eacute;sultats masqu&eacute;s</label>'
        '</fieldset>'
        '<div style="display:flex;flex-direction:column;gap:8px">'
        '<button type="submit">Filtrer</button>'
        '<a class="navlink" href="/graph">r&eacute;initialiser</a>'
        '</div></form></section>'
    )


def _data_table(data, max_rows=MAX_TABLE_ROWS):
    """Tableau de valeurs d'un graphique : lisible sans JavaScript."""
    lignes = []
    total = 0
    for serie in data.get("series") or []:
        label = serie.get("label") or serie.get("key")
        for point in serie.get("points") or []:
            total += 1
            if len(lignes) >= max_rows:
                continue
            statut = str(point.get("status") or "ok")
            cls = {"ok": "ok", "error": "err", "erreur": "err", "fail": "err"}.get(statut, "dim")
            lignes.append(
                f'<tr data-entry="{_esc(point.get("entry_id"))}">'
                f'<td>{_esc(label)}</td>'
                f'<td>{_esc(point.get("group"))}</td>'
                f'<td class="n">{_num(point.get("value"), 2)}</td>'
                f'<td class="n">{_num(point.get("duration_s"), 1, " s")}</td>'
                f'<td class="dim">{_esc(_date(point.get("date")))}</td>'
                f'<td class="{cls}">{_esc(statut)}</td></tr>'
            )
    if not lignes:
        return '<p class="dim">aucune valeur chiffr&eacute;e</p>'
    note = (f'<p class="dim">tableau tronqu&eacute; : {len(lignes)} lignes sur {total}.</p>'
            if total > len(lignes) else "")
    return ('<div class="scroll chart-values"><table><thead><tr>'
            '<th>s&eacute;rie</th><th>test</th><th>valeur</th><th>dur&eacute;e</th>'
            '<th>date</th><th>statut</th></tr></thead><tbody>'
            + "".join(lignes) + '</tbody></table></div>' + note)


def _chart_card(chart):
    """Une carte : titre (metrique + unite), SVG, legende, tableau de valeurs."""
    metric = chart["metric"]
    unite = f' {_esc(metric.get("unit"))}' if metric.get("unit") else ""
    sens = {"low": "plus bas = mieux", "high": "plus haut = mieux"}.get(
        str(metric.get("better") or "").lower(), "")
    return (
        f'<section class="card"><h2>{_esc(metric.get("label"))}'
        f'<span class="dim">{unite} &middot; {_esc(chart["kind_label"])} &middot; '
        f'{chart["points"]} point(s){" &middot; " + sens if sens else ""}</span></h2>'
        f'<div class="chart-wrap">{chart["svg"]}</div>'
        f'{chart["legend"]}'
        f'<details><summary class="dim">tableau des valeurs</summary>'
        f'{_data_table(chart["data"])}</details></section>'
    )


def _res_cellules(entry, masque):
    """Cellules du tableau de resultats (sans la colonne d'action)."""
    eid = _entry_id(entry)
    return [
        f'<td class="dim">{_esc(_date(entry.get("date")))}</td>',
        f'<td>{_esc(entry.get("model") or entry.get("model_slug"))}'
        f'<br><span class="dim">{_esc(entry.get("label") or "")}</span></td>',
        f'<td>{_esc(_group_label(entry))}</td>',
        f'<td><span class="tag">{_esc(_kind_label(entry.get("kind")))}</span></td>',
        f'<td>{_esc(_variant_label(entry))}</td>',
        f'<td class="n">{_num(entry.get("duration_s"), 1, " s")}</td>',
        f'<td class="{"dim" if not masque else "warn"}">{_esc("masqu&eacute;" if masque else (entry.get("status") or "ok"))}</td>',
    ]


def _resultats_table(view):
    """Tableau des resultats visibles : une ligne par execution, bouton « Masquer »."""
    visibles = sorted(view["visibles"], key=lambda e: str(e.get("date") or ""), reverse=True)
    masques = {m["id"] for m in view["masques"]} | {s["id"] for s in view["supprimes"]}
    lignes = []
    for entry in visibles[:MAX_RES_ROWS]:
        eid = _entry_id(entry)
        est_masque = eid in masques
        attrs = f' data-entry="{_esc(eid)}"'
        if est_masque:
            attrs += ' class="masque"'
        cellules = _res_cellules(entry, est_masque)
        if est_masque:
            action = (f'<td class="actions"><button data-vis="unhide" data-id="{_esc(eid)}"'
                      f'>R&eacute;afficher</button></td>')
        else:
            action = (f'<td class="actions"><button class="danger" data-vis="hide" '
                      f'data-id="{_esc(eid)}">Masquer</button></td>')
        lignes.append(f'<tr{attrs}>' + "".join(cellules) + action + '</tr>')
    if not lignes:
        return ('<section class="card"><h2>R&eacute;sultats</h2>'
                '<p class="dim">aucun r&eacute;sultat &agrave; afficher avec ces filtres.</p></section>')
    note = (f'<p class="dim">tableau tronqu&eacute; : {min(len(visibles), MAX_RES_ROWS)} lignes '
            f'sur {len(visibles)}.</p>' if len(visibles) > MAX_RES_ROWS else "")
    return (
        '<section class="card" id="resultats-card"><h2>R&eacute;sultats <span class="dim">'
        f'{len(visibles)} ligne(s) &middot; « Masquer » retire la ligne de toutes les vues '
        '(r&eacute;cup&eacute;rable plus bas)</span></h2>'
        '<div class="scroll"><table id="resultats"><thead><tr>'
        '<th>date</th><th>mod&egrave;le</th><th>test</th><th>famille</th><th>variante</th>'
        '<th>dur&eacute;e</th><th>statut</th><th></th></tr></thead><tbody>'
        + "".join(lignes) + '</tbody></table></div>' + note + '</section>'
    )


def _masques_table(view):
    """Liste des resultats masques / supprimes, avec boutons de restauration."""
    masques, supprimes = view["masques"], view["supprimes"]
    if not masques and not supprimes:
        return ""
    lignes = []
    for item in masques:
        lignes.append(_masque_row(item, "masqu&eacute;", "unhide", "R&eacute;afficher"))
    for item in supprimes:
        lignes.append(_masque_row(item, "supprim&eacute;", "restore", "Restaurer"))
    return (
        '<section class="card" id="masques"><h2>R&eacute;sultats masqu&eacute;s <span class="dim">'
        f'{len(masques)} masqu&eacute;(s) &middot; {len(supprimes)} supprim&eacute;(s) &middot; '
        'masqu&eacute; = retir&eacute; des vues mais r&eacute;cup&eacute;rable ; '
        'supprim&eacute; = exclu (la source d\'origine n\'est pas touch&eacute;e)</span></h2>'
        '<div class="scroll"><table><thead><tr><th>&eacute;tat</th><th>date</th>'
        '<th>test</th><th>mod&egrave;le</th><th>variante</th><th>identifiant</th>'
        '<th></th></tr></thead><tbody>' + "".join(lignes) + '</tbody></table></div>'
        '<p id="vis-msg" class="dim"></p></section>'
    )


def _masque_row(item, etat, action, libelle):
    """Une ligne de la liste des masques (entree retrouvee dans l'index, ou non)."""
    entry, eid = item.get("entry"), item["id"]
    if entry:
        date = _esc(_date(entry.get("date")))
        modele = _esc(entry.get("model") or entry.get("model_slug") or "-")
        test = _esc(_group_label(entry))
        variante = _esc(_variant_label(entry))
    else:
        date = modele = test = variante = '<span class="dim">introuvable dans l\'index</span>'
    return (
        f'<tr data-entry="{_esc(eid)}"><td>{etat}</td><td class="dim">{date}</td>'
        f'<td>{test}</td><td>{modele}</td><td>{variante}</td>'
        f'<td class="dim"><code>{_esc(eid)}</code></td>'
        f'<td class="actions"><button data-vis="{action}" data-id="{_esc(eid)}">{libelle}</button></td></tr>'
    )


# JS d'amelioration progressive : sans lui, la page reste lisible (SVG + tableaux).
SCRIPT = """
<script>
// Amelioration progressive : masquer / reafficher / restaurer (POST /api/visibility)
// et cases de la legende. Sans JavaScript la page reste lisible (SVG + tableaux).
(function () {
  function dire(txt, erreur) {
    var zone = document.getElementById('vis-msg');
    if (!zone) { if (txt) { console.log(txt); } return; }
    zone.textContent = txt || '';
    zone.className = erreur ? 'err' : 'dim';
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
    dire('action « ' + action + ' » en cours…', false);
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
      dire('echec de « ' + action + ' » : ' + err.message, true);
    });
  });
  // Legendes : masquer/afficher une serie cote client (aucune persistance).
  document.addEventListener('change', function (ev) {
    var box = ev.target;
    if (!box || !box.classList || !box.classList.contains('chart-toggle')) { return; }
    var cle = box.getAttribute('data-serie') || '';
    var carte = box.closest ? box.closest('section') : null;
    var barres = (carte || document).querySelectorAll('rect[data-serie]');
    for (var i = 0; i < barres.length; i++) {
      if (barres[i].getAttribute('data-serie') === cle) {
        barres[i].setAttribute('fill-opacity', box.checked ? '0.92' : '0.12');
      }
    }
  });
})();
</script>
"""


def _entete(view):
    """Barre d'onglets (meme habillage que /benchmarks et /compare)."""
    counts = view["counts"]
    return f"""<header>
  <h1>Graphique des tests <span class="dim">&middot; tout ce qui a &eacute;t&eacute; test&eacute;</span></h1>
  <div class="tabs">
    <a class="navlink" href="/">Prompt Duel</a>
    <a class="navlink" href="/benchmarks">Benchmarks</a>
    <a class="navlink" href="/graph" aria-current="page">Graphique</a>
    <a class="navlink" href="/compare">Comparaison</a>
  </div>
  <span class="dim">{counts['visibles']} r&eacute;sultat(s) affich&eacute;(s) &middot;
    {counts['masques']} masqu&eacute;(s) &middot; {counts['supprimes']} supprim&eacute;(s)</span>
</header>"""


def _resume(view):
    """Carte d'en-tete : ce que contient la page, et d'ou viennent les donnees."""
    counts = view["counts"]
    source = (f' &middot; source : <code>{_esc(view["source"])}</code>'
              if view["source"] else "")
    if view["index_error"]:
        avertissement = f'<p class="err">index : {_esc(view["index_error"])}</p>'
    elif not counts["total"]:
        avertissement = f'<p class="dim">{EMPTY_HINT}</p>'
    else:
        avertissement = ""
    return f"""<section class="card"><h2>&Ccedil;a a &eacute;t&eacute; test&eacute;
  <span class="dim">r&eacute;sultat chiffr&eacute; + temps que &ccedil;a a pris, par test et par r&eacute;glages</span></h2>
  <p class="dim">{counts['total']} entr&eacute;e(s) dans l'index &middot; {counts['visibles']} affich&eacute;e(s)
    &middot; {counts['masques']} masqu&eacute;e(s) &middot; {counts['supprimes']} supprim&eacute;e(s){source}</p>
  <p class="dim">Graphiques en SVG inline (aucune CDN, aucun JavaScript requis, info-bulles natives) ;
    un graphique par m&eacute;trique principale et un pour le temps pass&eacute;. Les r&eacute;sultats
    masqu&eacute;s disparaissent des vues mais restent r&eacute;cup&eacute;rables en bas de page.</p>
  {avertissement}</section>"""


def render_graph_page(index, state=None, query=None):
    """Page /graph complete (str UTF-8), sans effet de bord.

    `index` : dict de `bench_index.py` (ou liste d'entrees) ; `state` : etat de
    `visibility` ; `query` : parametres de requete (dict de `parse_qs` ou chaine).
    """
    view = build_view(index, state, query)
    counts = view["counts"]

    corps = [_resume(view)]
    if view["entries"]:
        corps.append(_filtres_form(view))

    if view["charts"]:
        corps.extend(_chart_card(chart) for chart in view["charts"])
    elif counts["total"]:
        corps.append('<section class="card"><h2>Graphiques</h2>'
                     '<p class="dim">aucune m&eacute;trique chiffr&eacute;e &agrave; tracer avec '
                     'ces filtres.</p></section>')

    carte_duree = ['<section class="card"><h2>Temps pass&eacute; '
                   '<span class="dim">par test et par r&eacute;glages &middot; '
                   'plus bas = mieux</span></h2>']
    if view["duree_svg"]:
        carte_duree.append(f'<div class="chart-wrap">{view["duree_svg"]}</div>')
        carte_duree.append(view["duree_legend"])
    else:
        carte_duree.append('<p class="dim">aucune dur&eacute;e enregistr&eacute;e.</p>')
    carte_duree.append("</section>")
    if counts["total"]:
        corps.append("".join(carte_duree))

    corps.append(_resultats_table(view))
    liste_masques = _masques_table(view)
    if liste_masques:
        corps.append(liste_masques)

    filtre_resume = ""
    if view["filtres"]["kinds"] or view["filtres"]["variants"]:
        filtre_resume = (" &middot; filtres actifs : "
                         + _esc(", ".join(view["filtres"]["kinds"] + view["filtres"]["variants"])))

    return f"""<!doctype html>
<html lang="fr"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Graphique des tests LLM &mdash; Prompt Duel</title>
<style>{CSS}</style></head><body>
{_entete(view)}
<main>{"".join(corps)}</main>
<footer>Prompt Duel 224 &middot; page <code>/graph</code>{filtre_resume} &middot;
index regenere par <code>python3 bench_index.py</code> (vault = source de verite,
copie locale en secours) &middot; etat de visibilite : <code>visibility.json</code>.</footer>
{SCRIPT}
</body></html>"""


# ---------------------------------------------------------------------------
# CLI de debug : rendu sur disque (jamais dans le vault)
# ---------------------------------------------------------------------------

def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="Rendu de la page /graph (debug)")
    ap.add_argument("--index", default=os.path.join(APP_DIR, "bench_index.json"),
                    help="fichier index.json (defaut : copie locale)")
    ap.add_argument("--out", default="", help="ecrit la page ici (defaut : stdout)")
    ap.add_argument("--kind", action="append", default=[], help="filtre famille (repetable)")
    ap.add_argument("--variant", action="append", default=[], help="filtre variante (repetable)")
    ap.add_argument("--masques", action="store_true", help="afficher les resultats masques")
    ap.add_argument("--no-stat", action="store_true",
                    help="n'affiche pas le resume chiffre sur stderr")
    args = ap.parse_args(argv)

    if args.index == "-":
        data = json.load(sys.stdin)
    else:
        with open(args.index, encoding="utf-8") as fh:
            data = json.load(fh)
    query = {}
    if args.kind:
        query["kind"] = args.kind
    if args.variant:
        query["variant"] = args.variant
    if args.masques:
        query["masques"] = "1"

    page = render_graph_page(data, visibility.load(), query)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(page)
        if not args.no_stat:
            view = build_view(data, visibility.load(), query)
            print(f"{args.out} : {len(page)} octets, {page.count('<svg')} <svg>, "
                  f"{page.count('<title>')} <title>, {page.count('chart-bar')} barres, "
                  f"{view['counts']['visibles']} resultat(s) affiche(s)", file=sys.stderr)
    else:
        sys.stdout.write(page)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
