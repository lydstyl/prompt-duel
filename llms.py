#!/usr/bin/env python3
"""llms.py — page /llms de Prompt Duel : « tous les tests d'un LLM, cote a cote ».

Le probleme resolu : /graph et /benchmarks rangent les resultats par VARIANTE
(modele + ctx + MTP + thinking + temperature). Deux tirs du meme modele faits le
meme jour apparaissent donc comme deux « reglages » distincts, et trois tirs du
meme LLM a trois dates ne se regardent jamais ensemble.

Ici l'axe de selection est le **LLM** (le nom du modele, independant de la
quantification, du contexte et du mode de raisonnement) :

  1. etape 1 (`/llms` sans parametre) = catalogue des LLM presents dans l'index,
     avec le nombre de tests, la plage de dates et le nombre de variantes ;
  2. etape 2 (`/llms?llm=<cle>&llm=<cle>`) = 1 a 6 LLM : un tableau
     « test x LLM » (meilleure valeur mesuree par LLM, avec sa date et le nombre
     de mesures), puis, pour chaque LLM, le detail chronologique de TOUS ses
     tests et de TOUTES leurs mesures.

Aucune dependance : stdlib uniquement. Le module ne lit rien en dehors de
l'index qu'on lui passe (meme contrat que page_graph.py), et retombe proprement
si `metrics`/`visibility` manquent.

Regle de regroupement : la table MODELES ci-dessous associe a un nom de modele
une famille stable. Ce qui change d'un tir a l'autre (quantification, ctx, MTP,
thinking, machine) reste visible comme *variante* sous la famille, jamais
confondu avec un autre LLM.
"""
from __future__ import annotations

import html
import json
import re
import sys
import unicodedata
from urllib.parse import parse_qs

try:
    import metrics
except ImportError:  # pragma: no cover - depend de l'environnement
    metrics = None
try:
    import visibility
except ImportError:  # pragma: no cover - depend de l'environnement
    visibility = None

# CSS partagee avec /benchmarks (une seule feuille pour les pages de resultats).
try:
    from benchmarks import CSS as BASE_CSS
except Exception:  # pragma: no cover
    BASE_CSS = ""

APP_VERSION = "1.7.0"

# Combien de LLM on peut comparer d'un coup. Borne haute : au-dela le tableau
# devient illisible et la page ne sert plus a rien.
MIN_SELECTION = 1
MAX_SELECTION = 6

# Familles de LLM : (motif, cle, libelle). L'ORDRE COMPTE — le premier motif qui
# matche gagne, donc un derive (Swift-1.5, Uncensored) passe avant son modele de
# base, et FlashNext avant « qwen3.8 ».
MODELES = (
    (r"flash[\s._-]?next", "flashnext-177b", "FlashNext 177B"),
    (r"swift[\s._-]?1[.,]?5", "swift15-27b", "Swift-1.5-27B"),
    (r"bonsai", "bonsai-27b", "Bonsai 2 27B"),
    (r"k2[\s._-]?horizon", "k2-horizon", "K2 Horizon"),
    (r"thinkingcap", "thinkingcap-27b", "ThinkingCap 27B"),
    (r"qwen3[._-]?8[\s._-]?27b[\s._-]?uncensored", "qwen3-8-27b-uncensored", "Qwen3.8-27B Uncensored"),
    (r"qwen3[._-]?8[\s._-]?27b", "qwen3-8-27b", "Qwen3.8-27B"),
    (r"qwen3[._-]?6[\s._-]?35b", "qwen3-6-35b", "Qwen3.6-35B-A3B"),
    (r"qwen3[._-]?6[\s._-]?27b", "qwen3-6-27b", "Qwen3.6-27B"),
    (r"qwen3[._-]?8", "qwen3-8", "Qwen3.8 (taille non precisee)"),
)

# Prefixes de machine / d'auteur : ils ne disent rien du modele.
MACHINES = re.compile(r"^\s*(gabriel|louis|marie|harry|ron|luna|hermione)\s*[\(\-–:]", re.I)

# RATTACHEMENTS : des artefacts dont le nom ne porte pas le nom du modele, mais
# qui ne peuvent etre qu'UN SEUL LLM du parc. Chaque regle est justifiee et
# AFFICHEE sous la famille concernee : jamais de fusion silencieuse. Un artefact
# qui ne tombe dans aucune regle reste `autre-…` (nom de fichier visible tel quel)
# et se range, au catalogue, dans le bloc « mesures sans modele identifie ».
RATTACHEMENTS = (
    (r"35b[\s._-]?a3b", "qwen3-6-35b",
     "« 35B-A3B » est le nom complet du MoE Qwen3.6-35B-A3B"),
    (r"^35b[\s._-]", "qwen3-6-35b",
     "« 35b » seul (case-35b_n4.log, case-35b_split65.log) : le seul 35B du parc"),
    (r"probe[\s._-]?k2q4", "k2-horizon",
     "sonde « k2q4 » = K2 Horizon en Q4 (dossier 224-2026-09-30-bonsai2-k2)"),
)

# Libelles de la table MODELES, par cle (pour rattacher une famille connue).
LIBELLES = {cle: libelle for _, cle, libelle in MODELES}


# --------------------------------------------------------------- utilitaires
def _strip_accents(text):
    if metrics is not None and hasattr(metrics, "strip_accents"):
        return metrics.strip_accents(text)
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


def _slug(text):
    if metrics is not None and hasattr(metrics, "slug"):
        return metrics.slug(text)
    text = _strip_accents("" if text is None else str(text)).lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def _esc(value):
    return html.escape("" if value is None else str(value))


def _num(value, dec=1, unit=""):
    """Chiffre formate, ou tiret cadratin si absent (jamais de valeur inventee)."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "oui" if value else "non"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{_esc(value)}{(' ' + unit) if unit else ''}"
    texte = f"{number:.{dec}f}".rstrip("0").rstrip(".") if dec else f"{number:.0f}"
    return f"{texte}{(' ' + unit) if unit else ''}"


def _date_short(raw):
    """'2026-10-15T20:10:56' -> '15/10 20:10' (sans dependre d'une locale)."""
    texte = "" if raw is None else str(raw)
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})[T ]?(\d{2}):(\d{2})", texte)
    if m:
        return f"{m.group(3)}/{m.group(2)} {m.group(4)}:{m.group(5)}"
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", texte)
    if m:
        return f"{m.group(3)}/{m.group(2)}"
    return texte or "—"


# --------------------------------------------------------------- familles
def _candidats(entry):
    """Chaines ou chercher un nom de modele, de la plus fiable a la plus floue."""
    variant = entry.get("variant") or {}
    sortie = []
    for valeur in (variant.get("model"), entry.get("model"), entry.get("model_slug"),
                   variant.get("model_slug"), entry.get("label"), entry.get("source")):
        if isinstance(valeur, str) and valeur.strip():
            sortie.append(valeur)
    return sortie


def rattachement_de(entry):
    """(cle, justification) quand une regle RATTACHEMENTS s'applique, sinon (None, None).

    Sert a sortir du bloc « modele non identifie » les artefacts dont le nom
    d'execution ne dit pas le modele mais qui designent un seul LLM du parc.
    """
    if not isinstance(entry, dict):
        return None, None
    for texte in _candidats(entry):
        propre = MACHINES.sub("", texte)
        for motif, cle, justification in RATTACHEMENTS:
            if re.search(motif, propre, re.I):
                return cle, justification
    return None, None


def famille_de(entry):
    """(cle, libelle) de la famille LLM d'une entree. Jamais d'exception.

    Modele reconnu par la table MODELES ; sinon par une regle de rattachement
    explicite (RATTACHEMENTS) ; sinon la famille est deduite du nom nettoye,
    prefixee `autre-` pour qu'un modele inconnu ne soit jamais confondu avec un
    modele de la table.
    """
    if not isinstance(entry, dict):
        return "autre-inconnu", "Modèle non identifié"
    for texte in _candidats(entry):
        propre = MACHINES.sub("", texte)
        for motif, cle, libelle in MODELES:
            if re.search(motif, propre, re.I):
                return cle, libelle
    cle_regle, _justification = rattachement_de(entry)
    if cle_regle:
        return cle_regle, LIBELLES.get(cle_regle, cle_regle)
    # Aucun motif connu : on deduit du nom nettoye (quantification / ctx retires).
    brut = None
    variant = entry.get("variant") or {}
    for valeur in (variant.get("model"), entry.get("model"), entry.get("model_slug"),
                   entry.get("label")):
        if isinstance(valeur, str) and valeur.strip():
            brut = valeur
            break
    if not brut:
        return "autre-inconnu", "Modèle non identifié"
    propre = MACHINES.sub("", brut).strip()
    noyau = re.sub(r"\b(q|iq)\d+[_\-]?(k|x[sml]|0|1)?\b", " ", propre, flags=re.I)
    noyau = re.sub(r"\b(ud|gguf|f16|bf16|ptq\d*|m64|think|nothink|ctx)\b", " ", noyau, flags=re.I)
    noyau = re.sub(r"\s+", " ", noyau).strip(" -_·,") or propre
    cle = "autre-" + (_slug(noyau)[:40] or "inconnu")
    return cle, noyau


def build_familles(entries):
    """[{cle,label,entries,variantes,tests,premiere,derniere,identifie,regles}].

    Tri : les LLM identifiees d'abord (par volume), puis les mesures sans modele
    identifie — pour qu'elles tombent en bas du catalogue, sans etre melangees
    aux vrais modeles.
    """
    par_cle = {}
    for entry in entries or []:
        cle, libelle = famille_de(entry)
        bloc = par_cle.setdefault(cle, {"cle": cle, "label": libelle, "entries": []})
        bloc["entries"].append(entry)
    sortie = []
    for cle, bloc in par_cle.items():
        variantes, tests = [], {}
        dates = []
        regles = []
        for entry in bloc["entries"]:
            variant = entry.get("variant") or {}
            vk = variant.get("key") or "standard"
            if vk not in [v["key"] for v in variantes]:
                variantes.append({"key": vk, "label": variant.get("label") or vk,
                                  "model_slug": variant.get("model_slug") or ""})
            groupe = _test_de(entry)
            tests.setdefault(groupe[0], {"cle": groupe[0], "label": groupe[1], "entries": []})
            tests[groupe[0]]["entries"].append(entry)
            date = entry.get("date")
            if date:
                dates.append(str(date))
            _cle_regle, justification = rattachement_de(entry)
            if justification and justification not in regles:
                regles.append(justification)
        variantes.sort(key=lambda v: v["label"])
        sortie.append({
            "cle": cle,
            "label": bloc["label"],
            "entries": bloc["entries"],
            "tests": list(tests.values()),
            "variantes": variantes,
            "premiere": min(dates) if dates else None,
            "derniere": max(dates) if dates else None,
            "identifie": not str(cle).startswith("autre-"),
            "regles": regles,
        })
    sortie.sort(key=lambda f: (not f["identifie"], -len(f["entries"]), f["label"].lower()))
    return sortie


def catalogue_llm(entries):
    """Catalogue des LLM pour l'axe `?llm=` (forme proche de `metrics.series_catalogue`).

    -> [{"key","label","points","partiels","tests","mesures","last_date",
         "identifie","regles"}], LLM identifiees d'abord.
    """
    items = []
    for famille in build_familles(entries or []):
        points = partiels = 0
        for entry in famille["entries"]:
            valeurs = entry.get("metric_values") or {}
            if not any(v is not None for v in valeurs.values()):
                continue                       # entree en erreur : rien a tracer
            if entry.get("partiel"):
                partiels += 1
            else:
                points += 1
        items.append({
            "key": famille["cle"],
            "label": famille["label"],
            "points": points,
            "partiels": partiels,
            "tests": len(famille["tests"]),
            "mesures": len(famille["entries"]),
            "last_date": famille["derniere"] or "",
            "identifie": famille["identifie"],
            "regles": famille["regles"],
        })
    return items


def _test_de(entry):
    """(cle, libelle) du TEST auquel appartient une entree (via metrics)."""
    if metrics is not None:
        try:
            return metrics.group_of(entry)
        except Exception:
            pass
    return str(entry.get("kind") or "autre"), str(entry.get("kind") or "Autres")


def _ordre_kind(entry):
    kind = (entry.get("kind") or "").lower()
    if metrics is not None:
        try:
            kind = metrics.canon_kind(kind) or kind
            return metrics.KIND_RANK.get(kind, 9)
        except Exception:
            pass
    return 9


def _metriques_test(entries, limite=2):
    """Les metriques d'un test, dans l'ordre du registre, bornees a `limite`."""
    if metrics is None:
        return []
    vus, specs = set(), []
    for entry in entries:
        kind = entry.get("kind")
        for spec in metrics.metric_specs(kind):
            if spec["key"] in vus:
                continue
            if any((e.get("metric_values") or {}).get(spec["key"]) is not None
                   for e in entries):
                vus.add(spec["key"])
                specs.append(spec)
    return specs[:limite]


def _meilleure(entries, cle_metrique, better="high"):
    """Meilleure mesure d'une metrique pour un LLM : {valeur,date,variante,entree,n}."""
    mesures = []
    for entry in entries:
        valeur = (entry.get("metric_values") or {}).get(cle_metrique)
        if valeur is None:
            continue
        mesures.append((valeur, str(entry.get("date") or ""), entry))
    if not mesures:
        return None
    try:
        mesures.sort(key=lambda m: float(m[0]), reverse=(better != "low"))
    except (TypeError, ValueError):
        mesures.sort(key=lambda m: str(m[0]))
    valeur, date, entry = mesures[0]
    return {"valeur": valeur, "date": date,
            "variant": (entry.get("variant") or {}).get("label") or "standard",
            "n": len(mesures), "entry": entry}


# --------------------------------------------------------------- selection
def parse_selection(query, connues):
    """(selection, inconnues, tronquees) : cles demandees -> cles connues.

    Alias acceptes : llm, llms, modele, modeles, model. Une cle inconnue est
    SIGNALEE mais jamais appliquee (meme regle que /graph).
    """
    demandees, vues = [], set()
    for nom, valeurs in (query or {}).items():
        if nom.lower() not in ("llm", "llms", "modele", "modeles", "model"):
            continue
        for valeur in valeurs if isinstance(valeurs, list) else [valeurs]:
            for morceau in re.split(r"[,;\s]+", str(valeur)):
                morceau = morceau.strip()
                if morceau and morceau not in vues:
                    vues.add(morceau)
                    demandees.append(morceau)
    connues_set = {c["cle"] for c in connues}
    retenues = [c for c in demandees if c in connues_set]
    inconnues = [c for c in demandees if c not in connues_set]
    tronquees = retenues[MAX_SELECTION:]
    return retenues[:MAX_SELECTION], inconnues, tronquees


def liens_graph(famille, max_series=6):
    """Clés de variantes d'une famille pour /graph (max 6, alias `sel`)."""
    cles = [v["key"] for v in famille["variantes"]][:max_series]
    return "/graph?" + "&".join("sel=" + _urlencode(c) for c in cles) if cles else None


def _urlencode(texte):
    from urllib.parse import quote_plus
    return quote_plus(str(texte))


# --------------------------------------------------------------- rendu
EXTRA_CSS = """
.llm-form{display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap}
select[name=llm]{min-width:340px;height:260px;background:#0b0e14;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:6px;font:inherit}
select[name=llm] option{padding:4px 6px}
.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:2px 9px;font-size:11.5px;color:var(--dim);margin:0 4px 4px 0}
.pill b{color:var(--fg);font-weight:600}
td.best{color:var(--ok);font-weight:600}
td.mes{white-space:normal;max-width:300px}
.sub{color:var(--dim);font-size:11px}
h3{font-size:13.5px;margin:14px 0 4px}
.llm-carte{border:1px solid var(--line);border-radius:10px;padding:12px;margin:12px 0;background:#12161e}
tr.sep td{background:#171c26;color:var(--fg);white-space:normal}
.rule{background:#12161e;border:1px solid var(--line);border-left:3px solid var(--acc);border-radius:8px;padding:10px 12px;margin:10px 0;font-size:12.5px;color:var(--dim)}
details>summary{cursor:pointer;color:var(--acc);font-size:12.5px}
"""


def _nav(gen=None):
    retour = f'<span class="dim">index {_esc(gen)}</span>' if gen else ""
    return ('<a class="navlink" href="/">&#8592; Prompt Duel</a>\n'
            '  <a class="navlink" href="/benchmarks">Benchmarks</a>\n'
            '  <a class="navlink" href="/llms" aria-current="page">Tests par LLM</a>\n'
            '  <a class="navlink" href="/graph">Graphique</a>\n'
            '  <a class="navlink" href="/compare">Comparaison de runs</a>\n  ' + retour)


def _page(titre, corps, gen=None):
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(titre)} — Prompt Duel</title>
<style>{BASE_CSS}{EXTRA_CSS}</style></head><body>
<header>
  <h1>Tests par LLM <span class="dim">&middot; tous les tests d'un modèle, dates comprises</span></h1>
  <div class="spacer" style="flex:1"></div>
  {_nav(gen)}
</header>
<main>{corps}</main>
<footer>Prompt Duel {APP_VERSION} &middot; page <code>/llms</code> &middot; un LLM = un nom de modèle
(la quantification, le contexte et le mode de raisonnement restent visibles comme
<em>variantes</em> sous le même LLM) &middot; valeurs = <strong>mesurées</strong>, jamais recopiées
d'une fiche modèle &middot; un tiret cadratin signifie « pas de mesure ».</footer>
</body></html>""".encode("utf-8")


def _regle_html():
    return ('<div class="rule"><b>1 LLM par famille.</b> Deux tirs du même modèle à deux dates, '
            'ou avec deux quantifications, comptent pour le même LLM : ils sont regroupés ici et '
            'comparés aux autres LLM. Choisis 1 à 6 LLM, puis « Afficher ». Le tableau donne la '
            '<b>meilleure</b> valeur mesurée par LLM (avec sa date et le nombre de mesures) ; le '
            'détail chronologique de chaque LLM suit en dessous.</div>')


def render_catalogue(familles, inconnues=None):
    """Etape 1 : le catalogue des LLM + le formulaire de sélection."""
    if not familles:
        return ('<section class="card"><h2>Aucun LLM dans l\'index</h2>'
                '<p class="dim">L\'index unifié est vide ou introuvable. Régénère-le : '
                '<code>python3 bench_index.py</code>.</p></section>')
    identifiees = [f for f in familles if f.get("identifie")]
    sans_modele = [f for f in familles if not f.get("identifie")]
    options = []
    if identifiees:
        options.append(f'<optgroup label="LLM reconnus ({len(identifiees)})">'
                       + "".join(_option(f) for f in identifiees) + '</optgroup>')
    if sans_modele:
        n = sum(len(f["entries"]) for f in sans_modele)
        options.append(f'<optgroup label="Mesures sans modèle identifié '
                       f'({n} mesure(s), {len(sans_modele)} nom(s) d\'artefact)">'
                       + "".join(_option(f) for f in sans_modele) + '</optgroup>')
    lignes = [_ligne_catalogue(f) for f in identifiees]
    lignes_sans = [_ligne_catalogue(f) for f in sans_modele]
    bloc_sans = ""
    if lignes_sans:
        bloc_sans = (
            f'<tr class="sep"><td colspan="6"><b>Mesures sans modèle identifié</b> '
            f'<span class="dim">· {sum(len(f["entries"]) for f in sans_modele)} mesure(s) — '
            'le nom de fichier de la mesure ne dit pas le modèle : elles restent visibles '
            'telles quelles, jamais fusionnées avec un LLM reconnu</span></td></tr>'
            + "".join(lignes_sans))
    avert = ""
    if inconnues:
        avert = (f'<div class="rule warn">Clé(s) inconnue(s), ignorée(s) : '
                 f'{_esc(", ".join(inconnues))}</div>')
    return f"""
{avert}
{_regle_html()}
<section class="card">
  <h2>1. Choisis les LLM à comparer <span class="dim">(1 à {MAX_SELECTION}, Ctrl pour en prendre plusieurs)</span></h2>
  <form class="llm-form" method="get" action="/llms">
    <select name="llm" multiple size="14">{''.join(options)}</select>
    <div>
      <button type="submit">Afficher les tests</button>
      <p class="dim" style="margin-top:8px">Une sélection = un tableau « test × LLM », puis
      le détail chronologique de chaque LLM.</p>
    </div>
  </form>
</section>
<section class="card">
  <h2>2. Les LLM de l'index <span class="dim">({len(identifiees)} reconnu(s) ·
  {len(sans_modele)} sans modèle identifié)</span></h2>
  <div class="scroll"><table>
    <thead><tr><th>LLM</th><th class="n">mesures</th><th class="n">tests</th><th class="n">variantes</th>
    <th>période</th><th>variantes (quant, ctx, mode…)</th></tr></thead>
    <tbody>{''.join(lignes)}{bloc_sans}</tbody>
  </table></div>
  <p class="dim">Un LLM = un <b>nom de modèle</b>, indépendant de la quantification, du contexte et
  du mode de raisonnement : ces différences restent visibles comme <em>variantes</em> sous le même
  LLM. Les artefacts dont le nom ne porte pas de modèle (« dry-run », « smoke-n1 », sonde de
  vitesse…) sont regroupés en bas, sans être rattachés à un modèle au hasard.</p>
</section>"""


def _option(famille):
    n_tests = len(famille["tests"])
    suffixe = "" if famille.get("identifie") else " · modèle non identifié"
    return (f'<option value="{_esc(famille["cle"])}">{_esc(famille["label"])} '
            f'— {n_tests} test(s) · {len(famille["entries"])} mesure(s){suffixe}</option>')


def _ligne_catalogue(famille):
    """Une ligne du tableau du catalogue (LLM ou artefact non identifié)."""
    n_tests = len(famille["tests"])
    regles = ""
    if famille.get("regles"):
        regles = ('<div class="sub">rattaché au modèle par règle explicite : '
                  + _esc(" · ".join(famille["regles"])) + '</div>')
    return ("<tr>"
            f'<td><b>{_esc(famille["label"])}</b><div class="sub">{_esc(famille["cle"])}</div>'
            f'{regles}</td>'
            f'<td class="n">{len(famille["entries"])}</td>'
            f'<td class="n">{n_tests}</td>'
            f'<td class="n">{len(famille["variantes"])}</td>'
            f'<td>{_esc(_date_short(famille["premiere"]))} → {_esc(_date_short(famille["derniere"]))}</td>'
            f'<td class="mes">{_esc(", ".join(v["label"] for v in famille["variantes"])[:180])}</td>'
            "</tr>")


def render_selection(familles, selection, inconnues=None, tronquees=None):
    """Etape 2 : tableau test x LLM + detail chronologique par LLM."""
    choisies = [f for f in familles if f["cle"] in selection]
    if not choisies:
        return render_catalogue(familles, inconnues)
    bandeau = ['<section class="card">',
               f'<h2>{len(choisies)} LLM comparé(s)</h2>',
               '<div>']
    for famille in choisies:
        bandeau.append(f'<span class="pill"><b>{_esc(famille["label"])}</b> · '
                       f'{len(famille["entries"])} mesure(s) · {len(famille["tests"])} test(s)</span>')
    bandeau.append('</div>')
    lien = "/llms?edit=1"
    bandeau.append(f'<p class="dim"><a href="{lien}">← modifier la sélection</a> · '
                   '<a href="/llms">catalogue</a></p>')
    if inconnues:
        bandeau.append(f'<div class="rule warn">Clé(s) inconnue(s), ignorée(s) : '
                       f'{_esc(", ".join(inconnues))}</div>')
    if tronquees:
        bandeau.append(f'<div class="rule warn">Sélection limitée à {MAX_SELECTION} LLM : '
                       f'ignorés — {_esc(", ".join(tronquees))}</div>')
    bandeau.append('</section>')

    # --- tableau test x LLM -------------------------------------------------
    lignes_test = []
    for cle_test, label_test, entrees_test in _tests_ordonnes(choisies):
        specs = _metriques_test(entrees_test) or []
        if not specs:
            continue
        for rang, spec in enumerate(specs):
            cellules = []
            meilleures = []
            for famille in choisies:
                du_llm = [e for e in famille["entries"] if _test_de(e)[0] == cle_test]
                meilleure = _meilleure(du_llm, spec["key"], spec.get("better", "high"))
                meilleures.append(meilleure)
                if meilleure is None:
                    cellules.append('<td class="n">—</td>')
                    continue
                titre = _esc(meilleure["variant"]) + " · " + _esc(meilleure["date"])
                complement = f'<div class="sub">+{meilleure["n"] - 1} mesure(s)</div>' if meilleure["n"] > 1 else ""
                cellules.append(
                    f'<td class="n" title="{titre}">{_num(meilleure["valeur"], 1, spec["unit"])}'
                    f'<div class="sub">{_esc(_date_short(meilleure["date"]))}</div>{complement}</td>')
            _marquer_meilleure(cellules, meilleures, spec.get("better", "high"))
            premiere = (f'<td rowspan="{len(specs)}"><b>{_esc(label_test)}</b>'
                        f'<div class="sub">{_esc(cle_test)}</div></td>') if rang == 0 else ""
            lignes_test.append(
                f'<tr>{premiere}<td>{_esc(spec["label"])}'
                f'{(" <span class=dim>(" + _esc(spec["unit"]) + ")</span>") if spec["unit"] else ""}</td>'
                + "".join(cellules) + "</tr>")
    entetes = "".join(f'<th>{_esc(f["label"])}</th>' for f in choisies)
    tableau = f"""
<section class="card">
  <h2>Tableau comparatif <span class="dim">meilleure valeur mesurée par LLM · 1 ligne = 1 métrique</span></h2>
  <div class="scroll"><table>
    <thead><tr><th>Test</th><th>Métrique</th>{entetes}</tr></thead>
    <tbody>{''.join(lignes_test)}</tbody>
  </table></div>
  <p class="dim">Vert = meilleure valeur de la ligne entre les LLM comparés. La date sous le
  chiffre est celle de la mesure retenue ; « +N mesure(s) » indique les autres mesures du même
  LLM pour ce test (toutes listées au détail ci-dessous).</p>
</section>"""

    # --- detail chronologique par LLM --------------------------------------
    details = []
    for famille in choisies:
        blocs = []
        for cle_test, label_test, entrees_test in _tests_ordonnes([famille]):
            specs = []
            if metrics is not None and entrees_test:
                try:
                    specs = metrics.metric_specs(entrees_test[0].get("kind")) or []
                except Exception:
                    specs = []
            lignes = []
            for entry in sorted(entrees_test, key=lambda e: str(e.get("date") or ""), reverse=True):
                valeurs = []
                for spec in specs:
                    valeur = (entry.get("metric_values") or {}).get(spec["key"])
                    if valeur is not None:
                        valeurs.append(f'{_esc(spec["label"])} {_num(valeur, 1, spec["unit"])}')
                statut = str(entry.get("status") or "")
                classe = "" if statut in ("", "ok", "finished") else "err"
                lignes.append(
                    "<tr>"
                    f'<td>{_esc(_date_short(entry.get("date")))}</td>'
                    f'<td class="variant">{_esc((entry.get("variant") or {}).get("label") or "standard")}</td>'
                    f'<td class="mes">{_esc(" · ".join(valeurs)) or "—"}</td>'
                    f'<td class="{classe}">{_esc(statut or "—")}</td>'
                    f'<td class="dim">{_esc(str(entry.get("source") or "")[-70:])}</td>'
                    "</tr>")
            blocs.append(f'<h3>{_esc(label_test)}</h3><div class="scroll"><table>'
                         '<thead><tr><th>date</th><th>variante</th><th>mesures</th><th>statut</th>'
                         '<th>source</th></tr></thead>'
                         f'<tbody>{"".join(lignes)}</tbody></table></div>')
        lien_graph = liens_graph(famille)
        lien = (f'<a href="{_esc(lien_graph)}">voir les courbes de ce LLM sur /graph</a>'
                if lien_graph else "")
        if famille.get("regles"):
            note = ('<div class="rule">Rattaché au modèle par règle explicite : '
                    + _esc(" · ".join(famille["regles"])) + '</div>')
        elif not famille.get("identifie"):
            note = ('<div class="rule warn">Modèle non identifié : le nom de l\'artefact ne dit pas '
                    'quel LLM a tourné. Ces mesures restent visibles telles quelles et ne sont '
                    'fusionnées avec aucune famille.</div>')
        else:
            note = ""
        details.append(f"""
<div class="llm-carte">
  <h2>{_esc(famille["label"])} <span class="dim">· {len(famille["entries"])} mesure(s) ·
  {len(famille["tests"])} test(s) · {_esc(_date_short(famille["premiere"]))} →
  {_esc(_date_short(famille["derniere"]))}</span></h2>
  <p class="dim">{lien}</p>
  {note}
  <details><summary>variantes regroupées sous ce LLM ({len(famille["variantes"])})</summary>
    <div>{''.join(f'<span class="pill">{_esc(v["label"])}</span>' for v in famille["variantes"])}</div>
  </details>
  {''.join(blocs)}
</div>""")
    return f"""{''.join(bandeau)}
{tableau}
<section class="card"><h2>Détail chronologique par LLM</h2>
<p class="dim">Chaque mesure enregistrée, du plus récent au plus ancien, avec la variante
réellement servie et la source. Rien n'est masqué ici : un statut autre que
<code>ok</code> est affiché en rouge.</p></section>
{''.join(details)}"""


def _marquer_meilleure(cellules, meilleures, better="high"):
    """Passe en vert la (les) meilleure(s) cellule(s) d'une ligne.

    `better` vient du registre des metriques (`metrics.metric_spec(key)["better"]`) :
    pour une duree ou une VRAM, « mieux » = plus PETIT. Sans ce sens, la page
    mettrait en vert le pire chiffre de la ligne.
    """
    valeurs = [(i, float(m["valeur"])) for i, m in enumerate(meilleures)
               if m is not None and isinstance(m["valeur"], (int, float))
               and not isinstance(m["valeur"], bool)]
    if len(valeurs) < 2:
        return
    cible = min(v for _, v in valeurs) if str(better).lower() == "low" else max(v for _, v in valeurs)
    for i, valeur in valeurs:
        if abs(valeur - cible) < 1e-9:
            cellules[i] = cellules[i].replace('<td class="n"', '<td class="n best"', 1)


def _tests_ordonnes(familles):
    """[(cle, label, entrees)] : les tests de ces LLM, dans l'ordre du registre."""
    vus = {}
    for famille in familles:
        for entry in famille["entries"]:
            cle, label = _test_de(entry)
            bloc = vus.setdefault(cle, {"label": label, "entries": [], "rang": _ordre_kind(entry)})
            bloc["entries"].append(entry)
            if entry.get("kind") == "vitesse":
                bloc["rang"] = 0
    sortie = sorted(vus.items(), key=lambda kv: (kv[1]["rang"], kv[1]["label"].lower()))
    return [(cle, bloc["label"], bloc["entries"]) for cle, bloc in sortie]


# --------------------------------------------------------------- API
def render_llms_page(index, state=None, query=None):
    """Page /llms. `index` : dict de bench_index.json ; `state` : etat de visibilite."""
    entetes = {}
    for nom, valeurs in (query or {}).items():
        entetes[nom.lower()] = valeurs
    entries = index.get("entries") if isinstance(index, dict) else None
    entries = entries if isinstance(entries, list) else []
    if metrics is not None:
        entries = metrics.normalize_entries(entries)
    if visibility is not None and state is not None:
        try:
            entries = visibility.apply(state, entries)
        except Exception:
            pass
    familles = build_familles(entries)
    selection, inconnues, tronquees = parse_selection(entetes, familles)
    gen = index.get("generated_at") if isinstance(index, dict) else None
    if not selection:
        return _page("Tests par LLM", render_catalogue(familles, inconnues), gen)
    return _page("Tests par LLM", render_selection(familles, selection, inconnues, tronquees), gen)


def main(argv=None):
    """Rendu en ligne de commande (debug / test sans serveur)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    import os
    index_path = None
    for i, arg in enumerate(argv):
        if arg == "--index" and i + 1 < len(argv):
            index_path = argv[i + 1]
    if index_path is None:
        for candidat in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_index.json"),
                         "/home/gab/NAS/AgentsMirror/vaults/personnel/documents/llm-benchmarks/index.json"):
            if os.path.exists(candidat):
                index_path = candidat
                break
    if index_path is None:
        print("index introuvable", file=sys.stderr)
        return 2
    with open(index_path, encoding="utf-8") as fh:
        index = json.load(fh)
    query = {}
    for arg in argv:
        if "=" in arg:
            cle, valeur = arg.split("=", 1)
            query.setdefault(cle, []).append(valeur)
    sys.stdout.write(render_llms_page(index, None, query).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
