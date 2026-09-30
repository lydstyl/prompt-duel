#!/usr/bin/env python3
"""Tests unitaires de l'etape E2 « interface » (unittest, stdlib uniquement).

Trois volets :
  * `page_graph.render_graph_page` : rendu SANS exception sur un index vide, sur
    un index d'une seule entree, et sur l'**index reel du vault en LECTURE SEULE**
    (le fichier ne doit pas etre touche et aucune ecriture ne doit apparaitre) ;
  * les filtres (par test et par variante), l'affichage des masques, la liste des
    resultats masques / supprimes et les boutons « Reafficher » / « Restaurer » ;
  * `benchmarks.py` : colonne variante, bouton « Masquer » par ligne, filtre
    « afficher les resultats masques », sans casser la structure HTML existante.

Aucune ecriture : ni dans le vault, ni dans le depot. Tout passe par des index
de fixtures ou par l'index reel lu (jamais modifie).
"""
import json
import os
import re
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmarks   # noqa: E402
import metrics      # noqa: E402
import page_graph   # noqa: E402
import visibility   # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
E2E_FIXTURES = ROOT / "e2e" / "fixtures"
# Index REEL du vault (lecture seule). Absent sur une machine sans vault : les
# tests correspondants sont alors ignores, jamais en echec.
VAULT_INDEX = Path("/home/lydstyl/agentsmirror-mnt/vaults/personnel/"
                   "documents/llm-benchmarks/index.json")

ETAT_VIDE = {"hidden": [], "deleted": [], "filters": {}}


# --------------------------------------------------------------- outils de test
def charger(nom, dossier=FIXTURES):
    with open(dossier / nom, encoding="utf-8") as fh:
        return json.load(fh)


def svgs(page):
    """Fragments <svg> de la page, valides en XML (leve si mal formes)."""
    fragments = re.findall(r"<svg\b.*?</svg>", page, re.S)
    for fragment in fragments:
        ET.fromstring(fragment)
    return fragments


def lignes_resultats(page):
    """Identifiants des lignes du tableau #resultats, dans l'ordre."""
    bloc = re.search(r'<table id="resultats">.*?</table>', page, re.S)
    return re.findall(r'<tr data-entry="([^"]*)"', bloc.group(0)) if bloc else []


def bloc_masques(page):
    return re.search(r'<section class="card" id="masques">.*?</section>', page, re.S)


def entree_duel(n=0, **kwargs):
    """Entree synthetique de famille `duel` (une execution d'un test)."""
    entry = {
        "kind": "duel",
        "date": f"2026-09-2{1 + n}T1{n}:00:00",
        "run_id": f"2026-09-2{1 + n}_100{n}__test-model__ctx8192__t0.2",
        "model": "Test-Model",
        "model_slug": "test-model",
        "ctx": 8192,
        "temperature": 0.2,
        "max_tokens": 8192,
        "thinking": False,
        "prompts_total": 3,
        "prompts_ok": 2,
        "duration_s": 12.5 + n,
        "tokens_out": 500,
        "tok_s": 120.0 + n,
        "source": f"runs/2026-09-2{1 + n}_100{n}__test-model/run.json",
    }
    entry.update(kwargs)
    return entry


def index_de(entries, **kwargs):
    data = {"generated_at": "2026-09-30T12:00:00", "counts": {}, "entries": list(entries),
            "vault_notes": [], "_source": "<fixture>"}
    data.update(kwargs)
    return data


def index_deux_reglages():
    """Index a DEUX reglages distincts (contexte 8k / 16k) et leurs cles.

    Deux reglages = le minimum pour que `/graph` passe a l'etape 2.
    """
    entries = [entree_duel(0), entree_duel(1, ctx=16384)]
    cles = [item["key"] for item in metrics.series_catalogue(entries)]
    assert len(cles) == 2, cles
    return index_de(entries), cles


def index_plusieurs_reglages(n=8):
    """Index a `n` reglages distincts (contextes croissants) et leurs cles."""
    entries = [entree_duel(i, ctx=8192 * (2 ** i)) for i in range(n)]
    cles = [item["key"] for item in metrics.series_catalogue(entries)]
    return index_de(entries), cles


def series_tracees(page):
    """Cles de reglage presentes dans les graphiques (barres + legendes)."""
    return set(re.findall(r'data-serie="([^"]+)"', page))


# =========================================================================
# 1) Rendu sans exception : index vide, index d'une entree
# =========================================================================
class TestRenduSansDonnees(unittest.TestCase):
    """La page doit toujours sortir du HTML valide, meme sans une seule entree."""

    def test_index_vide(self):
        page = page_graph.render_graph_page({}, ETAT_VIDE, {})
        self.assertIsInstance(page, str)
        self.assertTrue(page.startswith("<!doctype html>"))
        self.assertIn("<main>", page)
        self.assertIn("</html>", page)
        self.assertIn("Aucun index", page)          # EMPTY_HINT aiguille l'utilisateur
        self.assertEqual(svgs(page), [])            # rien a tracer : aucun SVG casse

    def test_index_vide_liste_de_plusieurs_types(self):
        for index in (None, [], {"entries": []}, {"error": "index illisible"},
                      {"entries": None}, "pas un dict"):
            with self.subTest(index=repr(index)[:40]):
                page = page_graph.render_graph_page(index, ETAT_VIDE, {})
                self.assertIn("</html>", page)

    def test_index_avec_erreur_affiche_le_message(self):
        page = page_graph.render_graph_page({"error": "fichier absent"}, ETAT_VIDE, {})
        self.assertIn("fichier absent", page)

    def test_query_absente_ou_vide(self):
        for query in (None, {}, "", "?", []):
            with self.subTest(query=repr(query)):
                page = page_graph.render_graph_page({}, ETAT_VIDE, query)
                self.assertIn("</html>", page)

    def test_state_absent_ou_abime(self):
        index = index_de([entree_duel()])
        for state in (None, [], "pas un dict", {"hidden": "pas une liste"}):
            with self.subTest(state=repr(state)[:30]):
                page = page_graph.render_graph_page(index, state, {})
                self.assertIn("</html>", page)

    def test_jamais_de_nan_ni_infinity_dans_les_svg(self):
        entries = [entree_duel(0, tok_s=float("nan")),
                   entree_duel(1, ctx=16384, duration_s=None)]
        cles = [item["key"] for item in metrics.series_catalogue(entries)]
        page = page_graph.render_graph_page(index_de(entries), ETAT_VIDE, {"sel": cles})
        self.assertTrue(svgs(page))                    # des graphiques, donc une vraie epreuve
        for fragment in svgs(page):
            self.assertNotIn("nan", fragment.lower())
            self.assertNotIn("infinity", fragment.lower())


class TestRenduUneEntree(unittest.TestCase):
    """Deux reglages coches : graphiques chiffres + tableau des resultats."""

    def setUp(self):
        self.index, self.cles = index_deux_reglages()
        self.page = page_graph.render_graph_page(self.index, ETAT_VIDE, {"sel": self.cles})

    def test_svg_presents_et_valides(self):
        fragments = svgs(self.page)
        self.assertGreaterEqual(len(fragments), 2)      # metrique + durees
        for fragment in fragments:
            self.assertIn("<title>", fragment)          # info-bulle native

    def test_entete_onglets_et_titre(self):
        self.assertIn('<a class="navlink" href="/benchmarks">', self.page)
        self.assertIn('<a class="navlink" href="/graph" aria-current="page">', self.page)
        self.assertIn("Graphique des tests", self.page)

    def test_tableau_de_resultats_et_bouton_masquer(self):
        self.assertEqual(len(lignes_resultats(self.page)), 2)   # les 2 reglages coches
        self.assertIn('data-vis="hide"', self.page)
        self.assertIn(">Masquer<", self.page)

    def test_legende_sans_javascript(self):
        self.assertIn('class="chart-legend"', self.page)
        self.assertIn('class="chart-toggle"', self.page)

    def test_legerement_plusieurs_entrees_donnent_plusieurs_lignes(self):
        """A l'etape 1 (aucune selection), le tableau liste toujours tout."""
        page = page_graph.render_graph_page(
            index_de([entree_duel(0), entree_duel(1), entree_duel(2)]), ETAT_VIDE, {})
        self.assertEqual(len(lignes_resultats(page)), 3)


# =========================================================================
# 1bis) Le parcours en deux etapes : choisir 2 a 6 reglages, puis comparer
# =========================================================================
class TestParcoursEnDeuxEtapes(unittest.TestCase):
    """Etape 1 = le catalogue coche ; etape 2 = seulement les reglages coches."""

    def setUp(self):
        self.index, self.cles = index_plusieurs_reglages(8)

    def page(self, query):
        return page_graph.render_graph_page(self.index, ETAT_VIDE, query)

    def vue(self, query):
        return page_graph.build_view(self.index, ETAT_VIDE, query)

    # -- etape 1 : le choix --------------------------------------------------
    def test_etape1_liste_les_reglages_cocables(self):
        page = self.page({})
        self.assertEqual(page.count('name="sel"'), len(self.cles))
        for cle in self.cles:
            self.assertIn(f'value="{cle}"', page)         # une case par reglage
        self.assertIn("point", page)                      # nombre de points disponibles
        self.assertIn("dernier le", page)                 # date du dernier resultat
        self.assertIn('action="/graph"', page)            # GET : aucun JavaScript

    def test_etape1_annonce_la_regle_2_6(self):
        page = self.page({})
        self.assertIn("2 au minimum", page)
        self.assertIn("6 au maximum", page)
        self.assertIn("Comparer", page)

    def test_etape1_aucun_graphique_mais_resultats_accessibles(self):
        page = self.page({})
        self.assertEqual(svgs(page), [])
        self.assertNotIn("<svg", page)
        # le reste ne disparait pas : il est replie dans un <details> natif (sans JS)
        self.assertIn('<details id="tous-resultats">', page)
        self.assertTrue(lignes_resultats(page))

    def test_etape1_un_seul_reglage_est_dit_en_clair(self):
        index = index_de([entree_duel(0)])
        page = page_graph.render_graph_page(index, ETAT_VIDE, {})
        self.assertEqual(page.count('name="sel"'), 1)
        self.assertIn("1 disponible(s)", page)
        self.assertEqual(svgs(page), [])

    # -- etape 2 : la comparaison -------------------------------------------
    def test_etape2_ne_trace_que_les_reglages_coches(self):
        cles = self.cles[:3]
        page = self.page({"sel": cles})
        self.assertEqual(series_tracees(page), set(cles))
        for cle in self.cles[3:]:
            self.assertNotIn(f'data-serie="{cle}"', page)
        self.assertIn("Comparaison de 3 r", page)

    def test_etape2_lien_modifier_la_selection(self):
        cles = self.cles[:2]
        page = self.page({"sel": cles})
        self.assertIn("modifier la sélection", page)
        self.assertIn(f'href="/graph?edit=1&amp;sel={cles[0]}&amp;sel={cles[1]}"', page)

    def test_retour_etape1_conserve_les_cases_cochees(self):
        cles = self.cles[:2]
        page = self.page({"edit": ["1"], "sel": cles})
        self.assertEqual(svgs(page), [])                  # retour au choix
        self.assertNotIn("banner err", page)              # sans message d'erreur
        for cle in cles:
            self.assertEqual(page.count(f'value="{cle}" checked'), 1)

    # -- bornes --------------------------------------------------------------
    def test_une_seule_cle_ramene_a_l_etape1(self):
        cles = self.cles[:1]
        page, vue = self.page({"sel": cles}), self.vue({"sel": cles})
        self.assertEqual(vue["etape"], 1)
        self.assertEqual(vue["selection"], cles)          # la case reste cochee
        self.assertEqual(svgs(page), [])
        self.assertIn("au moins 2", page)

    def test_plus_de_six_reglages_est_tronque(self):
        cles = self.cles[:7]
        vue = self.vue({"sel": cles})
        self.assertEqual(vue["etape"], 2)
        self.assertEqual(vue["selection"], cles[:6])
        self.assertIn(cles[6], vue["selection_warning"])
        page = self.page({"sel": cles})
        self.assertEqual(series_tracees(page), set(cles[:6]))
        self.assertIn("6 réglages au maximum", page)

    def test_cle_inconnue_ignoree_et_signalee(self):
        cles, inconnue = self.cles[:2], "reglage-qui-nexiste-pas"
        page = self.page({"sel": cles + [inconnue]})
        vue = self.vue({"sel": cles + [inconnue]})
        self.assertEqual(vue["etape"], 2)                 # les 2 vraies cles suffisent
        self.assertEqual(vue["selection_inconnues"], [inconnue])
        self.assertEqual(series_tracees(page), set(cles))
        self.assertIn("inconnu", page)
        self.assertIn(inconnue, page)

    def test_que_des_cles_inconnues_message_clair(self):
        page = self.page({"sel": ["x", "y"]})
        self.assertEqual(svgs(page), [])
        self.assertIn("banner err", page)
        self.assertIn("inconnu", page)

    # -- URLs tolerantes -----------------------------------------------------
    def test_aliases_de_selection(self):
        cles = self.cles[:2]
        for param in ("sel", "selection", "series", "serie", "llm", "reglages", "reglage"):
            with self.subTest(param=param):
                vue = self.vue({param: cles})
                self.assertEqual(vue["etape"], 2)
                self.assertEqual(vue["selection"], cles)
        for chaine in (f"?sel={cles[0]}&series={cles[1]}",
                       f"?llm={cles[0]}&reglages={cles[1]}"):
            with self.subTest(chaine=chaine):
                self.assertEqual(self.vue(chaine)["etape"], 2)

    # -- affiner / lisibilite ------------------------------------------------
    def test_filtre_de_famille_a_l_etape2_conserve_la_selection(self):
        cles = self.cles[:2]
        page = self.page({"sel": cles, "kind": ["duel"]})
        self.assertEqual(series_tracees(page), set(cles))
        self.assertIn('<section class="card" id="affiner">', page)
        self.assertEqual(page.count('<input type="hidden" name="sel"'), 2)

    def test_six_series_ont_des_couleurs_distinctes(self):
        page = self.page({"sel": self.cles[:6]})
        bloc = re.search(r'<div class="chart-legend".*?</div>', page, re.S).group(0)
        couleurs = re.findall(r"border-radius:3px;background:(#[0-9a-f]{6})", bloc)
        self.assertEqual(len(couleurs), 6)
        self.assertEqual(len(set(couleurs)), 6)           # aucune couleur reutilisee
        self.assertNotEqual(couleurs[0], couleurs[1])     # series voisines distinctes

    def test_max_charts_respecte(self):
        vue = self.vue({"sel": self.cles})
        self.assertLessEqual(len(vue["charts"]), page_graph.MAX_CHARTS)


# =========================================================================
# 2) Filtres, masquage, restauration
# =========================================================================
class TestFiltresEtMasques(unittest.TestCase):
    def setUp(self):
        self.extrait = charger("index_extract.json")     # 5 familles reelles
        self.entries = self.extrait["entries"]
        self.page = page_graph.render_graph_page(self.extrait, ETAT_VIDE, {})

    def test_toutes_les_familles_ont_un_graphique(self):
        familles = {e.get("kind") for e in self.entries}
        self.assertGreaterEqual(len(familles), 4)
        for kind in familles:
            self.assertIn(f'value="{kind}"', self.page)  # case du filtre (formulaire GET)

    def test_filtre_par_test_formulaire_get(self):
        for kind in sorted({e.get("kind") for e in self.entries}):
            with self.subTest(kind=kind):
                page = page_graph.render_graph_page(self.extrait, ETAT_VIDE, {"kind": [kind]})
                ids = [e["entry_id"] for e in metrics.normalize_entries(self.entries)
                       if metrics.canon_kind(e.get("kind")) == kind]
                self.assertEqual(sorted(lignes_resultats(page)), sorted(ids))
                self.assertIn("filtres actifs", page)

    def test_filtre_par_variante(self):
        norm = metrics.normalize_entries(self.entries)
        cle = norm[0]["variant"]["key"]
        page = page_graph.render_graph_page(self.extrait, ETAT_VIDE, {"variant": [cle]})
        attendus = [e["entry_id"] for e in norm if e["variant"]["key"] == cle]
        self.assertEqual(sorted(lignes_resultats(page)), sorted(attendus))
        self.assertLess(len(attendus), len(norm))

    def test_filtre_par_chaine_de_requete(self):
        page = page_graph.render_graph_page(self.extrait, ETAT_VIDE, "?kind=duel")
        norm = metrics.normalize_entries(self.entries)
        attendus = [e["entry_id"] for e in norm if e.get("kind") == "duel"]
        self.assertEqual(sorted(lignes_resultats(page)), sorted(attendus))

    def test_filtre_qui_ne_laisse_rien_passer(self):
        page = page_graph.render_graph_page(self.extrait, ETAT_VIDE, {"kind": ["inexistant"]})
        self.assertEqual(lignes_resultats(page), [])
        self.assertIn("aucun r", page.lower())

    def test_masque_retire_la_ligne_des_vues(self):
        norm = metrics.normalize_entries(self.entries)
        cible = norm[0]["entry_id"]
        etat = {"hidden": [cible], "deleted": [], "filters": {}}
        page = page_graph.render_graph_page(self.extrait, etat, {})
        self.assertNotIn(cible, lignes_resultats(page))
        self.assertEqual(len(lignes_resultats(page)), len(norm) - 1)

    def test_masque_reste_affiche_avec_le_parametre_masques(self):
        norm = metrics.normalize_entries(self.entries)
        cible = norm[0]["entry_id"]
        etat = {"hidden": [cible], "deleted": [], "filters": {}}
        page = page_graph.render_graph_page(self.extrait, etat, {"masques": ["1"]})
        self.assertIn(cible, lignes_resultats(page))
        self.assertEqual(len(lignes_resultats(page)), len(norm))

    def test_liste_des_masques_et_supprimes_avec_actions(self):
        norm = metrics.normalize_entries(self.entries)
        masque, supprime = norm[0]["entry_id"], norm[1]["entry_id"]
        etat = {"hidden": [masque], "deleted": [supprime], "filters": {}}
        page = page_graph.render_graph_page(self.extrait, etat, {})
        bloc = bloc_masques(page)
        self.assertIsNotNone(bloc, "la section des masques doit exister")
        bloc = bloc.group(0)
        self.assertIn(masque, bloc)
        self.assertIn(supprime, bloc)
        self.assertIn(f'data-vis="unhide" data-id="{masque}"', bloc)
        self.assertIn(f'data-vis="restore" data-id="{supprime}"', bloc)
        self.assertIn("R&eacute;afficher", bloc)
        self.assertIn("Restaurer", bloc)

    def test_pas_de_section_masques_quand_tout_est_visible(self):
        self.assertIsNone(bloc_masques(self.page))

    def test_identifiant_masque_absent_de_l_index(self):
        etat = {"hidden": ["duel:disparu"], "deleted": [], "filters": {}}
        page = page_graph.render_graph_page(self.extrait, etat, {})
        bloc = bloc_masques(page)
        self.assertIn("duel:disparu", bloc.group(0))
        self.assertIn("introuvable", bloc.group(0))

    def test_les_filtres_de_series_persistes_ne_s_appliquent_pas(self):
        """Sur /graph, les filtres viennent de l'URL (etat.filters est ignore)."""
        etat = {"hidden": [], "deleted": [],
                "filters": {"kinds": ["humaneval"], "models": [], "variants": [], "series": []}}
        page = page_graph.render_graph_page(self.extrait, etat, {})
        self.assertEqual(len(lignes_resultats(page)),
                         len(metrics.normalize_entries(self.entries)))

    def test_post_api_visibility_annonce_dans_la_page(self):
        page = page_graph.render_graph_page(self.extrait, ETAT_VIDE, {})
        self.assertIn("/api/visibility", page)
        self.assertIn("location.reload()", page)

    def test_etat_recu_non_modifie(self):
        etat = {"hidden": ["x"], "deleted": ["y"], "filters": {"kinds": ["duel"]}}
        avant = json.dumps(etat, sort_keys=True)
        page_graph.render_graph_page(self.extrait, etat, {})
        self.assertEqual(json.dumps(etat, sort_keys=True), avant)


class TestRobustesse(unittest.TestCase):
    """Entrees abimees et tentatives d'injection : la page tient, et echappe tout."""

    def test_entrees_abimees(self):
        entrees = [
            {"kind": "duel", "source": "a.json"},                       # champs manquants
            {"kind": None, "model": None, "tok_s": "pas un nombre"},
            {"kind": "inconnu", "value": [1, 2, 3]},
            "pas un dict",
            {"kind": "humaneval", "score_pct": float("inf"), "total": None},
            {"kind": "speed", "pp": 1e308, "duration_s": -5},
        ]
        page = page_graph.render_graph_page(index_de(entrees), {"hidden": [None, 42]}, {})
        self.assertIn("</html>", page)
        svgs(page)

    def test_echappement_html(self):
        piege = '<script>alert(1)</script>'
        entries = [entree_duel(model=piege, label='" onmouseover="x',
                               source=f"runs/{piege}/run.json")]
        page = page_graph.render_graph_page(index_de(entries), ETAT_VIDE, {})
        # rien n'est injecte brut : tout passe par html.escape
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page)
        self.assertNotIn('" onmouseover="x', page)
        self.assertEqual(page.count("<script>"), 1)          # un seul <script> : le notre
        for fragment in svgs(page):
            self.assertNotIn("<script", fragment)
            self.assertNotIn("javascript:", fragment)
            self.assertNotIn('"><script', fragment)

    def test_many_entrees_ne_leve_pas(self):
        entries = [entree_duel(i, ctx=8192 if i % 2 else 16384, duration_s=i)
                   for i in range(80)]
        cles = [item["key"] for item in metrics.series_catalogue(entries)]
        page = page_graph.render_graph_page(index_de(entries), ETAT_VIDE, {"sel": cles})
        self.assertGreaterEqual(len(svgs(page)), 2)
        # etape 1 avec 80 entrees : le catalogue reste lisible (une case par reglage)
        page1 = page_graph.render_graph_page(index_de(entries), ETAT_VIDE, {})
        self.assertEqual(page1.count('name="sel"'), 2)
        self.assertEqual(svgs(page1), [])


# =========================================================================
# 3) Index reel du vault — LECTURE SEULE
# =========================================================================
class TestIndexReel(unittest.TestCase):
    """L'index reel : rendu complet, et aucune ecriture (ni vault, ni depot)."""

    @classmethod
    def setUpClass(cls):
        if not VAULT_INDEX.exists():
            raise unittest.SkipTest(f"index du vault absent : {VAULT_INDEX}")
        with open(VAULT_INDEX, encoding="utf-8") as fh:
            cls.index = json.load(fh)
        cls.stat = VAULT_INDEX.stat()
        cls.listings = sorted(os.listdir(ROOT))
        cls.etat = visibility.load()
        cls.catalogue = metrics.series_catalogue(cls.index.get("entries") or [])

    def requete_etape2(self, n=6):
        """Requete d'etape 2 : les `n` reglages couvrant le plus de familles."""
        catalogue = sorted(self.catalogue,
                           key=lambda item: (-len(item.get("kinds") or []),
                                             -item["points"], item["key"]))
        return {"sel": [item["key"] for item in catalogue[:n]]}

    def test_rendu_avec_svg_valides(self):
        page = page_graph.render_graph_page(self.index, self.etat or ETAT_VIDE,
                                            self.requete_etape2())
        self.assertGreater(len(page), 20000)
        fragments = svgs(page)
        self.assertGreaterEqual(len(fragments), 5)
        for fragment in fragments:
            self.assertIn("<title>", fragment)
        self.assertIn('<table id="resultats">', page)

    def test_etape1_sans_aucun_graphique(self):
        """Vue par defaut de /graph : le choix des reglages, et rien a tracer."""
        page = page_graph.render_graph_page(self.index, self.etat or ETAT_VIDE, {})
        self.assertEqual(svgs(page), [])
        self.assertGreater(page.count('name="sel"'), 1)
        self.assertIn("Comparer des réglages", page)
        self.assertNotIn("modifier la sélection", page)   # le lien n'a de sens qu'a l'etape 2

    def test_toutes_les_entrees_sont_listees(self):
        page = page_graph.render_graph_page(self.index, ETAT_VIDE, {})
        self.assertEqual(len(lignes_resultats(page)), len(self.index["entries"]))

    def test_aucune_ecriture_dans_le_dossier_de_l_app(self):
        page_graph.render_graph_page(self.index, ETAT_VIDE, {})
        self.assertEqual(sorted(os.listdir(ROOT)), self.listings)

    def test_index_du_vault_inchange(self):
        page_graph.render_graph_page(self.index, ETAT_VIDE, {})
        self.assertEqual(VAULT_INDEX.stat().st_mtime_ns, self.stat.st_mtime_ns)
        self.assertEqual(VAULT_INDEX.stat().st_size, self.stat.st_size)

    def test_aucune_nan_dans_la_page(self):
        page = page_graph.render_graph_page(self.index, ETAT_VIDE, {})
        for fragment in svgs(page):
            self.assertNotIn("nan", fragment.lower())


# =========================================================================
# 4) benchmarks.py : variante, Masquer, filtre des masques
# =========================================================================
class TestBenchmarksVariante(unittest.TestCase):
    """La page /benchmarks garde sa structure et gagne variante + masquage."""

    def setUp(self):
        if not (E2E_FIXTURES / "bench_index.json").exists():
            self.skipTest("fixture d'index e2e absente")
        self.data = charger("bench_index.json", E2E_FIXTURES)
        self.norm = metrics.normalize_entries(self.data["entries"])

    def test_variable_affichee_quand_elle_existe(self):
        entry = dict(self.norm[0])
        entry["variant"] = {"key": "k1", "label": "MTP on · 256k · Test-Model"}
        data = index_de([entry])
        html = benchmarks.render_body(data, ETAT_VIDE)
        self.assertIn("MTP on · 256k · Test-Model", html)
        self.assertIn("<th>variante</th>", html)

    def test_pas_de_colonne_variante_si_tout_est_standard(self):
        """Sans aucun reglage identifiable, la colonne n'est pas ajoutee."""
        html = benchmarks.render_body(index_de([{"kind": "duel", "source": "runs/a/run.json"}]),
                                      ETAT_VIDE)
        self.assertNotIn("<th>variante</th>", html)
        self.assertIn(">Masquer<", html)          # les autres apports restent la
        self.assertFalse(benchmarks._avec_variante([{"kind": "vram", "gpu": "x"}]))

    def test_bouton_masquer_par_ligne(self):
        html = benchmarks.render_body(self.data, ETAT_VIDE)
        eid = self.norm[0]["entry_id"]
        self.assertIn(f'data-vis="hide" data-id="{eid}"', html)
        self.assertEqual(html.count('data-vis="hide"'), len(self.norm))

    def test_ligne_masquee_porte_la_classe_et_disparait(self):
        cible = self.norm[0]["entry_id"]
        etat = {"hidden": [cible], "deleted": [], "filters": {}}
        html = benchmarks.render_body(self.data, etat)
        self.assertIn(f'<tr data-entry="{cible}" class="res-masque">', html)
        self.assertEqual(html.count('class="res-masque"'), 1)
        self.assertIn('data-vis="unhide"', html)

    def test_filtre_afficher_les_masques(self):
        cible = self.norm[0]["entry_id"]
        etat = {"hidden": [cible], "deleted": [], "filters": {}}
        cache = benchmarks.render_body(self.data, etat, show_hidden=False)
        montre = benchmarks.render_body(self.data, etat, show_hidden=True)
        self.assertIn('class="res-masque"', cache)
        self.assertNotIn('class="res-masque"', montre)
        self.assertIn('id="voir-masques" checked', montre)
        self.assertIn('id="voir-masques">', cache)

    def test_bloc_des_masques_avec_restauration(self):
        masque, supprime = self.norm[0]["entry_id"], self.norm[1]["entry_id"]
        etat = {"hidden": [masque], "deleted": [supprime], "filters": {}}
        html = benchmarks.render_body(self.data, etat)
        bloc = re.search(r'<section class="card" id="masques">.*?</section>', html, re.S).group(0)
        self.assertIn(f'data-vis="unhide" data-id="{masque}"', bloc)
        self.assertIn(f'data-vis="restore" data-id="{supprime}"', bloc)

    def test_structure_des_tableaux_intacte(self):
        """Chaque ligne a autant de cellules que d'en-tetes (aucune casse)."""
        html = benchmarks.render_body(self.data, ETAT_VIDE)
        tableaux = re.findall(r"<table[^>]*>.*?</table>", html, re.S)
        self.assertGreater(len(tableaux), 1)
        for tableau in tableaux:
            entetes = len(re.findall(r"<th>", tableau))
            corps = tableau.split("<tbody>")[1] if "<tbody>" in tableau else tableau
            for ligne in re.findall(r"<tr(?: [^>]*)?>(.*?)</tr>", corps, re.S):
                self.assertEqual(len(re.findall(r"<td", ligne)), entetes, ligne[:120])

    def test_cles_historiques_conservees(self):
        """Les titres et compteurs attendus par les specs e2e ne changent pas."""
        html = benchmarks.render_body(self.data, ETAT_VIDE)
        familles = {e.get("kind") for e in self.data["entries"]}
        attendus = {"Index unifie des tests LLM", "Duels HTML", "Vitesse &amp; VRAM",
                    "Batteries de petites taches", "HumanEval", "Contexte long"}
        titres = {"duel": "Duels HTML", "vram": "Empreinte VRAM", "speed": "Vitesse &amp; VRAM",
                  "battery": "Batteries de petites taches", "humaneval": "HumanEval",
                  "context": "Contexte long"}
        for kind, titre in titres.items():
            if kind in familles:
                attendus.add(titre)
        for attendu in attendus:
            self.assertIn(attendu, html)
        for kind, titre in titres.items():
            if kind not in familles:
                self.assertNotIn(f"<h2>{titre} ", html)   # famille absente : pas de carte
        self.assertEqual(html.count(f'{len(self.data["entries"])} entrees'), 1)

    def test_page_complete_avec_etat_et_requete(self):
        cible = self.norm[0]["entry_id"]
        etat = {"hidden": [cible], "deleted": [], "filters": {}}
        page = benchmarks.render_benchmarks_page(state=etat, query={"masques": ["1"]})
        self.assertIsInstance(page, bytes)
        texte = page.decode("utf-8")
        self.assertIn("Benchmarks LLM", texte)
        self.assertIn('href="/graph"', texte)
        self.assertNotIn('class="res-masque"', texte)
        page2 = benchmarks.render_benchmarks_page(state=etat, query="?masques=0")
        self.assertIn('class="res-masque"', page2.decode("utf-8"))

    def test_rendu_sans_argument(self):
        """app.py appelle encore `render_benchmarks_page()` sans argument."""
        page = benchmarks.render_benchmarks_page()
        self.assertIsInstance(page, bytes)
        self.assertIn(b"</html>", page)

    def test_liens_de_filtre_adaptes_a_la_requete(self):
        """Sans requete transmise par app.py, aucun lien « ?masques=1 » mort."""
        page = benchmarks.render_benchmarks_page(state=ETAT_VIDE).decode("utf-8")
        self.assertNotIn("?masques=1", page)
        self.assertIn("restaurables depuis", page)          # renvoi vers /graph
        self.assertIn('id="voir-masques"', page)            # la bascule JS reste la
        page2 = benchmarks.render_benchmarks_page(state=ETAT_VIDE, query={}).decode("utf-8")
        self.assertIn("?masques=1", page2)


class TestPasDecriture(unittest.TestCase):
    """Aucun de ces rendus ne doit creer de fichier dans le depot."""

    def test_rendus_sans_ecriture(self):
        avant = sorted(os.listdir(ROOT))
        benchmarks.render_benchmarks_page(state=ETAT_VIDE)
        page_graph.render_graph_page(index_de([entree_duel()]), ETAT_VIDE, {})
        self.assertEqual(sorted(os.listdir(ROOT)), avant)


if __name__ == "__main__":
    unittest.main(verbosity=2)
