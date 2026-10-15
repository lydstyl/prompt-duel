#!/usr/bin/env python3
"""Tests de l'axe LLM : `/llms` (page « Tests par LLM ») et `/graph?llm=…` (v1.7).

Ce que ces tests verrouillent (la demande de Gabriel : « comparer des LLM, pas des
dates de tests ») :

  * un LLM = un **nom de modele** : la quantification, le contexte, le mode de
    raisonnement et la machine ne creent pas un second LLM (`famille_de`) ;
  * un nom inconnu tombe en `autre-…` — jamais fusionne avec un modele de la table,
    jamais d'exception ;
  * les artefacts sans nom de modele (journaux bruts) sont regroupes dans le bloc
    « mesures sans modele identifie », et seuls ceux qu'une regle RATTACHEMENTS
    explicite designe sont agreges a un LLM du parc ;
  * `/llms` : etape 1 = catalogue (`<select multiple>`), etape 2 = tableau
    « test × LLM » + detail chronologique des deux LLM ;
  * `/graph?llm=&llm=` : une serie par LLM, la **meilleure** valeur mesuree par test
    (jamais une moyenne entre deux dates), bornes 2-6, cle inconnue signalee mais
    jamais appliquee, et aucune source de regression sur l'axe `?sel=` historique.

Aucune ecriture : index de fixtures ou index reel du vault en LECTURE SEULE.
"""
import json
import re
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llms        # noqa: E402
import metrics     # noqa: E402
import page_graph  # noqa: E402

VAULT_INDEX = Path("/home/lydstyl/agentsmirror-mnt/vaults/personnel/"
                   "documents/llm-benchmarks/index.json")
ETAT_VIDE = {"hidden": [], "deleted": [], "filters": {}}


# --------------------------------------------------------------- outils de test
def svgs(page):
    """Fragments <svg> de la page, valides en XML (leve si mal formes)."""
    fragments = re.findall(r"<svg\b.*?</svg>", page, re.S)
    for fragment in fragments:
        ET.fromstring(fragment)
    return fragments


def entree(model, *, kind="duel", date="2026-10-15T10:00:00", quant="", label=None,
           tok_s=100.0, lat_median=2.0, slug=None, **kw):
    """Entree d'index synthetique, forme REELLE : les valeurs vivent au premier
    niveau (`tok_s`, `score_avg`…) — `metrics.normalize_entries` en deduit
    `metric_values`, comme pour un vrai fichier de l'index."""
    slug = slug or metrics.slug(model)
    nom_variant = f"{model} {quant}".strip()
    entry = {
        "kind": kind,
        "date": date,
        "model": nom_variant,
        "model_slug": metrics.slug(nom_variant),
        "label": label or nom_variant,
        "source": f"benches/{date[:10]}/{slug}.json",
        "duration_s": 12.0,
        "variant": {
            "key": f"{slug}_{date[8:10]}",
            "label": nom_variant,
            "model": nom_variant,
            "model_slug": metrics.slug(nom_variant),
            "kind": kind,
            "ctx": kw.pop("ctx", 8192),
        },
    }
    if kind == "duel":
        entry.update({"tok_s": tok_s, "prompts_ok": 3, "prompts_total": 3,
                      "tokens_out": 500})
    elif kind == "battery":
        entry.update({"score_avg": tok_s, "lat_median": lat_median,
                      "lat_max": lat_median, "fails": ["T4_date_jour_semaine"]})
    else:
        entry.update({"pp": tok_s, "tg": lat_median})
    entry.update(kw)
    return entry


def index_de(entries):
    return {"generated_at": "2026-10-15T22:00:00", "counts": {}, "entries": list(entries),
            "_source": "<fixture>"}


def index_deux_llms():
    """Deux LLM, plusieurs variantes/quantifications et plusieurs dates chacun."""
    entrees = [
        # FlashNext : 3 tirs, 2 quantifications, 2 dates -> UN seul LLM
        entree("qwen3.8-flash-next-iq3_xxs", date="2026-10-13T09:00:00", tok_s=41.0),
        entree("qwen3.8-flash-next-unsloth-ud-q4_k_xl", date="2026-10-15T09:00:00", tok_s=48.5),
        entree("Gabriel (FlashNext Q4_K_M — thinking ON)", date="2026-10-15T21:00:00", tok_s=44.0),
        # Qwen3.8-27B : 2 tirs, 2 dates
        entree("Qwen3.8-27B UD-Q4_K_XL", date="2026-10-14T09:00:00", tok_s=61.0),
        entree("Gabriel (Qwen3.8-27B-IQ4_XS ctx262144)", date="2026-10-15T18:00:00", tok_s=58.0),
    ]
    return index_de(entrees)


# =========================================================================
# 1) Regroupement : un LLM = un nom de modele
# =========================================================================
class TestFamilles(unittest.TestCase):

    def test_meme_modele_a_travers_slugs_et_quantifications(self):
        """(a) trois noms d'un meme modele -> UNE famille (le coeur du sujet)."""
        modeles = ("qwen3.8-flash-next-iq3_xxs",
                   "qwen3.8-flash-next-unsloth-ud-q4_k_xl",
                   "Gabriel (FlashNext Q4_K_M — thinking ON)")
        cles = set()
        for modele in modeles:
            with self.subTest(modele=modele):
                cle, libelle = llms.famille_de({"model": modele})
                self.assertEqual(cle, "flashnext-177b")
                self.assertEqual(libelle, "FlashNext 177B")
                cles.add(cle)
        self.assertEqual(len(cles), 1)

    def test_contexte_thinking_et_machine_ne_font_pas_un_autre_llm(self):
        base = {"model": "Qwen3.8-27B", "ctx": 8192}
        for variante in (dict(base, ctx=131072),
                         dict(base, thinking=True),
                         dict(base, temperature=0.7),
                         {"model": "Marie (Qwen3.8-27B-Q6_K)"},
                         {"model": "Louis (Qwen3.8-27B-UD-Q3_K_XL ctx131072)"}):
            with self.subTest(variante=variante):
                self.assertEqual(llms.famille_de(variante)[0], "qwen3-8-27b")

    def test_quantification_ne_fait_pas_un_autre_llm(self):
        modeles = {"FlashNext 177B UD-Q4_K_XL": "flashnext-177b",
                   "FlashNext 177B IQ3_XXS": "flashnext-177b",
                   "Qwen3.8-27B UD-Q4_K_XL": "qwen3-8-27b",
                   "Qwen3.8-27B-IQ4_XS ctx262144": "qwen3-8-27b",
                   "Swift-1.5-27B IQ4_XS": "swift15-27b",
                   "Bonsai 2 27B UD-Q4_K_XL": "bonsai-27b"}
        for nom, attendu in modeles.items():
            with self.subTest(nom=nom):
                self.assertEqual(llms.famille_de({"model": nom})[0], attendu)

    def test_uncensored_et_swift15_ne_tombent_pas_dans_le_modele_de_base(self):
        """L'ORDRE de la table compte : un derive passe avant son modele de base."""
        self.assertEqual(llms.famille_de({"model": "Qwen3.8-27B-Uncensored-IQ4_XS"})[0],
                         "qwen3-8-27b-uncensored")
        self.assertEqual(llms.famille_de({"model": "Qwen3.6-35B-A3B UD-Q4_K_XL"})[0],
                         "qwen3-6-35b")
        self.assertEqual(llms.famille_de({"model": "Qwen3.6-27B-IQ4_NL"})[0], "qwen3-6-27b")

    def test_nom_inconnu_tombe_en_autre_sans_exception(self):
        """(b) un modele inconnu est en `autre-…`, jamais fusionne, jamais leve."""
        for entree_testee in ({"model": "ModelX-9000-UD-Q4_K_XL"},
                              {"model": "truc bizarroïde"},
                              {"label": "mon-super-modele"},
                              {},
                              None,
                              {"model": None, "variant": {"model": 12}}):
            with self.subTest(entree=entree_testee):
                cle, libelle = llms.famille_de(entree_testee)
                self.assertTrue(cle.startswith("autre-"), cle)
                self.assertTrue(isinstance(libelle, str))

    def test_sans_modele_du_tout(self):
        cle, libelle = llms.famille_de({})
        self.assertEqual(cle, "autre-inconnu")
        self.assertTrue(libelle)

    def test_rattachements_explicites_et_justifies(self):
        """Les artefacts sans nom de modele ne sont agreges que par regle explicite."""
        cas = {
            "35B-A3B UD-IQ4_NL ctx262144 n_max2 (mini-test)": "qwen3-6-35b",
            "35b n4": "qwen3-6-35b",
            "35b split65": "qwen3-6-35b",
            "probe-k2q4-48k.speed": "k2-horizon",
        }
        for nom, attendu in cas.items():
            with self.subTest(nom=nom):
                entree_testee = {"model": nom}
                self.assertEqual(llms.famille_de(entree_testee)[0], attendu)
                cle, justification = llms.rattachement_de(entree_testee)
                self.assertEqual(cle, attendu)
                self.assertTrue(justification, "une regle doit dire POURQUOI elle rattache")

    def test_artefacts_sans_modele_restent_non_identifies(self):
        for nom in ("dry-run", "smoke-n1", "probe-ptq1-32k.speed", "live-ptq1-32k"):
            with self.subTest(nom=nom):
                entree_testee = {"model": nom}
                cle, _ = llms.famille_de(entree_testee)
                self.assertTrue(cle.startswith("autre-"), cle)
                self.assertIsNone(llms.rattachement_de(entree_testee)[0])

    def test_build_familles_met_les_non_identifies_en_bas(self):
        entrees = index_deux_llms()["entries"] + [entree("dry-run"), entree("smoke-n1")]
        familles = llms.build_familles(metrics.normalize_entries(entrees))
        drapeaux = [f["identifie"] for f in familles]
        self.assertIn(True, drapeaux)
        self.assertEqual(drapeaux, sorted(drapeaux, reverse=True),
                         "les mesures sans modele doivent tomber en bas du catalogue")
        # trois tirs FlashNext = UNE famille, pas trois
        flashnext = [f for f in familles if f["cle"] == "flashnext-177b"]
        self.assertEqual(len(flashnext), 1)
        self.assertEqual(len(flashnext[0]["entries"]), 3)
        self.assertEqual(len(flashnext[0]["variantes"]), 3)

    def test_famille_de_ne_modifie_pas_les_entrees(self):
        entree_testee = entree("Qwen3.8-27B UD-Q4_K_XL")
        avant = json.dumps(entree_testee, sort_keys=True, ensure_ascii=False)
        llms.famille_de(entree_testee)
        llms.build_familles([entree_testee])
        self.assertEqual(json.dumps(entree_testee, sort_keys=True, ensure_ascii=False), avant)


# =========================================================================
# 2) Selection /llms : alias, dedoublonnage, bornes, cle inconnue
# =========================================================================
class TestParseSelection(unittest.TestCase):

    def setUp(self):
        self.index = index_deux_llms()
        self.familles = llms.build_familles(metrics.normalize_entries(self.index["entries"]))
        self.cles = [f["cle"] for f in self.familles]
        self.assertGreaterEqual(len(self.cles), 2)

    def test_alias_de_selection(self):
        """(c) `llm`, `llms`, `modele`, `modeles`, `model` designent le meme axe."""
        for parametre in ("llm", "llms", "modele", "modeles", "model"):
            with self.subTest(parametre=parametre):
                selection, inconnues, tronquees = llms.parse_selection(
                    {parametre: [self.cles[0], self.cles[1]]}, self.familles)
                self.assertEqual(selection, self.cles[:2])
                self.assertEqual(inconnues, [])
                self.assertEqual(tronquees, [])

    def test_dedoublonnage_et_separateurs(self):
        selection, inconnues, _ = llms.parse_selection(
            {"llm": [self.cles[0], self.cles[0], f"{self.cles[1]},{self.cles[1]}"]},
            self.familles)
        self.assertEqual(selection, self.cles[:2])
        self.assertEqual(inconnues, [])

    def test_max_selection(self):
        beaucoup = [{"cle": f"llm-{i}"} for i in range(9)]
        demandees = [f"llm-{i}" for i in range(9)]
        selection, inconnues, tronquees = llms.parse_selection({"llm": demandees}, beaucoup)
        self.assertEqual(len(selection), llms.MAX_SELECTION)
        self.assertEqual(tronquees, demandees[llms.MAX_SELECTION:])
        self.assertEqual(inconnues, [])

    def test_cle_inconnue_signalee_et_non_appliquee(self):
        selection, inconnues, _ = llms.parse_selection(
            {"llm": [self.cles[0], "modele-qui-nexiste-pas"]}, self.familles)
        self.assertEqual(selection, [self.cles[0]])
        self.assertEqual(inconnues, ["modele-qui-nexiste-pas"])

    def test_parametre_ignore_nest_pas_une_selection(self):
        selection, inconnues, tronquees = llms.parse_selection(
            {"kind": ["duel"], "llm": [self.cles[0]]}, self.familles)
        self.assertEqual(selection, [self.cles[0]])
        self.assertEqual((inconnues, tronquees), ([], []))


# =========================================================================
# 3) Page /llms : catalogue puis tableau + detail
# =========================================================================
class TestPageLlms(unittest.TestCase):

    def setUp(self):
        self.index = index_deux_llms()
        self.familles = llms.build_familles(metrics.normalize_entries(self.index["entries"]))
        self.cles = [f["cle"] for f in self.familles]

    def page(self, query=None):
        return llms.render_llms_page(self.index, ETAT_VIDE, query or {}).decode("utf-8")

    def test_etape1_rend_le_select_multiple(self):
        """(d) l'etape 1 est un catalogue : une case par LLM, selection multiple."""
        page = self.page()
        self.assertIn('<select name="llm" multiple', page)
        self.assertIn('name="llm"', page)
        for cle in self.cles:
            self.assertIn(f'value="{cle}"', page)
        self.assertEqual(re.findall(r"<svg", page), [], "l'etape 1 ne trace rien")

    def test_etape1_affiche_le_nombre_de_tests_et_la_plage_de_dates(self):
        page = self.page()
        self.assertIn("mesures", page)
        self.assertIn("test(s)", page)
        self.assertIn("13/10", page)      # premiere mesure FlashNext de la fixture

    def test_etape2_rend_le_tableau_et_les_deux_libelles(self):
        """(e) le tableau « test × LLM » et le detail des deux LLM sont rendus."""
        page = self.page({"llm": self.cles[:2]})
        self.assertGreaterEqual(page.count("<table"), 3)
        self.assertIn("Tableau comparatif", page)
        self.assertIn("FlashNext 177B", page)
        self.assertIn("Qwen3.8-27B", page)
        self.assertIn("meilleure valeur", page)
        # une colonne par LLM coche
        self.assertEqual(len(re.findall(r"<th>(FlashNext 177B|Qwen3\.8-27B)</th>", page)), 2)
        # le detail chronologique de CHAQUE LLM est la
        self.assertGreaterEqual(page.count('<div class="llm-carte">'), 2)

    def test_etape2_agrege_les_dates_et_compte_les_autres_mesures(self):
        """Trois tirs FlashNext : une seule colonne, la meilleure valeur + « +N mesure(s) »."""
        page = self.page({"llm": ["flashnext-177b", "qwen3-8-27b"]})
        self.assertIn("FlashNext 177B", page)
        # 3 mesures FlashNext sur le meme test -> 2 autres mesures annoncees
        self.assertRegex(page, r"\+[12] mesure")

    def test_etape2_cle_inconnue_signalee(self):
        page = self.page({"llm": ["flashnext-177b", "qwen3-8-27b", "pas-un-llm"]})
        self.assertIn("pas-un-llm", page)
        self.assertIn("inconnue", page)

    def test_etape2_bornes(self):
        page = self.page({"llm": [f"llm-{i}" for i in range(9)]})
        self.assertIn("Choisis les LLM à comparer", page)   # aucune cle connue -> catalogue
        self.assertIn("llm-0", page)                        # ... et les cles sont nommees
        self.assertIn("inconnue", page)

    def test_trois_llm_selectionnes(self):
        entrees = self.index["entries"] + [entree("Qwen3.6-35B-A3B UD-Q4_K_XL")]
        index = index_de(entrees)
        cles = [f["cle"] for f in llms.build_familles(metrics.normalize_entries(entrees))]
        page = llms.render_llms_page(index, ETAT_VIDE, {"llm": cles[:3]}).decode("utf-8")
        self.assertIn("3 LLM comparé(s)", page)

    def test_entree_masquee_napparait_pas(self):
        """(f) une entree masquee par la visibilite sort de la page."""
        norm = metrics.normalize_entries(self.index["entries"])
        visibles = [e for e in norm if llms.famille_de(e)[0] == "qwen3-8-27b"]
        self.assertTrue(visibles)
        etat = {"hidden": [e["entry_id"] for e in visibles], "deleted": [], "filters": {}}
        page = llms.render_llms_page(self.index, etat, {"llm": ["flashnext-177b"]}).decode("utf-8")
        self.assertNotIn("Qwen3.8-27B UD-Q4_K_XL", page)
        self.assertIn("FlashNext 177B", page)

    def test_index_vide_ou_abime(self):
        for index in ({}, {"entries": None}, {"entries": [None, 3, "x"]},
                      index_de([{"kind": "duel", "date": "2026-10-15T10:00:00"}])):
            with self.subTest(index=str(index)[:40]):
                page = llms.render_llms_page(index, ETAT_VIDE, {}).decode("utf-8")
                self.assertIn("<html", page)

    def test_index_reel_du_vault(self):
        if not VAULT_INDEX.exists():
            self.skipTest(f"index du vault absent : {VAULT_INDEX}")
        with open(VAULT_INDEX, encoding="utf-8") as fh:
            index = json.load(fh)
        familles = llms.build_familles(metrics.normalize_entries(index.get("entries") or []))
        self.assertGreaterEqual(len(familles), 5)
        identifies = [f for f in familles if f["identifie"]]
        self.assertGreaterEqual(len(identifies), 5)
        cles = [f["cle"] for f in identifies[:2]]
        page = llms.render_llms_page(index, ETAT_VIDE, {"llm": cles}).decode("utf-8")
        self.assertIn("Tableau comparatif", page)
        self.assertEqual(page.count("inconnue"), 0)
        xml_valides = svgs(page) if "<svg" in page else []
        self.assertEqual(xml_valides, [], "la page /llms ne trace pas de courbe")


# =========================================================================
# 4) /graph, axe LLM
# =========================================================================
class TestGraphAxeLlm(unittest.TestCase):

    def setUp(self):
        self.index = index_deux_llms()

    def test_mode_llm_produit_des_svg_sans_cle_inconnue(self):
        """(g) /graph en mode LLM trace une serie par LLM, sans cle inconnue."""
        page = page_graph.render_graph_page(self.index, ETAT_VIDE,
                                            {"llm": ["flashnext-177b", "qwen3-8-27b"]})
        self.assertGreaterEqual(len(svgs(page)), 1)
        self.assertEqual(page.count("inconnu"), 0)
        self.assertIn("Comparaison de 2 LLMs", page)
        self.assertIn("FlashNext 177B", page)
        self.assertIn("Qwen3.8-27B", page)

    def test_une_serie_par_llm_dans_la_legende(self):
        page = page_graph.render_graph_page(self.index, ETAT_VIDE,
                                            {"llm": ["flashnext-177b", "qwen3-8-27b"]})
        cles = set(re.findall(r'data-serie="([^"]+)"', page))
        self.assertTrue(cles)
        for cle in cles:
            self.assertIn(cle, {"flashnext-177b", "qwen3-8-27b"}, f"serie inattendue : {cle}")

    def test_un_seul_llm_ne_trace_rien_et_le_dit(self):
        page = page_graph.render_graph_page(self.index, ETAT_VIDE, {"llm": ["flashnext-177b"]})
        self.assertEqual(svgs(page), [])
        self.assertIn("au moins 2", page)
        self.assertIn("un seul LLM", page)

    def test_cle_inconnue_signalee_et_non_appliquee(self):
        page = page_graph.render_graph_page(
            self.index, ETAT_VIDE, {"llm": ["flashnext-177b", "qwen3-8-27b", "pas-un-llm"]})
        self.assertIn("pas-un-llm", page)
        self.assertGreaterEqual(len(svgs(page)), 1)      # les 2 vraies cles suffisent
        vue = page_graph.build_view(self.index, ETAT_VIDE,
                                    {"llm": ["flashnext-177b", "pas-un-llm"]})
        self.assertEqual(vue["selection_inconnues"], ["pas-un-llm"])
        self.assertEqual(vue["selection_llm"], ["flashnext-177b"])

    def test_max_six_llm(self):
        entrees = list(self.index["entries"])
        for i in range(8):
            entrees.append(entree(f"ModelX-{i}-UD-Q4_K_XL", date=f"2026-10-1{i}T10:00:00"))
        index = index_de(entrees)
        cles = [f["cle"] for f in llms.build_familles(metrics.normalize_entries(entrees))]
        vue = page_graph.build_view(index, ETAT_VIDE, {"llm": cles})
        self.assertEqual(len(vue["selection_llm"]), page_graph.MAX_SELECTION)
        self.assertIn("LLM au maximum", vue["selection_warning_llm"])
        page = page_graph.render_graph_page(index, ETAT_VIDE, {"llm": cles})
        self.assertIn("LLM au maximum", page)

    def test_meilleure_valeur_par_llm_pas_une_moyenne(self):
        """Trois tirs FlashNext (41, 48.5, 44 t/s) -> 48.5 tracé, pas la moyenne."""
        vue = page_graph.build_view(self.index, ETAT_VIDE,
                                    {"llm": ["flashnext-177b", "qwen3-8-27b"]})
        valeurs = {}
        for chart in vue["charts"]:
            if (chart.get("metric") or {}).get("key") != "tok_s":
                continue
            for serie in chart["data"]["series"]:
                for point in serie["points"]:
                    valeurs.setdefault(serie["key"], []).append(point["value"])
        self.assertIn(48.5, valeurs.get("flashnext-177b", []))
        self.assertNotIn(44.5, valeurs.get("flashnext-177b", []))   # jamais la moyenne
        self.assertIn(61.0, valeurs.get("qwen3-8-27b", []))

    def test_serie_llm_sens_des_metriques(self):
        """`better` = low (latence) : la plus PETITE valeur gagne."""
        entrees = [entree("Qwen3.8-27B UD-Q4_K_XL", kind="battery",
                          date="2026-10-14T09:00:00", tok_s=0.8, lat_median=9.0),
                   entree("Qwen3.8-27B UD-Q4_K_XL", kind="battery",
                          date="2026-10-15T09:00:00", tok_s=0.9, lat_median=2.0),
                   entree("Swift-1.5-27B IQ4_XS", kind="battery",
                          date="2026-10-15T09:00:00", tok_s=0.7, lat_median=5.0)]
        data = page_graph._series_llm(entrees, metric_key="lat_median", kind="battery")
        par_llm = {s["key"]: s["points"][0]["value"] for s in data["series"]}
        self.assertEqual(par_llm["qwen3-8-27b"], 2.0, "la latence la plus basse doit gagner")
        # la mesure ecartee n'est pas perdue : elle est signalee
        self.assertTrue(data["duplicates"])
        self.assertIn("better", data["duplicates"][0])

    def test_compatibilite_ancien_lien_llm_vers_une_variante(self):
        """`?llm=<cle de variante>` (avant la v1.7) reste une selection de variantes."""
        cles = [item["key"] for item in metrics.series_catalogue(self.index["entries"])]
        self.assertGreaterEqual(len(cles), 2)
        vue = page_graph.build_view(self.index, ETAT_VIDE, {"llm": cles[:2]})
        self.assertEqual(vue["etape"], 2)
        self.assertEqual(vue["axe"], "variant")
        self.assertEqual(vue["selection"], cles[:2])
        self.assertEqual(vue["selection_llm"], [])
        page = page_graph.render_graph_page(self.index, ETAT_VIDE, {"llm": cles[:2]})
        self.assertGreaterEqual(len(svgs(page)), 1)

    def test_axe_sel_inchange(self):
        """L'axe historique n'est pas touche : `?sel=` rend les memes series qu'avant."""
        cles = [item["key"] for item in metrics.series_catalogue(self.index["entries"])]
        page = page_graph.render_graph_page(self.index, ETAT_VIDE, {"sel": cles[:2]})
        self.assertIn("Comparaison de 2 réglages", page)
        self.assertEqual(page.count('<input type="hidden" name="sel"'), 2)
        self.assertGreaterEqual(len(svgs(page)), 1)

    def test_filtre_de_reglages_conserve_sur_laxe_llm(self):
        """Des reglages coches EN PLUS servent de filtre, et c'est dit en clair."""
        norm = metrics.normalize_entries(self.index["entries"])
        variante = next(e for e in norm
                        if llms.famille_de(e)[0] == "flashnext-177b")["variant"]["key"]
        page = page_graph.render_graph_page(self.index, ETAT_VIDE,
                                            {"llm": ["flashnext-177b", "qwen3-8-27b"],
                                             "sel": [variante]})
        self.assertIn("conservés comme", page)
        self.assertIn('name="llm" value="flashnext-177b"', page)

    def test_etape1_liste_les_llm_et_garde_le_catalogue_des_reglages(self):
        page = page_graph.render_graph_page(self.index, ETAT_VIDE, {})
        self.assertIn('id="selection-llm"', page)
        self.assertIn("Comparer des LLM", page)
        self.assertIn('name="llm"', page)
        self.assertIn("Comparer des réglages", page)
        self.assertIn('name="sel"', page)
        self.assertEqual(svgs(page), [])
        # le bouton de l'axe LLM ne s'appelle pas « Comparer » (deux formulaires)
        self.assertIn("Afficher les LLM", page)

    def test_lien_llms_dans_les_en_tetes(self):
        page = page_graph.render_graph_page(self.index, ETAT_VIDE, {})
        self.assertIn('href="/llms"', page)


# =========================================================================
# 5) Index reel du vault : la comparaison demandee, en vrai (lecture seule)
# =========================================================================
class TestIndexReelAxeLlm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not VAULT_INDEX.exists():
            raise unittest.SkipTest(f"index du vault absent : {VAULT_INDEX}")
        cls.stat = VAULT_INDEX.stat()
        with open(VAULT_INDEX, encoding="utf-8") as fh:
            cls.index = json.load(fh)
        cls.familles = llms.catalogue_llm(metrics.normalize_entries(cls.index["entries"]))

    def test_catalogue_non_identifie_regroupe_en_bas(self):
        non_identifies = [f for f in self.familles if not f["identifie"]]
        cles = [f["key"] for f in self.familles]
        for famille in non_identifies:
            self.assertTrue(famille["key"].startswith("autre-"))
        identifiees = [f["key"] for f in self.familles if f["identifie"]]
        self.assertEqual(cles, identifiees + [f["key"] for f in non_identifies])

    def test_deux_gros_llm_du_parc_se_comparent(self):
        identifiees = [f for f in self.familles if f["identifie"]]
        cles = [f["key"] for f in sorted(identifiees, key=lambda f: -f["mesures"])[:2]]
        page = page_graph.render_graph_page(self.index, ETAT_VIDE, {"llm": cles})
        self.assertGreaterEqual(len(svgs(page)), 3)
        self.assertEqual(page.count("inconnu"), 0)
        for cle in cles:
            self.assertIn(cle, page)

    def test_index_du_vault_non_modifie(self):
        self.assertEqual(VAULT_INDEX.stat().st_mtime, self.stat.st_mtime)


if __name__ == "__main__":
    unittest.main()
