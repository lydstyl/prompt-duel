#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests unitaires de charts.py (stdlib `unittest`, aucun réseau, aucune I/O hors
lecture des fixtures).

Ce que l'on vérifie :
1. le XML produit est bien formé et respecte le contrat de rendu (viewBox,
   `width="100%"`, `max-width`, police héritée, thème sombre) ;
2. l'échappement HTML de TOUT texte injecté (<script>, &, guillemets) ;
3. les cas limites (vide, 1 point, 0, négatif, None, unité absente, libellés
   très longs, 40+ séries) : jamais d'exception, jamais de NaN/Infinity ;
4. la cohérence des échelles (bornes « jolies » qui encadrent les données,
   hauteurs de barres proportionnelles aux valeurs, axe du zéro) ;
5. le déterminisme (deux appels identiques = chaîne identique).

Lancement :  python3 -m unittest discover -s tests -v
"""

import ast
import json
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:            # `import charts` depuis tests/
    sys.path.insert(0, str(ROOT))

import charts  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NS = "{http://www.w3.org/2000/svg}"


def charger(nom):
    return json.loads((FIXTURES / nom).read_text(encoding="utf-8"))


def racine(svg):
    """Parse le SVG : lève si le XML est mal formé (c'est le but du test)."""
    return ET.fromstring(svg)


def rects(svg):
    return racine(svg).findall(f".//{NS}rect")


def textes(svg):
    return [t.text or "" for t in racine(svg).findall(f".//{NS}text")]


def titres(svg):
    return [t.text or "" for t in racine(svg).findall(f".//{NS}title")]


def donnees(*, valeurs_par_serie, groupes=("g1", "g2", "g3"), unite="t/s",
            label="Prefill", better="high", duree=90):
    """Construit une donnée au format §6 (facile à lire dans les tests)."""
    series = []
    for i, valeurs in enumerate(valeurs_par_serie):
        points = []
        for j, val in enumerate(valeurs):
            points.append({
                "group": groupes[j], "value": val, "duration_s": duree,
                "entry_id": f"s{i}-{groupes[j]}", "date": "2026-09-30T12:00", "status": "ok",
            })
        series.append({"key": f"s{i}", "label": f"Série {i}", "hidden": False, "points": points})
    return {
        "metric": {"key": "pp", "label": label, "unit": unite, "better": better},
        "groups": [{"key": g, "label": g.upper()} for g in groupes],
        "series": series,
    }


# ---------------------------------------------------------------------------
# 1. Formatage FR (durées, nombres, dates, troncature)
# ---------------------------------------------------------------------------

class TestFormatage(unittest.TestCase):

    def test_duree_fr_lisible(self):
        attendu = {
            755: "12 min 35 s",
            60: "1 min",
            59: "59 s",
            0: "0 s",
            3600: "1 h 00 min",
            3900: "1 h 05 min",
            3661: "1 h 01 min 01 s",
            45.4: "45 s",
        }
        for entree, attendu_txt in attendu.items():
            self.assertEqual(charts._fmt_duration(entree), attendu_txt, entree)

    def test_duree_invalide(self):
        for vide in (None, "", "n/d", "12 min", float("nan"), float("inf")):
            self.assertEqual(charts._fmt_duration(vide), "—", vide)

    def test_duree_courte_pour_les_graduations(self):
        self.assertEqual(charts._fmt_duration(755, short=True), "12 min")
        self.assertEqual(charts._fmt_duration(3900, short=True), "1 h 05")
        self.assertEqual(charts._fmt_duration(45, short=True), "45 s")

    def test_nombres_fr(self):
        self.assertEqual(charts._fmt_num(12345.678), "12 345,7")
        self.assertEqual(charts._fmt_num(1420.5), "1 420,5")
        self.assertEqual(charts._fmt_num(245868.0), "245 868")
        self.assertEqual(charts._fmt_num(12.3), "12,3")
        self.assertEqual(charts._fmt_num(0.0), "0")
        self.assertEqual(charts._fmt_num(-3.25), "-3,25")
        self.assertEqual(charts._fmt_num(float("inf")), "—")
        self.assertEqual(charts._fmt_value(12.3, "t/s"), "12,3 t/s")
        self.assertEqual(charts._fmt_value(12.3, ""), "12,3")   # unité absente
        self.assertEqual(charts._fmt_value(12.3, None), "12,3")

    def test_dates_sans_horloge(self):
        self.assertEqual(charts._fmt_date("2026-09-30T11:46:08"), "30/09/2026 11:46")
        self.assertEqual(charts._fmt_date("2026-09-30"), "30/09/2026")
        self.assertEqual(charts._fmt_date(None), "—")
        self.assertEqual(charts._fmt_date("hier"), "hier")

    def test_troncature_en_pixels(self):
        self.assertEqual(charts._trunc_px("abc", 100), "abc")
        self.assertEqual(charts._trunc_px("", 100), "")
        self.assertEqual(charts._trunc_px("abcdefghij", 0), "")
        court = charts._trunc_px("abcdefghij", 30)
        self.assertTrue(court.endswith("…"))
        self.assertLessEqual(charts._largeur(court), 30)   # tient dans la place donnée
        self.assertLessEqual(charts._largeur(charts._trunc_px("M" * 40, 60)), 60)
        # la mesure tient compte des majuscules (plus larges que les minuscules)
        self.assertGreater(charts._largeur("MMMM"), charts._largeur("mmmm"))

    def test_nombres_en_texte(self):
        self.assertEqual(charts._num("11,7"), 11.7)
        self.assertEqual(charts._num("11.7 / 11.9 Go"), 11.7)
        self.assertIsNone(charts._num(True))
        self.assertIsNone(charts._num(None))
        self.assertIsNone(charts._num("n/d"))
        self.assertIsNone(charts._num(float("nan")))
        self.assertIsNone(charts._num(float("-inf")))


# ---------------------------------------------------------------------------
# 2. Échelle « jolie » et cohérence
# ---------------------------------------------------------------------------

class TestEchelle(unittest.TestCase):

    def test_bornes_arrondies_et_encadrantes(self):
        jeux = [
            [1.0, 2.0, 3.7], [0.0, 0.0], [0.001, 0.004], [-5.0, 0.0],
            [1234.0, 9876.0], [899.2, 1420.5], [1e6, 3.4e6], [-12.4, 72.5],
        ]
        for valeurs in jeux:
            with self.subTest(valeurs=valeurs):
                lo, hi, pas, ticks = charts._nice_scale(min(valeurs), max(valeurs))
                self.assertLessEqual(lo, min(valeurs))
                self.assertGreaterEqual(hi, max(valeurs))
                self.assertGreater(pas, 0)
                self.assertGreaterEqual(len(ticks), 2)
                self.assertLessEqual(len(ticks), 10)
                self.assertEqual(ticks[0], lo)
                self.assertAlmostEqual(ticks[-1], hi)
                # pas régulier
                for a, b in zip(ticks, ticks[1:]):
                    self.assertAlmostEqual(b - a, pas, places=9)
                # libellés de graduations tous distincts (pas de « 1.0 1.0 »)
                libelles = [charts._fmt_num(t) for t in ticks]
                self.assertEqual(len(libelles), len(set(libelles)))

    def test_borne_haute_jolie_pas_le_max(self):
        _, hi, _, _ = charts._nice_scale(0.0, 3.7)
        self.assertEqual(hi, 4.0)                     # pas 3.7
        _, hi2, pas2, _ = charts._nice_scale(0.0, 899.2)
        self.assertEqual(hi2 % pas2, 0)

    def test_valeurs_non_finies_ou_absentes(self):
        lo, hi, pas, ticks = charts._nice_scale(float("nan"), float("inf"))
        self.assertEqual((lo, hi, pas), (0.0, 1.0, 1.0))
        self.assertEqual(ticks, [0.0, 1.0])

    def test_hauteurs_proportionnelles(self):
        svg = charts.grouped_bars(donnees(valeurs_par_serie=[[100.0, 200.0, 50.0]]),
                                  width=980, height=420)
        barres = rects(svg)
        self.assertEqual(len(barres), 3)
        h = [float(r.get("height")) for r in barres]
        self.assertAlmostEqual(h[0] / h[1], 0.5, delta=0.02)
        self.assertAlmostEqual(h[2] / h[1], 0.25, delta=0.02)


# ---------------------------------------------------------------------------
# 3. Contrat SVG (autonome, thème sombre, sans JS ni CDN)
# ---------------------------------------------------------------------------

class TestContratSvg(unittest.TestCase):

    def setUp(self):
        self.fixtures = [charger("chart_speed.json"), charger("chart_weird.json"),
                         charger("chart_edge.json")]
        self.rendus = []
        for data in self.fixtures:
            self.rendus.append(charts.grouped_bars(data))
            self.rendus.append(charts.bars(data))
            self.rendus.append(charts.duration_chart(data))

    def test_xml_bien_forme(self):
        for svg in self.rendus:
            root = racine(svg)
            self.assertTrue(root.tag.endswith("svg"))

    def test_svg_autonome(self):
        for svg in self.rendus:
            self.assertIn('viewBox="0 0 ', svg)
            self.assertIn('width="100%"', svg)
            self.assertIn("max-width:", svg)
            self.assertIn("font-family:inherit", svg)
            self.assertIn('xmlns="http://www.w3.org/2000/svg"', svg)
            self.assertTrue(svg.endswith("</svg>"))

    def test_theme_sombre_et_pas_de_dependance(self):
        for svg in self.rendus:
            for interdit in ("<script", "javascript:", "cdn.", "http://cdn",
                             "https://", 'fill="#fff', 'fill="#ffffff', 'fill="white"'):
                self.assertNotIn(interdit, svg.lower().replace(
                    "http://www.w3.org/2000/svg", ""))
            # grille discrète + texte clair
            self.assertIn(charts.C_GRID, svg)
            self.assertIn(charts.C_FG, svg)

    def test_aucun_nan_ni_infini(self):
        for svg in self.rendus:
            for interdit in ("nan", "inf", "infinity"):
                self.assertNotIn(interdit, svg.lower())

    def test_imports_stdlib_uniquement(self):
        source = Path(charts.__file__).read_text(encoding="utf-8")
        arbre = ast.parse(source)
        modules = set()
        for noeud in ast.walk(arbre):
            if isinstance(noeud, ast.Import):
                modules.update(a.name.split(".")[0] for a in noeud.names)
            elif isinstance(noeud, ast.ImportFrom) and noeud.module:
                modules.add(noeud.module.split(".")[0])
        self.assertEqual(modules, {"html", "math", "re"})

    def test_fonctions_pures_sans_io_ni_horloge(self):
        source = Path(charts.__file__).read_text(encoding="utf-8")
        arbre = ast.parse(source)
        noms = set()
        for noeud in ast.walk(arbre):
            if isinstance(noeud, ast.Name):
                noms.add(noeud.id)
            elif isinstance(noeud, ast.Attribute):
                noms.add(noeud.attr)
        interdits = {"open", "time", "random", "datetime", "os", "subprocess",
                     "urllib", "socket", "requests", "environ", "now", "today"}
        self.assertFalse(noms & interdits, noms & interdits)

    def test_signatures_du_contrat(self):
        import inspect
        sig = inspect.signature(charts.grouped_bars)
        self.assertEqual(list(sig.parameters), ["data", "width", "height"])
        self.assertEqual(sig.parameters["width"].default, 980)
        self.assertEqual(sig.parameters["height"].default, 420)
        sig = inspect.signature(charts.bars)
        self.assertEqual(list(sig.parameters), ["data", "metric_key", "width", "height"])
        self.assertEqual(sig.parameters["height"].default, 320)
        self.assertEqual(list(inspect.signature(charts.duration_chart).parameters), ["data"])
        self.assertEqual(list(inspect.signature(charts.legend).parameters), ["data"])


# ---------------------------------------------------------------------------
# 4. Barres groupées : structure, couleurs, tooltips
# ---------------------------------------------------------------------------

class TestBarresGroupees(unittest.TestCase):

    def setUp(self):
        self.data = charger("chart_speed.json")
        self.svg = charts.grouped_bars(self.data)

    def test_une_barre_par_valeur_chiffree(self):
        attendues = sum(
            1 for s in self.data["series"] for p in s["points"]
            if isinstance(p.get("value"), (int, float))
        )
        self.assertEqual(len(rects(self.svg)), attendues)
        self.assertEqual(attendues, 13)          # fixture : 3 séries × 5 groupes − 2 absentes

    def test_une_couleur_par_serie(self):
        couleurs = {r.get("fill") for r in rects(self.svg)}
        self.assertLessEqual(len(couleurs), len(self.data["series"]))
        for couleur in couleurs:
            self.assertIn(couleur, charts.PALETTE)
        premiere = rects(self.svg)[0]
        self.assertEqual(premiere.get("fill"), charts.PALETTE[0])

    def test_un_tooltip_par_barre(self):
        self.assertEqual(len(titres(self.svg)), len(rects(self.svg)))

    def test_contenu_du_tooltip(self):
        tips = titres(self.svg)
        cible = [t for t in tips if "Vitesse court" in t and "Qwen3.6-35B · ctx 128k" in t]
        self.assertEqual(len(cible), 1)
        texte = cible[0]
        self.assertIn("Série : Qwen3.6-35B · ctx 128k", texte)
        self.assertIn("Groupe : Vitesse court", texte)
        self.assertIn("Valeur : 1 420,5 t/s", texte)
        self.assertIn("Durée : 2 min 32 s", texte)          # 152,4 s
        self.assertIn("Date : 30/09/2026 11:46", texte)
        self.assertIn("Statut : ok", texte)

    def test_statut_traduit_et_absent(self):
        self.assertTrue(any("Statut : ok" in t for t in titres(self.svg)))
        # le point en échec n'a pas de valeur (donc pas de barre) : il apparaît
        # dans le graphique des durées, où la donnée existe
        tips_duree = titres(charts.duration_chart(self.data))
        self.assertTrue(any("Statut : échec" in t for t in tips_duree))
        weird = charts.grouped_bars(charger("chart_weird.json"))
        self.assertTrue(any("Statut : —" in t for t in titres(weird)))
        self.assertTrue(any("Statut : échec" in t for t in titres(weird)))
        # « partiel » n'existe que là où la donnée est chiffrée (graphique des durées)
        self.assertTrue(any("Statut : partiel" in t
                            for t in titres(charts.duration_chart(charger("chart_weird.json")))))
        self.assertTrue(any("Durée : 12 min 35 s" in t for t in titres(weird)))

    def test_entree_absente_et_serie_masquee(self):
        rects_svg = rects(self.svg)
        self.assertTrue(all(r.get("data-serie") for r in rects_svg))
        opacites = {r.get("fill-opacity") for r in rects_svg}
        self.assertIn("0.35", opacites)          # série masquée = barre atténuée
        self.assertEqual(len([r for r in rects_svg if r.get("data-entry")]), 13)

    def test_libelles_de_groupes_sous_laxe(self):
        libelles = [t for t in textes(self.svg) if t in
                    ("Vitesse court", "Vitesse 128k", "Vitesse 256k")]
        self.assertEqual(sorted(libelles), ["Vitesse 128k", "Vitesse 256k", "Vitesse court"])

    def test_valeur_affichee_quand_la_place_le_permet(self):
        # 3 séries × 5 groupes : les valeurs courtes tiennent au-dessus des barres
        self.assertIn("899,2", textes(self.svg))
        # « 1 420,5 » est trop large pour sa barre : masquée, mais dans l'info-bulle
        self.assertNotIn("1 420,5", textes(self.svg))
        self.assertTrue(any("Valeur : 1 420,5 t/s" in t for t in titres(self.svg)))
        # barres larges (1 seul groupe) : la même valeur est affichée
        large = donnees(valeurs_par_serie=[[1420.5]], groupes=("seul",))
        self.assertIn("1 420,5", textes(charts.grouped_bars(large)))

    def test_graduations_avec_unite_dans_le_titre(self):
        self.assertIn("Prefill (t/s)", textes(self.svg))
        self.assertIn("plus haut = mieux", textes(self.svg))

    def test_bas_mieux_pour_low(self):
        svg = charts.grouped_bars(donnees(valeurs_par_serie=[[1.0, 2.0, 3.0]], better="low"))
        self.assertIn("plus bas = mieux", textes(svg))


# ---------------------------------------------------------------------------
# 5. Barres simples
# ---------------------------------------------------------------------------

class TestBarresSimples(unittest.TestCase):

    def test_une_barre_par_point(self):
        data = charger("chart_speed.json")
        svg = charts.bars(data)
        attendues = sum(1 for s in data["series"] for p in s["points"]
                        if isinstance(p.get("value"), (int, float)))
        self.assertEqual(len(rects(svg)), attendues)
        self.assertEqual(len([t for t in titres(svg) if t.startswith("Série :")]), attendues)

    def test_taille_par_defaut_du_contrat(self):
        svg = charts.bars(charger("chart_speed.json"))
        self.assertIn('viewBox="0 0 980 320"', svg)

    def test_metric_key_lit_values(self):
        data = donnees(valeurs_par_serie=[[None, None, None]])
        data["series"][0]["points"][0]["values"] = {"pp": 123.0}
        data["series"][0]["points"][0]["value"] = None
        svg = charts.bars(data, metric_key="pp")
        self.assertEqual(len(rects(svg)), 1)
        self.assertIn("123", titres(svg)[0])

    def _labels_x(self, svg):
        """Libellés de l'axe X : les <text> à l'ordonnée la plus basse."""
        textes_el = racine(svg).findall(f".//{NS}text")
        y_max = max(float(e.get("y")) for e in textes_el)
        return sorted((float(e.get("x")), e.text or "")
                      for e in textes_el if float(e.get("y")) == y_max)

    def _verifie_sans_chevauchement(self, labels):
        """Deux libellés centrés ne doivent jamais se recouvrir (6,2 px/caractère)."""
        for (x1, t1), (x2, t2) in zip(labels, labels[1:]):
            self.assertGreaterEqual(x2 - x1, (len(t1) + len(t2)) / 2 * 5.5, (t1, t2))

    def test_libelles_tronques_et_jamais_chevauches(self):
        svg = charts.grouped_bars(charger("chart_edge.json"))   # 3 groupes dont un très long
        self.assertIn("…", svg)
        labels = self._labels_x(svg)
        self.assertEqual(len(labels), 3)
        self._verifie_sans_chevauchement(labels)

    def test_trop_de_barres_pour_des_libelles(self):
        # 120 barres serrées : les libellés sont sautés plutôt que superposés
        svg = charts.bars(charger("chart_edge.json"))
        self.assertTrue(rects(svg))
        self._verifie_sans_chevauchement(self._labels_x(svg))
        # cas intermédiaire (12 séries) : libellés tronqués mais bien présents
        data = donnees(valeurs_par_serie=[[float(i)] * 3 for i in range(12)],
                       groupes=("Un libellé de groupe très long", "court", "moyen"))
        svg_moyen = charts.bars(data)
        self.assertIn("…", svg_moyen)
        labels = self._labels_x(svg_moyen)
        self.assertGreater(len(labels), 1)
        self._verifie_sans_chevauchement(labels)


# ---------------------------------------------------------------------------
# 6. Graphique de durées
# ---------------------------------------------------------------------------

class TestDuree(unittest.TestCase):

    def setUp(self):
        self.data = charger("chart_speed.json")
        self.svg = charts.duration_chart(self.data)

    def test_utilise_duration_s(self):
        attendues = sum(1 for s in self.data["series"] for p in s["points"]
                        if isinstance(p.get("duration_s"), (int, float)))
        self.assertEqual(len(rects(self.svg)), attendues)

    def test_axe_en_secondes_lisibles(self):
        self.assertIn("Temps passé par test (s)", textes(self.svg))
        self.assertTrue(any(t.endswith("min") or t.endswith("s") for t in textes(self.svg)))

    def test_tooltip_duree_lisible(self):
        tips = titres(self.svg)
        self.assertTrue(any("Durée : 2 min 32 s" in t for t in tips), tips[:2])   # 152,4 s
        self.assertTrue(any("Durée : 7 min 39 s" in t for t in tips))            # 459,1 s
        self.assertTrue(any("Valeur : 152,4 s" in t for t in tips))

    def test_valeurs_non_numeriques_ignorees(self):
        svg = charts.duration_chart(charger("chart_weird.json"))
        # seules les durées chiffrées comptent : 755 + 0 (série 1) et 3900 (série 2)
        self.assertEqual(len(rects(svg)), 3)
        self.assertFalse(any("n/d" in t for t in titres(svg)))

    def test_aucune_duree(self):
        data = donnees(valeurs_par_serie=[[1.0, 2.0, 3.0]])
        for p in data["series"][0]["points"]:
            p["duration_s"] = None
        svg = charts.duration_chart(data)
        self.assertIn("aucune durée enregistrée", svg)
        racine(svg)                                # XML tout de même valide


# ---------------------------------------------------------------------------
# 7. Légende HTML
# ---------------------------------------------------------------------------

class TestLegende(unittest.TestCase):

    def test_pastille_libelle_et_compteur(self):
        html = charts.legend(charger("chart_speed.json"))
        self.assertIn("chart-legend", html)
        self.assertEqual(html.count("chart-legend-item"), 3)
        self.assertIn(charts.PALETTE[0], html)
        self.assertIn("Qwen3.6-35B · ctx 128k", html)
        self.assertIn("4 points", html)            # série 1 : 4 valeurs chiffrées sur 5
        self.assertIn('<input type="checkbox"', html)

    def test_masquee_cochee_et_signalee(self):
        html = charts.legend(charger("chart_speed.json"))
        bloc = [b for b in html.split("<label") if "qwen35-9b-128k" in b][0]
        self.assertNotIn(" checked", bloc.split("style=")[0])
        self.assertIn("masquée", bloc)
        self.assertIn("opacity:0.5", bloc)

    def test_lisible_sans_js(self):
        for bloc in (charts.legend(charger("chart_speed.json")),
                     charts.legend(charger("chart_edge.json")),
                     charts.legend(charger("chart_weird.json"))):
            self.assertNotIn("<script", bloc)
            self.assertNotIn("onclick", bloc)
            self.assertNotIn("onchange", bloc)
            self.assertIn("background:", bloc)
            self.assertIn("point", bloc)

    def test_40_series_listees_et_couleurs_cyclees(self):
        html = charts.legend(charger("chart_edge.json"))
        self.assertEqual(html.count("chart-legend-item"), 40)
        for couleur in charts.PALETTE:
            self.assertIn(couleur, html)

    def test_vide(self):
        self.assertEqual(charts.legend(charger("chart_empty.json")), "")
        self.assertEqual(charts.legend(None), "")
        self.assertEqual(charts.legend({"series": None}), "")


# ---------------------------------------------------------------------------
# 8. Échappement HTML
# ---------------------------------------------------------------------------

class TestEchappement(unittest.TestCase):

    def setUp(self):
        self.data = charger("chart_weird.json")

    def test_svg_echappe_le_script(self):
        for svg in (charts.grouped_bars(self.data), charts.bars(self.data),
                    charts.duration_chart(self.data)):
            self.assertNotIn("<script", svg)
            self.assertIn("&lt;script&gt;", svg)
            self.assertIn("&amp;", svg)
            racine(svg)                            # toujours du XML valide

    def test_texte_retrouve_apres_parsing(self):
        tips = titres(charts.grouped_bars(self.data))
        self.assertTrue(any("<script>alert('xss')</script>" in t for t in tips))
        labelles = textes(charts.grouped_bars(self.data))
        self.assertTrue(any("« résume & compare »" in t for t in labelles))

    def test_legende_echappee(self):
        html = charts.legend(self.data)
        self.assertNotIn("<script", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&amp;", html)
        self.assertIn("&quot;", html)

    def test_unite_et_metric_echappes(self):
        svg = charts.grouped_bars(self.data)
        self.assertIn("&lt;tâches FR&gt;", svg)
        self.assertIn("%&lt;&amp;&gt;", svg)

    def test_attributs_echappes(self):
        svg = charts.grouped_bars(self.data)
        self.assertIn('data-serie="s&lt;1&gt;"', svg)


# ---------------------------------------------------------------------------
# 9. Robustesse : jamais d'exception, jamais de NaN
# ---------------------------------------------------------------------------

class TestRobustesse(unittest.TestCase):

    def test_donnees_vides_ou_absurdes(self):
        cas = [
            None, "", 0, [], {}, {"groups": [], "series": []},
            {"groups": None, "series": None},
            {"metric": "pas un dict", "groups": "pas une liste", "series": 3},
            {"groups": [{"key": "g"}], "series": [{"points": "pas une liste"}]},
            {"groups": [{"label": "sans clé", "key": None}],
             "series": [{"label": "s", "points": [{"group": None, "value": 1}]}]},
            {"groups": [{"key": "g", "label": "G"}], "series": [{"points": [None, 3, "x"]}]},
            {"groups": [{}], "series": [{}]},
        ]
        for data in cas:
            with self.subTest(data=data):
                for fonction in (charts.grouped_bars, charts.bars, charts.duration_chart):
                    rendu = fonction(data)
                    self.assertIsInstance(rendu, str)
                    if rendu:
                        racine(rendu)
                self.assertIsInstance(charts.legend(data), str)

    def test_donnees_vides_sans_cle_metrique(self):
        data = {"groups": [{"key": "g", "label": "G"}],
                "series": [{"key": "s", "label": "S", "points": [{"group": "g", "value": 5}]}]}
        svg = charts.grouped_bars(data)
        racine(svg)
        self.assertIn("Valeur", textes(svg))       # libellé par défaut, unité absente
        self.assertIn("5", textes(svg))

    def test_sans_groupe_mais_avec_points(self):
        data = {"metric": {"key": "k", "label": "L", "unit": "", "better": "high"},
                "series": [{"key": "s", "label": "S", "points": [
                    {"group": "test-a", "value": 1.5, "duration_s": 10},
                    {"group": "test-b", "value": 2.5, "duration_s": 20}]}]}
        svg = charts.grouped_bars(data)
        self.assertEqual(len(rects(svg)), 2)
        self.assertEqual(len(titres(svg)), 2)

    def test_points_sans_valeur(self):
        data = donnees(valeurs_par_serie=[[None, None]])
        svg = charts.bars(data)
        self.assertIn("aucune donnée chiffrée", svg)
        racine(svg)

    def test_un_seul_point(self):
        data = donnees(valeurs_par_serie=[[42.0]], groupes=("seul",))
        for svg in (charts.grouped_bars(data), charts.bars(data), charts.duration_chart(data)):
            self.assertEqual(len(rects(svg)), 1)
            self.assertEqual(len(titres(svg)), 1)
            racine(svg)

    def test_zero_pur(self):
        data = donnees(valeurs_par_serie=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        svg = charts.grouped_bars(data)
        self.assertEqual(len(rects(svg)), 6)                 # les barres nulles existent
        for r in rects(svg):
            self.assertAlmostEqual(float(r.get("height")), 1.0, delta=0.01)
        self.assertNotIn("nan", svg.lower())

    def test_valeurs_negatives(self):
        data = donnees(valeurs_par_serie=[[-10.0, 20.0, -3.5]])
        svg = charts.grouped_bars(data)
        barres = rects(svg)
        self.assertEqual(len(barres), 3)
        neg, pos, neg2 = barres
        # la barre négative pend sous l'axe du zéro, la positive est au-dessus
        self.assertGreaterEqual(float(neg.get("y")), float(pos.get("y")) + float(pos.get("height")))
        self.assertIn("0", textes(svg))                      # graduation du zéro
        self.assertIn("−", svg) if "−" in svg else self.assertIn("-10", svg.replace(",", ""))

    def test_valeurs_non_finies(self):
        data = donnees(valeurs_par_serie=[[float("inf"), float("nan"), -float("inf")]])
        data["series"][0]["points"][0]["duration_s"] = float("inf")
        svg = charts.grouped_bars(data)
        self.assertFalse(rects(svg))                         # tout est ignoré proprement
        self.assertIn("aucune donnée chiffrée", svg)
        self.assertNotIn("nan", svg.lower())
        self.assertNotIn("inf", svg.lower())

    def test_type_exotique_dans_les_points(self):
        data = {"metric": {"key": "k", "label": "L", "unit": "u", "better": "low"},
                "groups": [{"key": "g", "label": "G"}, {"key": "g2", "label": "G2"}],
                "series": [{"key": "s", "label": "S", "hidden": "oui", "points": [
                    {"group": "g", "value": [1, 2], "duration_s": {"a": 1}},
                    {"group": "g2", "value": 7.0},
                    "pas un dict",
                ]}]}
        svg = charts.grouped_bars(data)
        self.assertEqual(len(rects(svg)), 1)          # seule la valeur 7.0 est chiffrable
        racine(svg)

    def test_libelles_tres_longs_tronques(self):
        long = "Groupe " + "x" * 300
        data = donnees(valeurs_par_serie=[[1.0, 2.0, 3.0]], groupes=(long, "court", "moyen"))
        svg = charts.grouped_bars(data)
        self.assertIn("…", svg)
        for texte in textes(svg):
            self.assertLess(len(texte), 60)
        # le libellé complet reste accessible dans le <title> (texte et barre)
        self.assertTrue(any(long.upper() in t for t in titres(svg)))

    def test_libelle_de_serie_tres_long(self):
        data = donnees(valeurs_par_serie=[[1.0, 2.0, 3.0]])
        data["series"][0]["label"] = "Série " + "y" * 400
        svg = charts.grouped_bars(data)
        racine(svg)
        self.assertTrue(any("y" * 400 in t for t in titres(svg)))

    def test_40_series_et_plus(self):
        data = charger("chart_edge.json")
        svg = charts.grouped_bars(data)
        racine(svg)
        self.assertTrue(0 < len(rects(svg)) <= 120)
        # un <title> de barre par barre (+ ceux des libellés d'axe tronqués)
        self.assertEqual(len([t for t in titres(svg) if t.startswith("Série :")]), len(rects(svg)))
        # les 40 séries restent dans l'axe des groupes (3 groupes) et la légende
        self.assertEqual(charts.legend(data).count("chart-legend-item"), 40)

    def test_200_series_ne_cassent_rien(self):
        beaucoup = [[float(i), float(i * 2), None] for i in range(200)]
        data = donnees(valeurs_par_serie=beaucoup)
        svg = charts.grouped_bars(data)
        racine(svg)
        self.assertLessEqual(len(rects(svg)), 600)
        self.assertEqual(len([t for t in titres(svg) if t.startswith("Série :")]), len(rects(svg)))

    def test_1000_groupes(self):
        data = donnees(valeurs_par_serie=[[float(i) for i in range(1000)]],
                       groupes=tuple(f"g{i}" for i in range(1000)))
        svg = charts.bars(data)
        racine(svg)
        self.assertTrue(rects(svg))

    def test_taille_minimale(self):
        data = donnees(valeurs_par_serie=[[1.0, 2.0]])
        for svg in (charts.grouped_bars(data, width=10, height=10),
                    charts.bars(data, width=1, height=1)):
            racine(svg)
            self.assertIn("viewBox", svg)

    def test_duree_negative_et_zero(self):
        data = donnees(valeurs_par_serie=[[1.0, 2.0, 3.0]])
        for i, dur in enumerate([-5, 0, 0.4]):
            data["series"][0]["points"][i]["duration_s"] = dur
        svg = charts.duration_chart(data)
        racine(svg)
        self.assertEqual(len(rects(svg)), 3)


# ---------------------------------------------------------------------------
# 10. Déterminisme
# ---------------------------------------------------------------------------

class TestDeterminisme(unittest.TestCase):

    def test_deux_appels_identiques(self):
        for nom in ("chart_speed.json", "chart_weird.json", "chart_edge.json"):
            data = charger(nom)
            for fonction in (charts.grouped_bars, charts.bars, charts.duration_chart):
                self.assertEqual(fonction(data), fonction(data), (nom, fonction))
                self.assertEqual(fonction(json.loads(json.dumps(data))),
                                 fonction(json.loads(json.dumps(data))))
            self.assertEqual(charts.legend(data), charts.legend(json.loads(json.dumps(data))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
