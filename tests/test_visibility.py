#!/usr/bin/env python3
"""Tests unitaires de `visibility.py` (unittest, stdlib uniquement).

On insiste sur ce qui casse en production : etat absent ou corrompu, ecritures
concurrentes (l'app est multi-threads) et non-mutation des entrees recues.
"""
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import visibility  # noqa: E402


def ent(source, **kwargs):
    e = {"kind": "speed", "source": source, "date": "2026-01-01T00:00:00",
         "model": "Qwen3.6-35B-A3B UD-IQ4_NL", "label": "35b_128k", "ctx": 131072}
    e.update(kwargs)
    return e


class TestEtatPersiste(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "visibility.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_chemin_par_defaut_dans_le_dossier_du_module(self):
        self.assertEqual(os.path.dirname(visibility.DEFAULT_PATH),
                         os.path.dirname(os.path.abspath(visibility.__file__)))
        self.assertTrue(visibility.DEFAULT_PATH.endswith("visibility.json"))

    def test_fichier_absent_donne_un_etat_vide(self):
        etat = visibility.load(self.path)
        self.assertEqual(etat, {"hidden": [], "deleted": [],
                                "filters": {k: [] for k in visibility.FILTER_KEYS}})

    def test_fichier_illisible_donne_un_etat_vide(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("{ceci n'est pas du json")
        self.assertEqual(visibility.load(self.path)["hidden"], [])
        with open(self.path, "wb") as fh:
            fh.write(b"\xff\xfe\x00binaire")
        self.assertEqual(visibility.load(self.path)["deleted"], [])
        os.makedirs(os.path.join(self.tmp.name, "dossier"), exist_ok=True)
        self.assertEqual(
            visibility.load(os.path.join(self.tmp.name, "dossier"))["hidden"], [])

    def test_aller_retour_save_load(self):
        etat = visibility.load(self.path)
        visibility.hide(etat, ["speed:a"])
        visibility.delete(etat, ["speed:b"])
        visibility.set_filter(etat, "model", ["Qwen3.6-35B"])
        visibility.save(etat, self.path)
        relu = visibility.load(self.path)
        self.assertEqual(relu["hidden"], ["speed:a"])
        self.assertEqual(relu["deleted"], ["speed:b"])
        self.assertEqual(relu["filters"]["models"], ["Qwen3.6-35B"])

    def test_ecriture_atomique_ne_laisse_pas_de_temporaire(self):
        etat = visibility.hide(visibility.load(self.path), ["speed:a"])
        visibility.save(etat, self.path)
        restes = [f for f in os.listdir(self.tmp.name) if f != "visibility.json"]
        self.assertEqual(restes, [])
        self.assertTrue(os.path.exists(self.path))

    def test_save_cree_le_dossier(self):
        cible = os.path.join(self.tmp.name, "sous", "dossier", "visibility.json")
        visibility.save(visibility.hide(visibility.load(), ["speed:a"]), cible)
        self.assertTrue(os.path.exists(cible))

    def test_save_ne_leve_pas_si_le_chemin_est_impossible(self):
        cible = os.path.join(self.tmp.name, "fichier", "visibility.json")
        with open(os.path.join(self.tmp.name, "fichier"), "w", encoding="utf-8") as fh:
            fh.write("x")
        visibility.save({"hidden": ["a"]}, cible)   # ne doit pas lever

    def test_save_concurrent_est_atomique(self):
        def ecrire(i):
            etat = visibility.load(self.path)
            visibility.hide(etat, [f"id-{i}"])
            visibility.save(etat, self.path)

        threads = [threading.Thread(target=ecrire, args=(i,)) for i in range(24)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with open(self.path, encoding="utf-8") as fh:
            data = json.load(fh)          # le fichier final est du JSON valide
        self.assertIn("hidden", data)
        restes = [f for f in os.listdir(self.tmp.name) if f != "visibility.json"]
        self.assertEqual(restes, [], "fichier temporaire laisse derriere")

    def test_normalisation_de_l_etat(self):
        etat = visibility.normalize_state(
            {"hidden": "a", "deleted": ["b", "b", ""], "filters": {"model": "M",
                                                                   "bidon": ["x"]}})
        self.assertEqual(etat["hidden"], ["a"])
        self.assertEqual(etat["deleted"], ["b"])
        self.assertEqual(etat["filters"]["models"], ["M"])
        self.assertEqual(set(etat["filters"]), set(visibility.FILTER_KEYS))
        self.assertEqual(visibility.normalize_state(None), visibility.empty_state())


class TestMasquageEtTombstones(unittest.TestCase):

    def setUp(self):
        self.etat = visibility.empty_state()

    def test_hide_unhide(self):
        visibility.hide(self.etat, ["speed:a"])
        self.assertTrue(visibility.is_hidden(self.etat, "speed:a"))
        self.assertFalse(visibility.is_deleted(self.etat, "speed:a"))
        visibility.unhide(self.etat, ["speed:a"])
        self.assertFalse(visibility.is_hidden(self.etat, "speed:a"))

    def test_delete_et_restore(self):
        visibility.delete(self.etat, ["speed:a"])
        self.assertTrue(visibility.is_deleted(self.etat, "speed:a"))
        self.assertTrue(visibility.is_hidden(self.etat, "speed:a"),
                        "un supprime ne s'affiche pas non plus")
        self.assertNotIn("speed:a", self.etat["hidden"])
        visibility.restore(self.etat, ["speed:a"])
        self.assertFalse(visibility.is_deleted(self.etat, "speed:a"))
        self.assertFalse(visibility.is_hidden(self.etat, "speed:a"))

    def test_delete_retire_le_masquage(self):
        visibility.hide(self.etat, ["speed:a"])
        visibility.delete(self.etat, ["speed:a"])
        self.assertEqual(self.etat["hidden"], [])
        self.assertEqual(self.etat["deleted"], ["speed:a"])

    def test_pas_de_doublon_et_accepte_une_chaine(self):
        visibility.hide(self.etat, "speed:a")
        visibility.hide(self.etat, ["speed:a", "speed:b"])
        self.assertEqual(self.etat["hidden"], ["speed:a", "speed:b"])
        visibility.hide(self.etat, None)
        self.assertEqual(self.etat["hidden"], ["speed:a", "speed:b"])

    def test_independance_des_entrees(self):
        visibility.hide(self.etat, ["speed:a"])
        self.assertFalse(visibility.is_hidden(self.etat, "speed:b"))
        self.assertFalse(visibility.is_hidden(self.etat, "speed:a:"))
        self.assertFalse(visibility.is_hidden(None, "speed:a"))


class TestApply(unittest.TestCase):

    def setUp(self):
        self.entrees = [
            ent("a.log", kind="speed"),
            ent("b.log", kind="humaneval", model="Qwen3.8-27B", config="draft MTP"),
            ent("c.log", kind="duel", model="Qwen3.8-27B", ctx=262144),
        ]

    def test_retire_masques_et_supprimes(self):
        etat = visibility.empty_state()
        visibility.hide(etat, [visibility.entry_id(self.entrees[0])])
        visibility.delete(etat, [visibility.entry_id(self.entrees[2])])
        gardees = visibility.apply(etat, self.entrees)
        self.assertEqual([e["source"] for e in gardees], ["b.log"])

    def test_ne_modifie_pas_les_entrees_recues(self):
        etat = visibility.empty_state()
        visibility.hide(etat, [visibility.entry_id(self.entrees[0])])
        import copy
        avant = copy.deepcopy(self.entrees)
        gardees = visibility.apply(etat, self.entrees)
        self.assertEqual(self.entrees, avant)
        self.assertIsNot(gardees[0], self.entrees[1], "copie, pas la meme reference")
        gardees[0]["model"] = "modifie"
        self.assertEqual(self.entrees[1]["model"], "Qwen3.8-27B")

    def test_ordre_conserve(self):
        gardees = visibility.apply(visibility.empty_state(), self.entrees)
        self.assertEqual([e["source"] for e in gardees],
                         [e["source"] for e in self.entrees])

    def test_filtres_par_kind_model_variante_serie(self):
        etat = visibility.empty_state()
        visibility.set_filter(etat, "kind", ["humaneval"])
        self.assertEqual([e["source"] for e in visibility.apply(etat, self.entrees)],
                         ["b.log"])
        etat = visibility.empty_state()
        visibility.set_filter(etat, "model", ["qwen3.8-27b"])
        self.assertEqual([e["source"] for e in visibility.apply(etat, self.entrees)],
                         ["b.log", "c.log"])
        etat = visibility.empty_state()
        visibility.set_filter(etat, "series", [visibility.variant_key(self.entrees[0])])
        gardees = visibility.apply(etat, self.entrees)
        self.assertEqual([e["source"] for e in gardees], ["a.log"])

    def test_filtres_vides_ne_filtrent_rien(self):
        self.assertEqual(len(visibility.apply(visibility.empty_state(), self.entrees)),
                         3)
        etat = visibility.empty_state()
        visibility.set_filter(etat, "series", [])
        self.assertEqual(len(visibility.apply(etat, self.entrees)), 3)

    def test_filtres_et_masquage_se_cumulent(self):
        etat = visibility.empty_state()
        visibility.set_filter(etat, "model", ["Qwen3.8-27B"])
        visibility.hide(etat, [visibility.entry_id(self.entrees[1])])
        self.assertEqual([e["source"] for e in visibility.apply(etat, self.entrees)],
                         ["c.log"])

    def test_entrees_sans_identifiant_connu(self):
        etat = visibility.empty_state()
        visibility.hide(etat, ["speed:sans-source"])
        entrees = [{"kind": "speed"}, None, "x", {"kind": "duel", "source": "d"}]
        gardees = visibility.apply(etat, entrees)
        self.assertEqual(len(gardees), 2)
        self.assertTrue(all(isinstance(e, dict) for e in gardees))

    def test_apply_sur_entrees_deja_normalisees(self):
        import metrics
        norm = metrics.normalize_entries(self.entrees)
        etat = visibility.empty_state()
        visibility.hide(etat, [norm[0]["entry_id"]])
        gardees = visibility.apply(etat, norm)
        self.assertEqual([e["source"] for e in gardees], ["b.log", "c.log"])

    def test_etat_vide_ou_invalide(self):
        self.assertEqual(len(visibility.apply(None, self.entrees)), 3)
        self.assertEqual(visibility.apply({}, self.entrees), self.entrees)


if __name__ == "__main__":
    unittest.main()
