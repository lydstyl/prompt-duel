#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_mesures_honnetes.py — trois défauts de mesure corrigés le 30/09.

Chaque test part d'un cas RÉEL rencontré dans le vault, reproduit en miniature :

  1. pic VRAM : le sampler nvidia-smi continue d'écrire après l'arrêt du cas ; publier
     son maximum surestimait le pic (12 118 MiB publiés là où la mesure du cas est
     8 940). Le pic doit être borné à la fenêtre `start … stop` des jalons de phase, et
     la carte servante désignée par son amplitude DANS la fenêtre (une donnée parasite
     hors fenêtre ne doit pas la désigner à tort).
  2. duel partiel : un duel qui a perdu des prompts (serveur injoignable en cours de
     route) était publié comme un duel ordinaire, avec un tok/s calculé sur les seuls
     prompts survivants. Il doit être marqué `partiel` et écarté des graphiques.
  3. miroir : les duels rangés dans un dossier `echecs/` du vault (campagne close) ne
     doivent pas être recopiés dans `duels/` par `mirror_duels`.

Plus : `wait_for_free_slot` face à un serveur sans `/slots` (OpenAI-only) et face à un
serveur momentanément injoignable.

Python 3 stdlib uniquement. Aucune écriture hors d'un dossier temporaire.
"""
import importlib
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bench_index  # noqa: E402
import metrics      # noqa: E402


# ------------------------------------------------------------------ VRAM

def ecrire_csv_vram(dossier, label, lignes):
    """Ecrit `vram-<label>.csv` au format du sampler.

    L'epoch du vrai sampler est ecrit a la francaise (virgule decimale) quand LC_ALL
    n'est pas force : la ligne fait alors 9 champs et `_norm_row` la realigne en
    fusionnant l'epoch et sa partie decimale.
    """
    chemin = Path(dossier) / f"vram-{label}.csv"
    texte = ["iso_time,epoch,gpu,mem_used_mib,mem_total_mib,util_pct,power_w,temp_c"]
    for epoch, gpu, used, total in lignes:
        entiere, millis = divmod(int(round(epoch * 1000)), 1000)
        texte.append(f"2026-09-29T21:57:31+02:00,{entiere},{millis},{gpu},{used},{total}"
                     f",0,2.70,34")
    chemin.write_text("\n".join(texte) + "\n", encoding="utf-8")
    return str(chemin)


def ecrire_phases(dossier, label, jalons):
    chemin = Path(dossier) / f"vram-{label}.phases"
    chemin.write_text("".join(f"{t},{nom}\n" for t, nom in jalons), encoding="utf-8")
    return str(chemin)


class TestPicVramBorneALaFenetre(unittest.TestCase):
    """Défaut 1 : le pic publié est celui du CAS, pas celui du sampler orphelin."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dossier = self.tmp.name
        self.label = "bonsai2-ptq1-32k"
        self.sauvegarde = bench_index.LOG_SRC
        bench_index.LOG_SRC = self.dossier

    def tearDown(self):
        bench_index.LOG_SRC = self.sauvegarde
        self.tmp.cleanup()

    def _cas(self, gpu_index=0, **extra):
        case = {"n_ctx": 32768, "gpu": "NVIDIA GeForce RTX 4060 Ti", "started":
                "2026-09-29T21:57:31", "cmd": "-ngl 99 -c 32768"}
        if gpu_index is not None:
            case["gpu_index"] = gpu_index
        case.update(extra)
        (Path(self.dossier) / f"case-{self.label}.json").write_text(
            json.dumps(case), encoding="utf-8")

    def test_pic_borne_et_ecart_trace(self):
        self._cas()
        # GPU 0 : le cas (idle 500 -> 8 940 pendant le duel). GPU 1 : une donnée
        # parasite APRES l'arrêt (12 118) — c'est elle qui était publiée.
        lignes = [
            (1000.0, 0, 500, 16311), (1010.0, 0, 8200, 16311),
            (1020.0, 0, 8940, 16311), (1039.0, 0, 8820, 16311),
            (1000.0, 1, 200, 8188), (1039.0, 1, 210, 8188),
            (1050.0, 0, 12118, 16311), (1050.0, 1, 12120, 8188),
        ]
        chemin = ecrire_csv_vram(self.dossier, self.label, lignes)
        ecrire_phases(self.dossier, self.label, [
            (1000.0, "start"), (1001.0, "load"), (1005.0, "ready"),
            (1012.0, "ready_end"), (1012.5, "speed"), (1018.0, "speed_end"),
            (1018.5, "duel"), (1039.0, "duel_end"), (1039.5, "fill"),
            (1040.0, "fill_end"), (1040.5, "he"), (1045.0, "he_end"), (1045.5, "stop")])
        entree = bench_index.parse_vram_csv(chemin)
        self.assertIsNotNone(entree)
        self.assertEqual(entree["peak_mib"], 8940)
        self.assertEqual(entree["peak_hors_cas_mib"], 12118)
        self.assertIn("hors cas", entree["note"] or "",
                      "l'écart doit être tracé dans la note du cas")

    def test_carte_servante_choisie_dans_la_fenetre(self):
        """Sans `gpu_index` : l'amplitude hors fenêtre ne doit pas désigner la carte."""
        self._cas(gpu_index=None)
        lignes = [
            (1000.0, 0, 500, 16311), (1010.0, 0, 8200, 16311),
            (1030.0, 0, 8940, 16311),
            (1000.0, 1, 200, 8188), (1030.0, 1, 205, 8188),
            (1060.0, 1, 12000, 8188),      # parasite, hors fenêtre
        ]
        chemin = ecrire_csv_vram(self.dossier, self.label, lignes)
        ecrire_phases(self.dossier, self.label, [
            (1000.0, "start"), (1005.0, "ready"), (1012.0, "ready_end"),
            (1012.5, "duel"), (1035.0, "duel_end"), (1035.5, "stop")])
        entree = bench_index.parse_vram_csv(chemin)
        self.assertEqual(entree["peak_mib"], 8940,
                         "le pic doit rester celui de la carte servante, dans la fenêtre")
        self.assertNotEqual(entree["peak_mib"], 12000,
                            "le parasite hors fenêtre de l'autre carte ne doit pas gagner")


# ------------------------------------------------------- duels partiels

def duel_json(nb_ok, nb_total, run_id="2026-09-30_1423__modele__duel"):
    res = []
    for i in range(nb_total):
        res.append({"status": "ok" if i < nb_ok else "error", "duree_s": 10.0,
                    "completion_tokens": 200})
    return {"run_id": run_id, "started_at": "2026-09-30T14:23:00",
            "model_slug": "qwen3.8-flash-next-iq3_xxs", "n_ctx": 131072,
            "params": {"temperature": 0.2, "max_tokens": 4096,
                       "enable_thinking": False},
            "results": res}


class TestDuelPartielEcartéDesGraphiques(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self.tmp.name) / "runs"
        self.sauvegarde = bench_index.RUNS_DIR
        bench_index.RUNS_DIR = str(self.runs)

    def tearDown(self):
        bench_index.RUNS_DIR = self.sauvegarde
        self.tmp.cleanup()

    def _run(self, run_id, nb_ok, nb_total):
        d = self.runs / run_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.json").write_text(json.dumps(duel_json(nb_ok, nb_total, run_id)),
                                    encoding="utf-8")

    def test_duel_perdu_est_marque_partiel(self):
        self._run("2026-09-30_1423__duel-casse", 5, 34)
        self._run("2026-09-30_1559__duel-propre", 7, 7)
        entrees = {e["run_id"]: e for e in bench_index.scan_duels()}
        casse = entrees["2026-09-30_1423__duel-casse"]
        propre = entrees["2026-09-30_1559__duel-propre"]
        self.assertTrue(casse["partiel"])
        self.assertEqual(casse["status"], "partiel")
        self.assertEqual((casse["prompts_ok"], casse["prompts_failed"]), (5, 29))
        self.assertFalse(propre["partiel"])
        self.assertEqual(propre["status"], "ok")

    def test_seuil_partiel_a_75_pour_cent(self):
        """Le seuil est une décision de mesure : pertes = bruit vs duel à relancer."""
        self.assertEqual(bench_index.DUEL_OK_RATIO, 0.75)
        self._run("2026-09-29_2200__seuil-bas", 25, 34)     # 73,5 % -> partiel
        self._run("2026-09-29_2300__seuil-haut", 26, 34)    # 76,5 % -> comparable
        entrees = {e["run_id"]: e for e in bench_index.scan_duels()}
        self.assertTrue(entrees["2026-09-29_2200__seuil-bas"]["partiel"])
        self.assertFalse(entrees["2026-09-29_2300__seuil-haut"]["partiel"])

    def test_partiels_hors_courbes_par_defaut(self):
        self._run("2026-09-30_1423__duel-casse", 5, 34)
        self._run("2026-09-30_1559__duel-propre", 7, 7)
        entrees = bench_index.scan_duels()
        data = metrics.series_for_chart(entrees, metric_key="tok_s")
        self.assertEqual(data["partiels_exclus"], 1)
        points = [p for s in data["series"] for p in s["points"]]
        self.assertEqual(len(points), 1, "un seul duel comparable doit être tracé")
        avec = metrics.series_for_chart(entrees, metric_key="tok_s", include_partial=True)
        points_avec = [p for s in avec["series"] for p in s["points"]]
        self.assertEqual(len(points_avec), 2)
        self.assertEqual(avec["partiels_exclus"], 0)


class TestMiroirIgnoreLesEchecs(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        racine = Path(self.tmp.name)
        self.runs = racine / "runs"
        self.vault = racine / "vault"
        self.duels = self.vault / "documents" / "llm-benchmarks" / "duels"
        self.duels.mkdir(parents=True)
        self.sauvegarde = (bench_index.RUNS_DIR, bench_index.VAULT_DOCS)
        bench_index.RUNS_DIR = str(self.runs)
        bench_index.VAULT_DOCS = str(self.vault / "documents" / "llm-benchmarks")
        # un duel normal + un duel range dans echecs/
        for run_id in ("2026-09-30_1559__duel-propre", "2026-09-30_1423__duel-casse"):
            d = self.runs / run_id
            d.mkdir(parents=True)
            (d / "run.json").write_text(json.dumps(duel_json(7, 7, run_id)),
                                        encoding="utf-8")
        echecs = (Path(bench_index.VAULT_DOCS)
                  / "224-2026-09-30-campagne" / "echecs" / "2026-09-30_1423__duel-casse")
        echecs.mkdir(parents=True)
        (echecs / "run.json").write_text(json.dumps(duel_json(5, 34)), encoding="utf-8")

    def tearDown(self):
        bench_index.RUNS_DIR, bench_index.VAULT_DOCS = self.sauvegarde
        self.tmp.cleanup()

    def test_run_ids_ecartes(self):
        ecartes = bench_index.run_ids_ecartes()
        self.assertIn("2026-09-30_1423__duel-casse", ecartes)
        self.assertNotIn("2026-09-30_1559__duel-propre", ecartes)

    def test_mirror_ne_ressuscite_pas_un_echec(self):
        """`mirror_duels(root)` ecrit dans `root/duels/<run_id>/run.json`."""
        copies = bench_index.mirror_duels(str(bench_index.VAULT_DOCS), verbose=False)
        duels = Path(bench_index.VAULT_DOCS) / "duels"
        noms = sorted(p.name for p in duels.iterdir())
        self.assertIn("2026-09-30_1559__duel-propre", noms)
        self.assertNotIn("2026-09-30_1423__duel-casse", noms,
                         "un duel rangé dans echecs/ ne doit pas revenir dans duels/")
        self.assertEqual(copies, 1)


# --------------------------------------------------- /slots (Strata & co)

class TestSlotIndisponible(unittest.TestCase):
    """`wait_for_free_slot` ne doit pas coûter un prompt sur un serveur OpenAI-only."""

    def setUp(self):
        os.environ["LLM_DRY_RUN"] = "1"
        self.app = importlib.import_module("app")
        self.sauvegarde = (self.app.fetch_slot_busy, self.app.SLOTS_DISPONIBLE,
                           self.app.SLOT_ESSAIS, self.app.SLOT_ESSAI_S)
        self.app.SLOTS_DISPONIBLE = None
        self.app.SLOT_ESSAIS = 1
        self.app.SLOT_ESSAI_S = 0

    def tearDown(self):
        (self.app.fetch_slot_busy, self.app.SLOTS_DISPONIBLE,
         self.app.SLOT_ESSAIS, self.app.SLOT_ESSAI_S) = self.sauvegarde

    def test_slots_absent_404_continue(self):
        def absent():
            raise urllib.error.HTTPError("http://x/slots", 404, "Not Found", {}, None)
        self.app.fetch_slot_busy = absent
        self.premier = self.app.wait_for_free_slot()
        self.assertTrue(self.premier, "sans /slots, on continue (tirs séquentiels)")
        self.assertIs(self.app.SLOTS_DISPONIBLE, False)
        # deuxième appel : plus aucune requête, on ne re-sonde pas
        appels = {"n": 0}

        def compte():
            appels["n"] += 1
            raise AssertionError("ne doit pas être rappelé")
        self.app.fetch_slot_busy = compte
        self.assertTrue(self.app.wait_for_free_slot())
        self.assertEqual(appels["n"], 0)

    def test_serveur_injoignable_essais_puis_erreur(self):
        appels = {"n": 0}

        def injoignable():
            appels["n"] += 1
            raise urllib.error.URLError("[Errno 111] Connection refused")
        self.app.fetch_slot_busy = injoignable
        with self.assertRaises(RuntimeError):
            self.app.wait_for_free_slot()
        self.assertEqual(appels["n"], 2, "1 essai + 1 dépassement du quota")

    def test_slot_occupe_puis_libre(self):
        etats = [True, False]

        def occupe():
            return etats.pop(0)
        self.app.fetch_slot_busy = occupe
        self.app.SLOT_POLL_S = 0
        self.assertTrue(self.app.wait_for_free_slot())
        self.assertIs(self.app.SLOTS_DISPONIBLE, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
