#!/usr/bin/env python3
"""Tests unitaires de l'extension v1.6 de `bench_index.py` (unittest, stdlib).

Deux volets :
  * les benches de l'app (`benches/<run>/bench.json`) : ingestion d'un bench.json
    REEL (fixture) et metriques reelles reprises du fichier ;
  * une regeneration complete dans un dossier TEMPORAIRE (vault + runs + benches
    copiees depuis les fixtures) : compteurs coherents, identifiants stables,
    aucune ecriture dans le vault ni dans le depot.

Compatibilite ascendante : aucune cle existante ne doit disparaitre ou changer,
et `bench_index` doit continuer a fonctionner sans `metrics.py`.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bench_index  # noqa: E402
import metrics  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name):
    with open(FIXTURES / name, encoding="utf-8") as fh:
        return json.load(fh)


def make_run_json(run_id, **kwargs):
    run = {
        "run_id": run_id,
        "started_at": "2026-09-30T10:00:00",
        "model_slug": "test-35b",
        "model_path": "/models/test-35b.gguf",
        "n_ctx": 131072,
        "params": {"temperature": 0.2, "max_tokens": 8192, "enable_thinking": False},
        "results": [{"status": "ok", "duree_s": 1.5, "completion_tokens": 100},
                    {"status": "error", "duree_s": 0.5, "completion_tokens": 0}],
    }
    run.update(kwargs)
    return run


class TestBenchesDeLApp(unittest.TestCase):
    """Ingestion de `benches/<run>/bench.json` (fixture = fichier reel)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.benches = Path(self.tmp.name) / "benches"
        run_dir = self.benches / "2026-09-30_1146__qwen3-6-35b-a3b-ud-iq4-nl-256k__tests"
        os.makedirs(run_dir)
        shutil.copyfile(FIXTURES / "bench.json", run_dir / "bench.json")
        self.entries = bench_index.scan_app_benches(str(self.benches))

    def tearDown(self):
        self.tmp.cleanup()

    def test_un_resultat_par_entree(self):
        self.assertEqual(len(self.entries), 4)
        self.assertEqual({e["test_id"] for e in self.entries},
                         {"vitesse-court", "batterie-13", "memoire-128k", "humaneval-50"})

    def test_familles_reelles(self):
        kinds = {e["test_id"]: e["kind"] for e in self.entries}
        self.assertEqual(kinds, {"vitesse-court": "vitesse", "batterie-13": "intelligence",
                                 "memoire-128k": "memoire", "humaneval-50": "intelligence"})

    def test_metriques_reelles_du_bench_json(self):
        par_id = {e["test_id"]: e for e in self.entries}
        vitesse = par_id["vitesse-court"]
        self.assertEqual(vitesse["pp_tps"], 1158.2)
        self.assertEqual(vitesse["tg_tps"], 148.5)
        self.assertEqual(vitesse["duree_s"], 4.5)
        self.assertIn("Go", vitesse["vram_peak"])
        self.assertEqual(metrics.entry_metric_values(vitesse)["vram_peak"], 13.3)
        memoire = par_id["memoire-128k"]
        self.assertIs(memoire["needle"], True)
        self.assertEqual(memoire["tokens"], 119087)
        self.assertAlmostEqual(memoire["prefill_tps"], 1174.559213335399)
        self.assertEqual(memoire["duree_s"], 102.3)
        batterie = par_id["batterie-13"]
        self.assertEqual(batterie["score_avg"], 0.846)
        self.assertEqual(batterie["tasks"], 13)
        humaneval = par_id["humaneval-50"]
        self.assertEqual((humaneval["passed"], humaneval["total"]), (48, 50))
        self.assertEqual(humaneval["score_pct"], 96.0)

    def test_identifiants_uniques_et_stables(self):
        ids = [e["entry_id"] for e in self.entries]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertTrue(all(i.startswith(("vitesse:", "memoire:", "intelligence:"))
                            for i in ids))
        self.assertTrue(ids[0].endswith("#vitesse-court"))
        relus = bench_index.scan_app_benches(str(self.benches))
        self.assertEqual([e["entry_id"] for e in relus], ids)

    def test_variante_et_reglages(self):
        for entry in self.entries:
            self.assertIn("variant", entry)
            self.assertTrue(entry["variant"]["key"])
            self.assertEqual(entry["variant"]["ctx"], 262144)
            self.assertIn("ctx256k", entry["variant"]["key"])
            self.assertIn("mtp-on", entry["variant"]["key"],
                          "la note du run dit « draft MTP actif »")
            self.assertEqual(entry["settings"]["n_ctx"], 262144)
            self.assertIs(entry["settings"]["mtp"], True)

    def test_ordre_stable_et_resultats_dans_l_ordre_du_fichier(self):
        sources = [e["source"] for e in self.entries]
        self.assertEqual([s.split("#")[-1] for s in sources],
                         ["vitesse-court", "batterie-13", "memoire-128k", "humaneval-50"])
        relus = bench_index.scan_app_benches(str(self.benches))
        self.assertEqual([e["source"] for e in relus], sources)

    def test_series_pour_le_graphique(self):
        data = metrics.series_for_chart(self.entries, metric_key="pp_tps")
        self.assertEqual([g["key"] for g in data["groups"]], ["vitesse-court"])
        self.assertEqual(len(data["series"]), 1)
        point = data["series"][0]["points"][0]
        self.assertEqual(point["value"], 1158.2)
        self.assertEqual(point["duration_s"], 4.5)
        data = metrics.series_for_chart(self.entries, metric_key="needle")
        self.assertEqual([g["key"] for g in data["groups"]], ["memoire-128k"])
        self.assertEqual(data["series"][0]["points"][0]["value"], 1.0)

    def test_famille_deduite_de_l_identifiant(self):
        self.assertEqual(bench_index.app_bench_family({"id": "batterie-13"}), "intelligence")
        self.assertEqual(bench_index.app_bench_family({"id": "humaneval-50"}), "intelligence")
        self.assertEqual(bench_index.app_bench_family({"id": "vitesse-256k"}), "vitesse")
        self.assertEqual(bench_index.app_bench_family({"group": "memoire", "id": "x"}), "memoire")
        self.assertIsNone(bench_index.app_bench_family({"id": "bidon-1"}))

    def test_dossier_benches_absent(self):
        self.assertEqual(bench_index.scan_app_benches(str(Path(self.tmp.name) / "vide")), [])

    def test_bench_json_casse_est_ignore(self):
        casse = Path(self.tmp.name) / "benches2" / "run-x"
        os.makedirs(casse)
        (casse / "bench.json").write_text("{pas du json", encoding="utf-8")
        vide = Path(self.tmp.name) / "benches2" / "run-y"
        os.makedirs(vide)
        (vide / "bench.json").write_text("[]", encoding="utf-8")
        self.assertEqual(bench_index.scan_app_benches(str(Path(self.tmp.name) / "benches2")), [])

    def test_fonctionne_sans_metrics(self):
        """Compatibilite : `metrics.py` absent -> l'index reste utilisable."""
        sauvegarde = bench_index.metrics
        bench_index.metrics = None
        try:
            entries = bench_index.scan_app_benches(str(self.benches))
        finally:
            bench_index.metrics = sauvegarde
        self.assertEqual(len(entries), 4)
        self.assertNotIn("entry_id", entries[0])
        for cle in ("kind", "test_id", "source", "summary", "duree_s"):
            self.assertIn(cle, entries[0])


class TestRegenerationEnVaultTemporaire(unittest.TestCase):
    """Regeneration complete dans /tmp : vault, runs et benches copies."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.vault_root = root / "vault"
        self.docs = self.vault_root / "documents" / "llm-benchmarks"
        shutil.copytree(FIXTURES / "vault_min" / "documents" / "llm-benchmarks", self.docs)
        self.wiki = self.vault_root / "wiki" / "cas-developpeur" / "llm-benchmarks"
        self.wiki.mkdir(parents=True)
        (self.wiki / "note.md").write_text("# Note de test\ncontenu\n", encoding="utf-8")

        self.runs = root / "runs"
        run_dir = self.runs / "2026-09-30_1000__test-35b__ctx131072__t0.2__nothink"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps(make_run_json(run_dir.name)), encoding="utf-8")

        self.benches = root / "benches"
        bench_dir = self.benches / "2026-09-30_1146__tests"
        bench_dir.mkdir(parents=True)
        shutil.copyfile(FIXTURES / "bench.json", bench_dir / "bench.json")

        self.sauvegarde = {nom: getattr(bench_index, nom) for nom in
                           ("VAULT_ROOT", "VAULT_DOCS", "VAULT_WIKI", "RUNS_DIR",
                            "BENCHES_DIR")}
        bench_index.VAULT_ROOT = str(self.vault_root)
        bench_index.VAULT_DOCS = str(self.docs)
        bench_index.VAULT_WIKI = str(self.wiki)
        bench_index.RUNS_DIR = str(self.runs)
        bench_index.BENCHES_DIR = str(self.benches)
        self.local_index_avant = self._empreinte(bench_index.LOCAL_INDEX)
        self.fichiers_vault = self._fichiers(self.vault_root)

    def tearDown(self):
        for nom, valeur in self.sauvegarde.items():
            setattr(bench_index, nom, valeur)
        self.tmp.cleanup()

    @staticmethod
    def _fichiers(racine):
        out = set()
        for chemin, _, noms in os.walk(racine):
            for nom in noms:
                out.add(os.path.relpath(os.path.join(chemin, nom), racine))
        return out

    @staticmethod
    def _empreinte(chemin):
        if not os.path.exists(chemin):
            return None
        st = os.stat(chemin)
        return (st.st_size, st.st_mtime_ns)

    def test_build_sans_miroir_produit_toutes_les_familles(self):
        index = bench_index.build(mirror=False, verbose=False)
        counts = index["counts"]
        for kind in ("duel", "speed", "context", "battery", "humaneval",
                     "vitesse", "memoire", "intelligence"):
            self.assertIn(kind, counts, f"famille {kind} absente de l'index")
            self.assertGreater(counts[kind], 0, kind)
        self.assertEqual(set(counts), {e["kind"] for e in index["entries"]})
        self.assertEqual(counts["vitesse"], 1)
        self.assertEqual(counts["memoire"], 1)
        self.assertEqual(counts["intelligence"], 2)

    def test_compteurs_coherents(self):
        index = bench_index.build(mirror=False, verbose=False)
        entries = index["entries"]
        self.assertEqual(sum(index["counts"].values()), len(entries))
        for kind, total in index["counts"].items():
            self.assertEqual(total, sum(1 for e in entries if e["kind"] == kind))

    def test_toutes_les_entrees_sont_enrichies(self):
        index = bench_index.build(mirror=False, verbose=False)
        ids = [e["entry_id"] for e in index["entries"]]
        self.assertEqual(len(set(ids)), len(ids), "entry_id en collision")
        for entry in index["entries"]:
            self.assertTrue(entry["entry_id"])
            self.assertTrue(entry["variant"]["key"])
            self.assertIn("metric_values", entry)

    def test_ordre_et_identifiants_stables_entre_deux_generations(self):
        premier = bench_index.build(mirror=False, verbose=False)
        second = bench_index.build(mirror=False, verbose=False)
        self.assertEqual([e["entry_id"] for e in premier["entries"]],
                         [e["entry_id"] for e in second["entries"]])

    def test_aucune_ecriture_dans_le_vault_temporaire(self):
        bench_index.build(mirror=False, verbose=False)
        self.assertEqual(self._fichiers(self.vault_root), self.fichiers_vault)
        self.assertEqual(self._empreinte(bench_index.LOCAL_INDEX),
                         self.local_index_avant,
                         "bench_index.json du depot ne doit pas bouger (build n'ecrit pas)")
        self.assertFalse(os.path.exists(self.docs / "index.json"))

    def test_cles_attendues_par_benchmarks_py_conservees(self):
        index = bench_index.build(mirror=False, verbose=False)
        par_famille = {}
        for entry in index["entries"]:
            par_famille.setdefault(entry["kind"], entry)
        for cle in ("date", "source", "kind", "model"):
            for kind, entry in par_famille.items():
                self.assertIn(cle, entry, f"{kind} : cle {cle} manquante")
        duel = par_famille["duel"]
        for cle in ("run_id", "run_url", "tok_s", "prompts_ok", "prompts_total",
                    "tokens_out", "duration_s", "temperature", "max_tokens", "thinking"):
            self.assertIn(cle, duel, f"duel : cle {cle} disparue")
        for cle in ("label", "pp", "tg", "method", "load_s"):
            self.assertIn(cle, par_famille["speed"], f"speed : cle {cle} disparue")
        for cle in ("label", "score_avg", "lat_median", "lat_max", "fails", "tasks"):
            self.assertIn(cle, par_famille["battery"], f"battery : cle {cle} disparue")
        for cle in ("score_pct", "passed", "total", "avg_latency_s", "wall_s",
                    "avg_tokens_per_sec", "config", "machine"):
            self.assertIn(cle, par_famille["humaneval"], f"humaneval : cle {cle} disparue")
        for cle in ("prefill_tps", "tokens", "needle", "peak_vram", "kv", "status"):
            self.assertIn(cle, par_famille["context"], f"context : cle {cle} disparue")

    def test_valeurs_reelles_du_vault_synthetique(self):
        index = bench_index.build(mirror=False, verbose=False)
        vitesse = next(e for e in index["entries"] if e["kind"] == "speed")
        self.assertEqual(vitesse["pp"], 620.0)          # moyenne de 600/620,5/640,5 arrondie
        self.assertEqual(vitesse["tg"], 41.1)
        self.assertEqual(vitesse["load_s"], 42)
        self.assertEqual(vitesse["ctx"], 131072)
        self.assertIn("Go", vitesse["vram"])
        memoire = next(e for e in index["entries"] if e["kind"] == "context")
        self.assertEqual(memoire["tokens"], 238115)
        self.assertIs(memoire["needle"], True)
        self.assertEqual(memoire["prefill_tps"], 904.2)
        self.assertEqual(memoire["elapsed_s"], 263.3)
        batterie = next(e for e in index["entries"] if e["kind"] == "battery")
        self.assertEqual(batterie["score_avg"], 0.846)
        self.assertEqual(batterie["fails"], ["T3_arithmetique"])
        humaneval = next(e for e in index["entries"] if e["kind"] == "humaneval")
        self.assertEqual(humaneval["total"], 50)
        self.assertEqual(humaneval["score_pct"], 96.0)
        self.assertIn("draft MTP", humaneval["config"])

    def test_series_sur_l_index_regeneré(self):
        index = bench_index.build(mirror=False, verbose=False)
        data = metrics.series_for_chart(index["entries"], metric_key="pp")
        self.assertTrue(data["series"])
        for serie in data["series"]:
            for point in serie["points"]:
                self.assertIsNotNone(point["duration_s"])
        data = metrics.series_for_chart(index["entries"], metric_key="score_pct")
        groupes = {g["key"] for g in data["groups"]}
        self.assertEqual(groupes, {"humaneval-50"})


class TestCompatIndexExistant(unittest.TestCase):

    def test_les_entrees_reelles_gardent_toutes_leurs_cles(self):
        entries = load_fixture("index_extract.json")["entries"]
        for brut, norm in zip(entries, metrics.normalize_entries(entries)):
            for cle, valeur in brut.items():
                self.assertIn(cle, norm)
                self.assertEqual(norm[cle], valeur, f"cle {cle} alteree")
            ajoutes = set(norm) - set(brut)
            self.assertTrue(ajoutes <= {"entry_id", "variant", "metric_values",
                                        "duration_s"},
                            f"champs inattendus : {ajoutes}")

    def test_les_champs_ajoutes_sont_ceux_prevus(self):
        entries = load_fixture("index_extract.json")["entries"]
        ajoutes = set(metrics.normalize_entries(entries)[0]) - set(entries[0])
        self.assertTrue({"entry_id", "variant", "metric_values"} <= ajoutes)

    def test_kinds_du_benchmarks_py_toujours_geres(self):
        entries = load_fixture("index_vault.json")["entries"]
        kinds = {e["kind"] for e in entries}
        self.assertEqual(kinds, {"duel", "speed", "context", "battery", "humaneval"})
        ids = [metrics.entry_id(e) for e in entries]
        self.assertEqual(len(set(ids)), 68)


if __name__ == "__main__":
    unittest.main()
