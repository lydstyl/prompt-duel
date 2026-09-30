#!/usr/bin/env python3
"""test_integration.py — câblage backend v1.6 (contrat §7-E1).

On lance une VRAIE instance de l'app (`app.py`) en LLM_DRY_RUN=1 sur un port
éphémère, avec :
  * RUNS_DIR / BENCH_DIR pointés sur un dossier temporaire,
  * BENCH_INDEX sur un index de test (jamais le vault, jamais l'index réel),
  * VISIBILITY_PATH sur un fichier temporaire,
et on interroge les routes HTTP réelles :

  - /api/state expose `settings` (détectés) et `variant` ;
  - /api/graph renvoie des graphiques + des comptes, sans jamais planter ;
  - POST /api/visibility masque (l'entrée disparaît de /api/graph) puis la
    restaure ; une action inconnue répond 400 ;
  - /graph répond 503 tant que page_graph.py n'est pas livré (200 sinon).

Python 3 stdlib uniquement (unittest, urllib). Jamais d'écriture dans le vault.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")

# Entrées d'index de test : une par famille principale, `source` unique.
TEST_ENTRIES = [
    {
        "kind": "speed", "source": "test/vitesse.log", "label": "27b_128k",
        "model": "Qwen3.8-27B", "ctx": 131072, "date": "2026-09-29T20:00:00",
        "pp": 657.0, "tg": 46.0, "vram": "12.4 / 12.8 Go", "load_s": 70,
        "method": "bench 128 tok", "status": "ok",
    },
    {
        "kind": "context", "source": "test/contexte.out", "label": "35b_256k",
        "model": "Qwen3.6-35B-A3B", "ctx": 262144, "kv": "q8_0", "status": "ok",
        "tokens": 238115, "needle": True, "prefill_tps": 904.2,
        "peak_vram": "10.6 Go", "date": "2026-09-29T20:00:00", "elapsed_s": 263.3,
    },
    {
        "kind": "humaneval", "source": "test/humaneval.json", "model": "Qwen3.8-27B",
        "passed": 48, "total": 50, "score_pct": 96.0, "avg_latency_s": 1.28,
        "avg_tokens_per_sec": 33.3, "wall_s": 67.1, "date": "2026-09-29T20:00:00",
    },
]


def _free_port():
    """Un port TCP libre sur la boucle locale."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


class IntegrationTestCase(unittest.TestCase):
    """Une instance réelle de l'app, des données temporaires, des appels HTTP."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="prompt-duel-itest.")
        cls.runs = os.path.join(cls.tmp, "runs")
        cls.benches = os.path.join(cls.tmp, "benches")
        cls.index = os.path.join(cls.tmp, "index.json")
        cls.visibility_path = os.path.join(cls.tmp, "visibility.json")
        os.makedirs(cls.runs, exist_ok=True)
        os.makedirs(cls.benches, exist_ok=True)
        cls.write_index(TEST_ENTRIES)

        cls.port = _free_port()
        env = dict(os.environ)
        env.update({
            "PORT": str(cls.port), "HOST": "127.0.0.1",
            "RUNS_DIR": cls.runs, "BENCH_DIR": cls.benches,
            "BENCH_INDEX": cls.index, "VISIBILITY_PATH": cls.visibility_path,
            "LLM_DRY_RUN": "1", "LLM_BASE": "http://127.0.0.1:9",
            "BENCH_HE_SSH": "e2e@127.0.0.1", "BENCH_LLM_HOME": cls.tmp,
            "APP_TITLE": "Prompt Duel (integration)",
        })
        cls.log_path = os.path.join(cls.tmp, "app.log")
        cls.log = open(cls.log_path, "w", encoding="utf-8")
        cls.proc = subprocess.Popen(
            [sys.executable, APP], cwd=ROOT, env=env,
            stdout=cls.log, stderr=subprocess.STDOUT)
        cls.base = f"http://127.0.0.1:{cls.port}"
        if not cls._wait_ready():
            cls.proc.terminate()
            cls.log.flush()
            with open(cls.log_path, encoding="utf-8") as fh:
                log = fh.read()[-3000:]
            raise AssertionError(f"l'app n'a pas répondu sur {cls.base}\n{log}")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.proc.terminate()
            cls.proc.wait(timeout=10)
        except Exception:
            try:
                cls.proc.kill()
            except Exception:
                pass
        try:
            cls.log.close()
        except Exception:
            pass

    # -- utilitaires --------------------------------------------------------

    @classmethod
    def write_index(cls, entries):
        """Réécrit l'index de test (lu à chaque requête /api/graph)."""
        with open(cls.index, "w", encoding="utf-8") as fh:
            json.dump({"generated_at": "test", "counts": {}, "entries": entries}, fh)

    @classmethod
    def _wait_ready(cls, timeout=25.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(cls.base + "/api/state", timeout=3) as r:
                    if r.status == 200:
                        return True
            except Exception:
                time.sleep(0.2)
        return False

    def get(self, path):
        """-> (status, body texte)."""
        try:
            with urllib.request.urlopen(self.base + path, timeout=15) as r:
                return r.status, r.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8")

    def get_json(self, path):
        status, body = self.get(path)
        return status, json.loads(body)

    def post_json(self, path, payload):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def entry_ids_from_graph(self):
        """Tous les entry_id présents dans les séries de /api/graph."""
        status, payload = self.get_json("/api/graph")
        self.assertEqual(status, 200)
        ids = []
        for chart in payload["charts"]:
            for series in chart["series"]:
                for point in series["points"]:
                    if point.get("entry_id"):
                        ids.append(point["entry_id"])
        return sorted(set(ids))

    def restore_all(self, ids):
        for entry_id in ids:
            self.post_json("/api/visibility", {"action": "restore", "ids": [entry_id]})

    # -- /api/state ---------------------------------------------------------

    def test_state_exposes_settings_and_variant(self):
        status, state = self.get_json("/api/state")
        self.assertEqual(status, 200)
        self.assertIn("settings", state)
        self.assertIsInstance(state["settings"], dict)
        self.assertIn("variant", state)
        self.assertIsInstance(state["variant"], dict)
        # la variante porte au moins une clé et un libellé (contrat §6)
        self.assertIn("key", state["variant"])
        self.assertIn("label", state["variant"])
        # réglages détectés : le port visé est bien exposé
        self.assertIn("source", state["settings"])

    # -- /api/graph ---------------------------------------------------------

    def test_graph_returns_charts_and_counts(self):
        status, payload = self.get_json("/api/graph")
        self.assertEqual(status, 200)
        for key in ("charts", "hidden", "counts"):
            self.assertIn(key, payload)
        self.assertIsInstance(payload["charts"], list)
        self.assertTrue(payload["charts"], "au moins un graphique attendu")
        counts = payload["counts"]
        self.assertEqual(counts["entries"], len(TEST_ENTRIES))
        self.assertEqual(counts["visible"], len(TEST_ENTRIES))
        # la série vitesse (pp) contient au moins un point chiffré
        speed = [c for c in payload["charts"]
                 if c["metric"]["key"] == "pp" and c["kind"] == "speed"]
        self.assertTrue(speed, "graphique vitesse (pp) absent")
        self.assertTrue(speed[0]["series"], "série vitesse vide")
        self.assertTrue(speed[0]["series"][0]["points"], "aucun point vitesse")

    def test_hide_removes_entry_from_graph_then_restore(self):
        ids = self.entry_ids_from_graph()
        self.assertTrue(ids, "aucun entry_id dans /api/graph")
        target = ids[0]
        try:
            # masquer -> l'entrée disparaît des vues
            status, resp = self.post_json("/api/visibility",
                                          {"action": "hide", "ids": [target]})
            self.assertEqual(status, 200)
            self.assertTrue(resp["ok"])
            self.assertIn(target, resp["state"]["hidden"])

            _, payload = self.get_json("/api/graph")
            self.assertIn(target, payload["hidden"])
            self.assertEqual(payload["counts"]["visible"], len(TEST_ENTRIES) - 1)
            self.assertNotIn(target, self.entry_ids_from_graph())
        finally:
            # restaurer -> l'entrée revient
            status, resp = self.post_json("/api/visibility",
                                          {"action": "restore", "ids": [target]})
            self.assertEqual(status, 200)
            self.assertTrue(resp["ok"])
        _, payload = self.get_json("/api/graph")
        self.assertNotIn(target, payload["hidden"])
        self.assertEqual(payload["counts"]["visible"], len(TEST_ENTRIES))
        self.assertIn(target, self.entry_ids_from_graph())

    def test_benchmarks_page_suit_l_etat_de_visibilite_sans_javascript(self):
        """app.py doit transmettre état + requête, sinon les liens GET sont morts.

        Sans le couplage, /benchmarks n'affiche ni la marque « masquée », ni le
        lien « afficher les masques » : la page resterait utilisable seulement
        avec JavaScript, ce que le §7-E2 interdit.
        """
        ids = self.entry_ids_from_graph()
        self.assertTrue(ids)
        target = ids[0]
        marquee = f'<tr data-entry="{target}" class="res-masque"'
        _, html = self.get("/benchmarks")
        self.assertIn("/benchmarks?masques=1", html)
        self.assertNotIn(marquee, html)
        try:
            status, resp = self.post_json("/api/visibility",
                                          {"action": "hide", "ids": [target]})
            self.assertEqual(status, 200)
            self.assertTrue(resp["ok"])

            # sans JavaScript : ligne marquée masquée + compteur + bouton Réafficher
            _, html = self.get("/benchmarks")
            self.assertIn(marquee, html)
            self.assertIn("masques / supprimes : 1", html)
            self.assertIn(f'data-vis="unhide" data-id="{target}"', html)

            # ?masques=1 ré-affiche la ligne (et garde le bouton Réafficher)
            _, affiche = self.get("/benchmarks?masques=1")
            self.assertNotIn(marquee, affiche)
            self.assertIn(f'data-vis="unhide" data-id="{target}"', affiche)

            # une valeur fausse (« non ») doit recacher, pas ré-afficher
            _, cache = self.get("/benchmarks?masques=non")
            self.assertIn(marquee, cache)
        finally:
            self.restore_all([target])
        _, html = self.get("/benchmarks")
        self.assertNotIn(marquee, html)

    def test_delete_then_restore(self):
        ids = self.entry_ids_from_graph()
        self.assertTrue(ids)
        target = ids[-1]
        try:
            status, resp = self.post_json("/api/visibility",
                                          {"action": "delete", "ids": [target]})
            self.assertEqual(status, 200)
            self.assertTrue(resp["ok"])
            _, payload = self.get_json("/api/graph")
            self.assertIn(target, payload["hidden"])
            self.assertEqual(payload["counts"]["visible"], len(TEST_ENTRIES) - 1)
        finally:
            self.post_json("/api/visibility", {"action": "restore", "ids": [target]})
        _, payload = self.get_json("/api/graph")
        self.assertEqual(payload["counts"]["visible"], len(TEST_ENTRIES))

    def test_graph_on_empty_index_does_not_crash(self):
        self.write_index([])
        try:
            status, payload = self.get_json("/api/graph")
            self.assertEqual(status, 200)
            self.assertEqual(payload["counts"]["entries"], 0)
            self.assertEqual(payload["counts"]["visible"], 0)
            self.assertIsInstance(payload["charts"], list)
            self.assertTrue(payload["charts"], "les graphiques restent listés (vides)")
        finally:
            self.write_index(TEST_ENTRIES)

    # -- /api/visibility : erreurs -----------------------------------------

    def test_unknown_action_returns_400(self):
        status, resp = self.post_json("/api/visibility",
                                      {"action": "explode", "ids": ["x"]})
        self.assertEqual(status, 400)
        self.assertFalse(resp["ok"])
        self.assertIn("inconnue", resp["error"])

    def test_empty_ids_returns_400(self):
        status, resp = self.post_json("/api/visibility", {"action": "hide", "ids": []})
        self.assertEqual(status, 400)
        self.assertFalse(resp["ok"])

    def test_visibility_state_is_persisted(self):
        _status, resp = self.post_json("/api/visibility",
                                       {"action": "hide", "ids": ["test:introuvable"]})
        self.assertEqual(_status, 200)
        self.assertTrue(os.path.exists(self.visibility_path))
        with open(self.visibility_path, encoding="utf-8") as fh:
            saved = json.load(fh)
        self.assertIn("test:introuvable", saved["hidden"])
        self.post_json("/api/visibility", {"action": "restore", "ids": ["test:introuvable"]})

    # -- /graph -------------------------------------------------------------

    def test_graph_page_status_depends_on_page_graph(self):
        status, body = self.get("/graph")
        try:
            import importlib
            importlib.import_module("page_graph")
            has_page = True
        except Exception:
            has_page = False
        if has_page:
            self.assertEqual(status, 200)
        else:
            self.assertEqual(status, 503)
            self.assertIn("page_graph", body)


if __name__ == "__main__":
    unittest.main()
