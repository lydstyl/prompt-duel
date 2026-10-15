#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pensée des bancs (v1.6.5) — le NIVEAU de raisonnement ne doit jamais se perdre.

Ce que ces cas verrouillent, et pourquoi :

  1. `normalize_thinking` rend un niveau canonique : nothink | low | medium | xhigh.
     Les alias que Gabriel emploie vraiment (« think », « on », `true`, « medium ») valent
     `medium` — c'est ce que faisait la v1.6.4 ; `high` vaut `xhigh` parce que le template
     Qwen3.8 servi par Strata remappe lui-même high -> xhigh (vérifié dans /props). Tout
     AUTRE mot retombe sur `nothink` : un mode inconnu ne doit JAMAIS allumer le
     raisonnement par accident.
  2. `think_payload` envoie LES DEUX formes (`chat_template_kwargs.enable_thinking` +
     `reasoning_effort`, et le niveau dans `chat_template_kwargs.reasoning_effort`) : les
     runtimes qui n'écoutent que l'une des deux existent dans le parc (Ternary Bonsai 2 /
     PrismML ignorent enable_thinking).
  3. Le corps CHAT réellement posté par `_completion` porte le niveau demandé (et rien de
     plus quand on est en nothink, comportement d'avant la v1.6.4).
  4. L'étiquette d'un bench porte le niveau (`-think`, `-think-low`, `-think-xhigh`) :
     c'est elle qui sépare les séries dans /graph — low, medium et xhigh ne sont pas la
     même mesure.
  5. `params_label` / `run_suffix` : un run « think-low » ne partage jamais le dossier d'un
     run « think » (sinon find_reusable_run les confondrait).
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import app          # noqa: E402
import bench_tests  # noqa: E402


class TestNiveauxDeRaisonnement(unittest.TestCase):
    def test_les_quatre_niveaux_existent(self):
        self.assertEqual(tuple(bench_tests.THINKING_MODES),
                         ("nothink", "low", "medium", "xhigh"))
        for mode in bench_tests.THINKING_MODES:
            self.assertIn(mode, bench_tests.THINKING_LABELS)

    def test_alias_historiques_vers_medium(self):
        for value in ("think", "thinking", "medium", "on", "true", "1", "oui", True):
            self.assertEqual(bench_tests.normalize_thinking(value), "medium", value)

    def test_niveaux_explicites(self):
        self.assertEqual(bench_tests.normalize_thinking("low"), "low")
        self.assertEqual(bench_tests.normalize_thinking("xhigh"), "xhigh")
        # le template Qwen3.8 remappe high -> xhigh : on suit le modèle, pas l'intuition
        self.assertEqual(bench_tests.normalize_thinking("high"), "xhigh")

    def test_valeurs_qui_eteignent(self):
        for value in ("nothink", "no-think", "none", "off", "false", "0", "non", False, None):
            self.assertEqual(bench_tests.normalize_thinking(value), "nothink", value)

    def test_inconnu_ne_derive_pas(self):
        self.assertEqual(bench_tests.normalize_thinking("turbo"), "nothink")
        self.assertEqual(bench_tests.normalize_thinking(""), "nothink")

    def test_payload_les_deux_formes_et_le_niveau(self):
        for level, effort in (("low", "low"), ("medium", "medium"), ("xhigh", "xhigh")):
            body = bench_tests.think_payload(level)
            self.assertEqual(body["reasoning_effort"], effort)
            self.assertEqual(body["chat_template_kwargs"]["reasoning_effort"], effort)
            self.assertIs(body["chat_template_kwargs"]["enable_thinking"], True)
        off = bench_tests.think_payload("nothink")
        self.assertEqual(off["reasoning_effort"], "none")
        self.assertIs(off["chat_template_kwargs"]["enable_thinking"], False)
        self.assertNotIn("reasoning_effort", off["chat_template_kwargs"])

    def test_payload_alias(self):
        self.assertEqual(bench_tests.think_payload("think")["reasoning_effort"], "medium")
        self.assertEqual(bench_tests.think_payload("high")["reasoning_effort"], "xhigh")


class TestCorpsChat(unittest.TestCase):
    """Le niveau doit atteindre le corps réellement posté, pas seulement l'objet d'env."""

    def _corps(self, thinking):
        captured = {}

        def faux_post(url, body, timeout):
            captured["url"] = url
            captured["body"] = body
            return ({"choices": [{"message": {"content": "ok"}}],
                     "timings": {"prompt_n": 10, "predicted_n": 8,
                                 "prompt_per_second": 100.0, "predicted_per_second": 10.0}}, 1.0)

        ancien_post, ancien_caps = bench_tests._post, dict(bench_tests._SERVER_CAPS)
        bench_tests._post = faux_post
        try:
            bench_tests._SERVER_CAPS[("native", "http://x")] = False   # → tir chat direct
            bench_tests._completion("http://x", "salut", 8, 5, thinking=thinking)
        finally:
            bench_tests._post = ancien_post
            bench_tests._SERVER_CAPS.clear()
            bench_tests._SERVER_CAPS.update(ancien_caps)
        return captured["body"]

    def test_medium_dans_le_corps(self):
        body = self._corps("medium")
        self.assertEqual(body["reasoning_effort"], "medium")
        self.assertIs(body["chat_template_kwargs"]["enable_thinking"], True)

    def test_low_et_xhigh_dans_le_corps(self):
        self.assertEqual(self._corps("low")["reasoning_effort"], "low")
        self.assertEqual(self._corps("xhigh")["reasoning_effort"], "xhigh")

    def test_nothink_par_defaut(self):
        body = self._corps("nothink")
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertIs(body["chat_template_kwargs"]["enable_thinking"], False)


class TestEtiquetteEtEnv(unittest.TestCase):
    def test_etiquette_par_niveau(self):
        self.assertEqual(app.bench_label("Qwen3.8-27B-UD-Q4_K_XL", 262144, "medium"),
                         "qwen3-8-27b-ud-q4-k-xl-256k-think")
        self.assertEqual(app.bench_label("Qwen3.8-27B-UD-Q4_K_XL", 262144, "low"),
                         "qwen3-8-27b-ud-q4-k-xl-256k-think-low")
        self.assertEqual(app.bench_label("Qwen3.8-27B-UD-Q4_K_XL", 262144, "xhigh"),
                         "qwen3-8-27b-ud-q4-k-xl-256k-think-xhigh")
        self.assertEqual(app.bench_label("Qwen3.8-27B-UD-Q4_K_XL", 262144),
                         "qwen3-8-27b-ud-q4-k-xl-256k")
        # « think » reste l'alias de medium
        self.assertEqual(app.bench_label("Qwen3.8-27B-UD-Q4_K_XL", 262144, "think"),
                         "qwen3-8-27b-ud-q4-k-xl-256k-think")

    def test_env_par_defaut_sans_raisonnement(self):
        env = bench_tests.BenchEnv(base="http://127.0.0.1:8080", port=8080, label="x",
                                   run_dir="/tmp/x")
        self.assertEqual(env.thinking, "nothink")

    def test_env_garde_le_niveau(self):
        env = bench_tests.BenchEnv(base="http://127.0.0.1:8080", port=8080, label="x",
                                   run_dir="/tmp/x", thinking="medium")
        self.assertEqual(env.thinking, "medium")
        env = bench_tests.BenchEnv(base="http://127.0.0.1:8080", port=8080, label="x",
                                   run_dir="/tmp/x", thinking="low")
        self.assertEqual(env.thinking, "low")

    def test_journal_du_bench_dit_la_pensee(self):
        md = bench_tests.render_md({"run_id": "r", "thinking": "medium", "results": []})
        self.assertIn("raisonnement medium", md)
        md = bench_tests.render_md({"run_id": "r", "thinking": "low", "results": []})
        self.assertIn("raisonnement low", md)


class TestNouveauxTests(unittest.TestCase):
    """Les trois niveaux sont déclarés, disponibles et estimables."""

    def test_les_trois_niveaux_sont_au_catalogue(self):
        for tid in ("niveau-1", "niveau-2", "niveau-3"):
            test = bench_tests.TESTS_BY_ID.get(tid)
            self.assertIsNotNone(test, tid)
            self.assertEqual(test["kind"], "niveau")
            self.assertEqual(test["group"], "niveaux")
            self.assertIn(test["kind"], bench_tests.RUNNERS)
            self.assertTrue(test.get("needs_ctx"))
            self.assertTrue(test.get("script", "").startswith("niveau"))

    def test_estimation_dun_niveau(self):
        est, src = bench_tests.estimate_seconds(bench_tests.TESTS_BY_ID["niveau-1"], {})
        self.assertGreater(est, 60)
        self.assertIn(src, ("défaut", "mesuré"))

    def test_calibrage_par_test(self):
        calib = {}
        bench_tests.update_calib(calib, "modele@131072", {"by_test": {"niveau-1": 250.0}},
                                 last_summary="s", test_id="niveau-1")
        est, src = bench_tests.estimate_seconds(bench_tests.TESTS_BY_ID["niveau-1"],
                                                calib["modele@131072"])
        self.assertEqual(src, "mesuré")
        self.assertAlmostEqual(est, 260.0, places=1)
        # un AUTRE niveau ne doit pas hériter de cette mesure
        est2, _ = bench_tests.estimate_seconds(bench_tests.TESTS_BY_ID["niveau-3"],
                                               calib["modele@131072"])
        self.assertNotAlmostEqual(est2, 260.0, places=1)


class TestOptionsDeBench(unittest.TestCase):
    def test_options_validees(self):
        o = app.bench_options({"temperature": "0.6", "top_p": 0.95, "top_k": 20,
                               "seed": 7, "repeats": 3, "budget_s": 600, "bidon": 1})
        self.assertEqual(o["temperature"], 0.6)
        self.assertEqual(o["top_p"], 0.95)
        self.assertEqual(o["top_k"], 20)
        self.assertEqual(o["repeats"], 3)
        self.assertNotIn("bidon", o)

    def test_options_hors_bornes_ignorees(self):
        o = app.bench_options({"temperature": 9, "top_p": -1, "repeats": 99, "seed": "x"})
        self.assertEqual(o, {})

    def test_variables_denvironnement(self):
        app.apply_bench_options({"temperature": 0.6, "seed": 7})
        self.assertEqual(os.environ.get("BENCH_TEMP"), "0.6")
        self.assertEqual(os.environ.get("BENCH_SEED"), "7")
        app.apply_bench_options({})
        self.assertIsNone(os.environ.get("BENCH_TEMP"))
        self.assertIsNone(os.environ.get("BENCH_SEED"))

    def test_niveau_options_lues_par_le_harness(self):
        os.environ["BENCH_REPEATS"] = "2"
        os.environ["BENCH_TOP_K"] = "20"
        try:
            o = bench_tests.niveau_options()
        finally:
            os.environ.pop("BENCH_REPEATS", None)
            os.environ.pop("BENCH_TOP_K", None)
        self.assertEqual(o["repeats"], 2)
        self.assertEqual(o["top_k"], 20)


if __name__ == "__main__":
    unittest.main()
