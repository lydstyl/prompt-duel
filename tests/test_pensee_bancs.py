#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pensée des bancs (v1.6.4) — le mode raisonnement ne doit jamais se perdre.

Ce que ces cas verrouillent, et pourquoi :

  1. `normalize_thinking` accepte les mots que Gabriel emploie vraiment (« medium »,
     « on », `true`) et retombe sur `nothink` pour tout le reste — un mode inconnu ne
     doit JAMAIS allumer le raisonnement par accident.
  2. `think_payload` envoie LES DEUX formes (`chat_template_kwargs.enable_thinking` et
     `reasoning_effort`) : les runtimes qui n'écoutent que l'une des deux existent dans
     le parc (Ternary Bonsai 2 / PrismML ignorent enable_thinking).
  3. Le corps CHAT réellement posté par `_completion` porte le mode demandé (et rien de
     plus quand on est en nothink, comportement d'avant la v1.6.4).
  4. L'étiquette d'un bench avec raisonnement porte `-think` : c'est elle qui sépare les
     séries dans /graph (avec et sans raisonnement ne sont pas la même mesure).
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


class TestNormalizeThinking(unittest.TestCase):
    def test_valeurs_qui_allument(self):
        for value in ("think", "thinking", "medium", "on", "true", "1", "oui", True):
            self.assertEqual(bench_tests.normalize_thinking(value), "think", value)

    def test_valeurs_qui_eteignent(self):
        for value in ("nothink", "no-think", "none", "off", "false", "0", "non", False, None):
            self.assertEqual(bench_tests.normalize_thinking(value), "nothink", value)

    def test_inconnu_ne_derive_pas(self):
        self.assertEqual(bench_tests.normalize_thinking("high"), "nothink")
        self.assertEqual(bench_tests.normalize_thinking(""), "nothink")

    def test_payload_les_deux_formes(self):
        think = bench_tests.think_payload("think")
        self.assertEqual(think["reasoning_effort"], "medium")
        self.assertIs(think["chat_template_kwargs"]["enable_thinking"], True)
        off = bench_tests.think_payload("nothink")
        self.assertEqual(off["reasoning_effort"], "none")
        self.assertIs(off["chat_template_kwargs"]["enable_thinking"], False)


class TestCorpsChat(unittest.TestCase):
    """Le mode doit atteindre le corps réellement posté, pas seulement l'objet d'env."""

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

    def test_think_dans_le_corps(self):
        body = self._corps("think")
        self.assertEqual(body["reasoning_effort"], "medium")
        self.assertIs(body["chat_template_kwargs"]["enable_thinking"], True)

    def test_nothink_par_defaut(self):
        body = self._corps("nothink")
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertIs(body["chat_template_kwargs"]["enable_thinking"], False)


class TestEtiquetteEtEnv(unittest.TestCase):
    def test_etiquette_think(self):
        self.assertEqual(app.bench_label("Qwen3.8-27B-UD-Q4_K_XL", 262144, "think"),
                         "qwen3-8-27b-ud-q4-k-xl-256k-think")
        self.assertEqual(app.bench_label("Qwen3.8-27B-UD-Q4_K_XL", 262144),
                         "qwen3-8-27b-ud-q4-k-xl-256k")
        self.assertEqual(app.bench_label("Qwen3.8-27B-UD-Q4_K_XL", 262144, "medium"),
                         "qwen3-8-27b-ud-q4-k-xl-256k-think")

    def test_env_par_defaut_sans_raisonnement(self):
        env = bench_tests.BenchEnv(base="http://127.0.0.1:8080", port=8080, label="x",
                                   run_dir="/tmp/x")
        self.assertEqual(env.thinking, "nothink")

    def test_env_accepte_medium(self):
        env = bench_tests.BenchEnv(base="http://127.0.0.1:8080", port=8080, label="x",
                                   run_dir="/tmp/x", thinking="medium")
        self.assertEqual(env.thinking, "think")

    def test_journal_du_bench_dit_la_pensee(self):
        md = bench_tests.render_md({"run_id": "r", "thinking": "think", "results": []})
        self.assertIn("raisonnement medium", md)


if __name__ == "__main__":
    unittest.main()
