#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/test_openai_only.py — les bancs vitesse/mémoire face à un serveur SANS routes natives.

Cas réel visé : Strata (et les moteurs OpenAI-only) ne servent ni `/tokenize` ni
`/completion`. Avant le correctif, tout le banc tombait en `HTTPError 404` en 0,1 s :
aucun chiffre de vitesse, aucun rappel de contexte. Ce test monte un faux serveur qui
n'expose QUE `/v1/chat/completions` et vérifie que :

  - la détection conclut « pas de routes natives » (et qu'aucune requête inutile n'est
    faite sur `/completion` ensuite) ;
  - la vitesse est lue dans le bloc `timings` de la réponse chat (mêmes champs que
    llama.cpp) ;
  - le remplissage est estimé par un ratio caractères/token mesuré côté serveur, le
    compte EXACT venant de `prompt_n` ;
  - un needle dont le code n'apparaît que dans `reasoning_content` est compté comme
    un rappel RÉUSSI (modèle qui raisonne), pas comme un échec.
"""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench_tests  # noqa: E402

RATIO = 4.0          # 4 caractères par token : ratio que le faux serveur doit retrouver


class _Handler(BaseHTTPRequestHandler):
    """Serveur OpenAI-only : /tokenize et /completion rendent 404, le chat répond."""

    def log_message(self, *args):        # silence
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except ValueError:
            body = {}
        if self.path in ("/tokenize", "/completion"):
            return self._json(404, {"error": {"message": "Not Found"}})
        if self.path != "/v1/chat/completions":
            return self._json(404, {"error": {"message": "Not Found"}})
        text = ""
        for m in body.get("messages") or []:
            text += str(m.get("content") or "")
        prompt_n = int(round(len(text) / RATIO)) + 8      # +8 = gabarit du template
        raisonne = bench_tests.NEEDLE_TOKEN in text.upper().replace(" ", "-")
        return self._json(200, {
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant",
                "content": None if raisonne else "ok",
                # le code n'apparaît QUE dans le raisonnement (cas du modèle qui pense)
                "reasoning_content": (f"je cherche le code… {bench_tests.NEEDLE_TOKEN}"
                                      if raisonne else "réponse directe"),
            }}],
            "usage": {"prompt_tokens": prompt_n, "completion_tokens": 16},
            "timings": {"prompt_n": prompt_n, "prompt_per_second": 250.0,
                        "predicted_n": 16, "predicted_per_second": 12.5},
        })


class _CacheHandler(_Handler):
    """Même serveur, mais il sert les tirs LONGS suivants depuis son cache de prompt :
    le 1er tir évalue 504 tokens, les autres 7 (+497 du cache). Un PP moyenné sur les
    trois (43 t/s) serait faux — c'est exactement le défaut observé sur le .224.

    Les tirs courts (sondes de ratio caractères/token) ne comptent pas : seul un prompt
    de plus de 100 tokens est considéré comme un tir de vitesse.
    """

    longs = 0

    def _json(self, code, payload):
        t = payload.get("timings") if isinstance(payload, dict) else None
        if code == 200 and t and (t.get("prompt_n") or 0) > 100:
            _CacheHandler.longs += 1
            if _CacheHandler.longs <= 1:
                t.update({"prompt_n": 504, "cache_n": 0, "prompt_per_second": 250.0})
            else:
                t.update({"prompt_n": 7, "cache_n": 497, "prompt_per_second": 43.0})
        return super()._json(code, payload)


class OpenAiOnlyBenchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.tmp = tempfile.mkdtemp(prefix="pd-openai-only.")
        cls.env = bench_tests.BenchEnv(cls.base, cls.httpd.server_address[1], "faux-openai",
                                      cls.tmp, local=False, nvidia=False, model_slug="faux-openai",
                                      n_ctx=4096)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_01_detection_sans_routes_natives(self):
        self.assertFalse(bench_tests.server_is_llamacpp(self.base))
        self.assertEqual(bench_tests.server_is_llamacpp(self.base), False)   # mise en cache

    def test_02_tokenize_estime_par_le_serveur(self):
        # l'estimation ne sert qu'à DIMENSIONNER le remplissage : elle vaut ±2 % du vrai
        # compte (le chiffre publié, lui, vient de `prompt_n` de la réponse mesurée).
        texte = "a" * 4000
        n = bench_tests._tokenize(self.base, texte)
        self.assertLessEqual(abs(n - 1000), 25, f"estimation {n} trop loin de 1000")

    def test_03_vitesse_lue_dans_les_timings_chat(self):
        t = bench_tests._completion(self.base, "x" * 400, 16, 60)
        self.assertEqual(t["mode"], "chat")          # aucune tentative sur /completion
        self.assertEqual(t["pp_tps"], 250.0)
        self.assertEqual(t["tg_tps"], 12.5)
        self.assertEqual(t["prompt_n"], 108)         # 400/4 + 8

    def test_04_vitesse_courte_complete(self):
        test = bench_tests.TESTS_BY_ID["vitesse-court"]
        out = bench_tests._run_vitesse(test, self.env, lambda *_: None, lambda: False)
        self.assertTrue(out["ok"])
        self.assertEqual(out["metrics"]["pp_tps"], 250.0)
        self.assertEqual(out["metrics"]["tg_tps"], 12.5)

    def test_05_needle_lu_dans_le_reasoning(self):
        test = {"id": "memoire-test", "group": "memoire", "kind": "needle",
                "ctx": 8192, "fill": 0.5, "depth": 0.85, "n_predict": 32}
        out = bench_tests._run_needle(test, self.env, lambda *_: None, lambda: False)
        self.assertTrue(out["ok"])
        self.assertEqual(out["metrics"]["reponse_dans"], "reasoning")
        self.assertEqual(out["metrics"]["mode"], "chat")
        self.assertGreater(out["metrics"]["prompt_n"], 3000)


class CachePollutionTest(unittest.TestCase):
    """Un tir servi par le cache du serveur ne doit pas entrer dans la moyenne de prefill."""

    @classmethod
    def setUpClass(cls):
        _CacheHandler.longs = 0
        bench_tests._SERVER_CAPS.clear()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _CacheHandler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.tmp = tempfile.mkdtemp(prefix="pd-cache.")
        cls.env = bench_tests.BenchEnv(cls.base, cls.httpd.server_address[1], "faux-cache",
                                      cls.tmp, local=False, nvidia=False, model_slug="faux-cache",
                                      n_ctx=4096)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_les_tirs_du_cache_sont_ecartes_de_la_moyenne(self):
        test = bench_tests.TESTS_BY_ID["vitesse-court"]
        out = bench_tests._run_vitesse(test, self.env, lambda *_: None, lambda: False)
        self.assertEqual(out["metrics"]["runs_exclus_cache"], 2)
        self.assertEqual(out["metrics"]["pp_tps"], 250.0)     # pas 109 t/s de moyenne polluée
        self.assertEqual(out["metrics"]["prompt_n"], 504)
        self.assertIn("écarté", out["summary"])
        self.assertTrue(out["ok"])


if __name__ == "__main__":
    unittest.main()
