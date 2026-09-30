#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests unitaires de settings.py — AUCUN accès réseau, AUCUN /proc réel.

On injecte systématiquement `props_json` (coupe l'appel HTTP) et
`proc_snapshot` (coupe le scan de /proc). Un test dédié remplace les deux
lecteurs réels par des fonctions qui explosent, pour prouver qu'aucune I/O
n'est tentée quand les injections sont fournies.
"""

import json
import os
import sys
import unittest
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

import settings  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Jeu de clés attendu dans TOUT retour (contrat : dict toujours complet).
CLES = sorted(settings._empty().keys())


def fixture_json(nom):
    return json.loads((FIXTURES / nom).read_text(encoding="utf-8"))


def fixture_txt(nom):
    return (FIXTURES / nom).read_text(encoding="utf-8").strip()


class TestDetectProps(unittest.TestCase):
    """/props seul (pas de cmdline) : source=props, décodage inconnu."""

    def setUp(self):
        self.props = fixture_json("settings_props_qwen36_256k.json")
        self.res = settings.detect(8080, props_json=self.props, proc_snapshot=[])

    def test_source_props(self):
        self.assertEqual(self.res["source"], "props")

    def test_ctx_et_modele(self):
        self.assertEqual(self.res["n_ctx"], 262144)
        self.assertEqual(self.res["n_ctx_props"], 262144)
        self.assertIsNone(self.res["n_ctx_cmdline"])
        self.assertTrue(self.res["model_path"].endswith("Qwen3.6-35B-A3B-UD-IQ4_NL.gguf"))
        self.assertEqual(self.res["model"], "Qwen3.6-35B-A3B")
        self.assertEqual(self.res["build"], "b6211-945abe2")
        self.assertEqual(self.res["total_slots"], 1)

    def test_flags_non_decidables(self):
        # Sans cmdline on ne peut PAS affirmer on/off : ce doit rester inconnu.
        self.assertIsNone(self.res["mtp"])
        self.assertIsNone(self.res["flash_attn"])
        self.assertEqual(self.res["flags"], [])

    def test_contexte_par_slot_expose(self):
        self.assertIsInstance(self.res["default_generation_settings"], dict)
        self.assertEqual(self.res["default_generation_settings"]["n_ctx"], 262144)


class TestDetectCmdline(unittest.TestCase):
    """Cmdline présente : source=cmdline, flags réels pris en compte."""

    def setUp(self):
        snap = fixture_json("settings_cmdline_mtp.json")
        self.res = settings.detect(8080, props_json=None, proc_snapshot=snap)

    def test_source_cmdline(self):
        self.assertEqual(self.res["source"], "cmdline")

    def test_mtp_et_draft(self):
        self.assertIs(self.res["mtp"], True)
        self.assertIs(self.res["draft"], True)
        self.assertEqual(self.res["draft_max"], 16)

    def test_flags_reconnus(self):
        self.assertEqual(self.res["n_ctx"], 262144)
        self.assertEqual(self.res["n_ctx_cmdline"], 262144)
        self.assertEqual(self.res["n_gpu_layers"], 99)
        self.assertEqual(self.res["n_cpu_moe"], 10)
        self.assertIs(self.res["flash_attn"], True)
        self.assertEqual(self.res["cache_type_k"], "turbo4")
        self.assertEqual(self.res["cache_type_v"], "turbo3")
        self.assertEqual(self.res["kv"], "turbo4/turbo3")
        self.assertEqual(self.res["parallel"], 1)
        self.assertIs(self.res["jinja"], True)
        self.assertEqual(self.res["model"], "Qwen3.6-35B-A3B")
        self.assertEqual(self.res["port"], 8080)

    def test_flags_inconnus_conserves_bruts(self):
        # -t et --mlock ne sont pas dans la table : ils restent visibles.
        self.assertIn("-t", self.res["flags"])
        self.assertIn("--mlock", self.res["flags"])
        self.assertEqual(set(self.res["unknown_flags"]), {"-t", "--mlock"})

    def test_bon_process_choisi(self):
        # Le snapshot contient aussi un serveur sur :8081 -> ne pas le confondre.
        self.assertNotIn("27B", " ".join(self.res["flags"]))


class TestDetectSansMtp(unittest.TestCase):
    """--mtp absent d'une cmdline lisible => False (règle explicite §6)."""

    def setUp(self):
        ligne = fixture_txt("settings_cmdline_nomtp.txt")
        snap = [{"pid": 999, "cmdline": ligne}]
        self.res = settings.detect(8080, props_json={}, proc_snapshot=snap)

    def test_mtp_false(self):
        self.assertEqual(self.res["source"], "cmdline")
        self.assertIs(self.res["mtp"], False)
        self.assertIs(self.res["draft"], False)

    def test_flags_valeurs(self):
        self.assertEqual(self.res["n_ctx"], 131072)
        self.assertIs(self.res["flash_attn"], False)  # -fa off
        self.assertIs(self.res["mmap"], False)        # --no-mmap
        self.assertEqual(self.res["kv"], "turbo4/turbo3")

    def test_flag_inconnu_load_mode(self):
        self.assertIn("--load-mode", self.res["unknown_flags"])


class TestDetectEntreesDegradees(unittest.TestCase):
    """Aucune donnée / données pourries : jamais d'exception, dict complet."""

    def test_rien_du_tout(self):
        res = settings.detect(8080, props_json={}, proc_snapshot=[])
        self.assertEqual(res["source"], "declared")
        self.assertIsNone(res["n_ctx"])
        self.assertIsNone(res["mtp"])

    def test_json_props_invalide(self):
        res = settings.detect(8080, props_json="{ceci n'est pas du json", proc_snapshot=[])
        self.assertEqual(sorted(res.keys()), CLES)
        self.assertIsNone(res["n_ctx"])
        self.assertIsNotNone(res["error"])

    def test_props_type_absurde(self):
        res = settings.detect(8080, props_json=[1, 2, 3], proc_snapshot=[])
        self.assertEqual(sorted(res.keys()), CLES)

    def test_snapshot_avec_cmdline_en_chaine(self):
        res = settings.detect(8080, props_json={}, proc_snapshot=[
            {"pid": 7, "cmdline": "llama-server --port 8080 -c 65536 --mtp"},
        ])
        self.assertEqual(res["n_ctx"], 65536)
        self.assertIs(res["mtp"], True)

    def test_snapshot_vide_et_cles_presentes(self):
        res = settings.detect(9999, props_json={}, proc_snapshot=[{"pid": 1, "cmdline": []}])
        self.assertEqual(sorted(res.keys()), CLES)

    def test_ollama_nest_pas_un_llama_server(self):
        # Régression vue en prod : « ollama serve » contient « llama » mais ce
        # n'est PAS le serveur llama.cpp -> pas de source cmdline bidon.
        res = settings.detect(8080, props_json={},
                              proc_snapshot=[{"pid": 1659, "cmdline": ["/usr/local/bin/ollama", "serve"]}])
        self.assertEqual(res["source"], "declared")
        self.assertIsNone(res["mtp"])
        self.assertEqual(res["flags"], [])

    def test_jamais_d_exception_sur_entrees_hostiles(self):
        for props in (None, {}, "nope", 42, {"n_ctx": "zzz"}, {"default_generation_settings": []}):
            for snap in ([], None, [None], [{}], [{"pid": "x"}]):
                if props is None and snap is None:
                    continue  # ce cas toucherait le vrai réseau : on l'exclut
                res = settings.detect(8080, props_json=props, proc_snapshot=snap)
                self.assertEqual(sorted(res.keys()), CLES, (props, snap))


class TestAucuneIO(unittest.TestCase):
    """Injections fournies => aucun accès /proc ni réseau."""

    def test_lecteurs_reels_non_appeles(self):
        def boom(*a, **k):
            raise AssertionError("I/O interdite pendant les tests")

        anciens = settings._scan_proc, settings._fetch_props
        settings._scan_proc = boom
        settings._fetch_props = boom
        try:
            snap = fixture_json("settings_cmdline_mtp.json")
            props = fixture_json("settings_props_qwen36_256k.json")
            res = settings.detect(8080, props_json=props, proc_snapshot=snap)
            self.assertEqual(res["source"], "cmdline")
            self.assertEqual(res["n_ctx"], 262144)
        finally:
            settings._scan_proc, settings._fetch_props = anciens


class TestNormalize(unittest.TestCase):
    """Réglages déclarés depuis l'UI."""

    def test_declaration_flags(self):
        raw = fixture_json("settings_declared_ui.json")
        res = settings.normalize(raw)
        self.assertEqual(res["source"], "declared")
        self.assertIs(res["mtp"], True)          # --mtp explicite
        self.assertIs(res["flash_attn"], True)   # -fa on
        self.assertEqual(res["n_ctx"], 262144)
        self.assertEqual(res["kv"], "turbo4/turbo3")
        self.assertEqual(res["declared_flags"], res["flags"])
        self.assertIn("--weird-flag", res["unknown_flags"])

    def test_chaine_de_flags(self):
        res = settings.normalize({"flags": "--no-mtp -fa off -c 131072"})
        self.assertIs(res["mtp"], False)
        self.assertIs(res["flash_attn"], False)
        self.assertEqual(res["n_ctx"], 131072)

    def test_ctx_en_texte(self):
        self.assertEqual(settings.normalize({"n_ctx": "128k"})["n_ctx"], 131072)
        self.assertEqual(settings.normalize({"n_ctx": "256k"})["n_ctx"], 262144)
        self.assertIsNone(settings.normalize({"n_ctx": "beaucoup"})["n_ctx"])

    def test_vide_et_invalide(self):
        for raw in ({}, None, "nope", 3):
            res = settings.normalize(raw)
            self.assertEqual(sorted(res.keys()), CLES)
            self.assertEqual(res["source"], "declared")
            self.assertIsNone(res["mtp"])  # pas de --mtp => inconnu, jamais False

    def test_scalaire_prime_sur_flag(self):
        res = settings.normalize({"flags": "-c 65536", "n_ctx": 262144})
        self.assertEqual(res["n_ctx"], 262144)


class TestShortModel(unittest.TestCase):
    def test_cas_reels(self):
        cas = {
            "/home/gab/models/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf": "Qwen3.6-35B-A3B",
            "/m/qwen3.8-27b-ud-q2_k_xl.gguf": "qwen3.8-27b",
            "k2-horizon-4b-q4_k_m.gguf": "k2-horizon-4b",
            "/x/Qwen3.6-27B-IQ4_XS.gguf": "Qwen3.6-27B",
        }
        for chemin, attendu in cas.items():
            self.assertEqual(settings.short_model(chemin), attendu, chemin)

    def test_vide(self):
        self.assertIsNone(settings.short_model(None))
        self.assertIsNone(settings.short_model(""))


class TestVariantLabel(unittest.TestCase):
    """4 cas demandés : MTP on 256k, sans MTP 128k, flags inconnus, incomplet."""

    def test_mtp_on_256k(self):
        lab = settings.variant_label({"mtp": True, "n_ctx": 262144, "model": "Qwen3.6-35B"})
        self.assertEqual(lab, "MTP on · ctx 256k · Qwen3.6-35B")

    def test_sans_mtp_128k(self):
        lab = settings.variant_label({"mtp": False, "n_ctx": 131072, "flash_attn": True})
        self.assertEqual(lab, "sans MTP · ctx 128k · flash-attn")

    def test_flags_inconnus(self):
        lab = settings.variant_label({
            "mtp": True, "n_ctx": 262144, "model": "Qwen3.6-35B",
            "unknown_flags": ["--weird", "--autre"],
        })
        self.assertIn("+2 flags", lab)
        self.assertLessEqual(len(lab), 60)

    def test_reglages_incomplets(self):
        self.assertEqual(settings.variant_label({}), "réglages inconnus")
        self.assertEqual(settings.variant_label({"n_ctx": 65536}), "ctx 64k")

    def test_toujours_sous_60_caracteres(self):
        gros = {
            "mtp": True, "n_ctx": 262144, "flash_attn": True,
            "kv": "turbo4/turbo3",
            "model": "Qwen3.6-35B-A3B-UD-IQ4_NL-mega-long-name",
            "unknown_flags": ["--a", "--b", "--c", "--d"],
        }
        for cas in (gros, {"mtp": False, "n_ctx": 131072, "flash_attn": True,
                           "kv": "turbo4/turbo3", "model": "Qwen3.6-35B-A3B"}):
            self.assertLessEqual(len(settings.variant_label(cas)), 60)

    def test_kv_non_standard_affiche(self):
        lab = settings.variant_label({"mtp": True, "n_ctx": 262144, "kv": "turbo4/turbo3"})
        self.assertIn("kv turbo4/turbo3", lab)

    def test_entree_invalide(self):
        self.assertEqual(settings.variant_label(None), "réglages inconnus")


class TestVariantKey(unittest.TestCase):
    def test_exemple_du_contrat(self):
        k = settings.variant_key({"mtp": True, "n_ctx": 262144, "flash_attn": True,
                                  "model": "Qwen3.6-35B"})
        self.assertEqual(k, "mtp-on_ctx256k_fa_qwen3-6-35b")

    def test_stable_et_sans_accent_ni_espace(self):
        s = {"mtp": False, "n_ctx": 131072, "model": "Qwen3.6-35B-A3B",
             "kv": "turbo4/turbo3", "unknown_flags": ["--x"]}
        k1, k2 = settings.variant_key(s), settings.variant_key(s)
        self.assertEqual(k1, k2)
        self.assertRegex(k1, r"^[a-z0-9_\-]+$")
        self.assertNotIn(" ", k1)

    def test_deux_variantes_distinctes(self):
        a = settings.variant_key({"mtp": True, "n_ctx": 262144, "model": "Qwen3.6-35B"})
        b = settings.variant_key({"mtp": False, "n_ctx": 262144, "model": "Qwen3.6-35B"})
        c = settings.variant_key({"mtp": True, "n_ctx": 131072, "model": "Qwen3.6-35B"})
        self.assertEqual(len({a, b, c}), 3)

    def test_incomplet(self):
        self.assertEqual(settings.variant_key({}), "mtp-na_ctx-na")

    def test_depuis_detect(self):
        snap = fixture_json("settings_cmdline_mtp.json")
        res = settings.detect(8080, props_json=fixture_json("settings_props_qwen36_256k.json"),
                              proc_snapshot=snap)
        k = settings.variant_key(res)
        self.assertIn("mtp-on", k)
        self.assertIn("ctx256k", k)
        lab = settings.variant_label(res)
        self.assertIn("MTP on", lab)
        self.assertLessEqual(len(lab), 60)


class TestContratDict(unittest.TestCase):
    """Toute sortie expose les clés du contrat §6."""

    def test_cles_du_contrat_presentes(self):
        res = settings.detect(8080, props_json={}, proc_snapshot=[])
        for cle in ("n_ctx", "mtp", "flash_attn", "flags", "source", "model_path", "kv"):
            self.assertIn(cle, res)
        self.assertEqual(sorted(res.keys()), CLES)

    def test_load_json_utf8(self):
        # Doit rester sérialisable en JSON (l'app l'écrit dans bench.json).
        res = settings.detect(8080, props_json=fixture_json("settings_props_qwen36_256k.json"),
                              proc_snapshot=fixture_json("settings_cmdline_mtp.json"))
        json.dumps(res, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
