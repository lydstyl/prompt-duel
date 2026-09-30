#!/usr/bin/env python3
"""Tests unitaires de `metrics.py` (unittest, stdlib uniquement).

Les fixtures `index_extract.json` (5 entrees, une par famille) et
`index_vault.json` (les 68 entrees reelles de l'index du vault, verbatim) sont
des extraits REELS : les tests portent donc sur l'etat reel des donnees.
"""
import copy
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import metrics  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name):
    with open(FIXTURES / name, encoding="utf-8") as fh:
        return json.load(fh)


def vault_entries():
    """Les 68 entrees reelles (extrait du vault, lecture seule cote fixture)."""
    return load_fixture("index_vault.json")["entries"]


def extract_entries():
    """5 entrees reelles, une par famille historique."""
    return load_fixture("index_extract.json")["entries"]


def ent(**kwargs):
    """Entree synthetique minimale (pour les cas limites)."""
    base = {
        "kind": "speed",
        "source": "llm-benchmarks/test/vitesse/case-35b_256k.log",
        "date": "2026-01-01T00:00:00",
        "label": "35b_iq4nl_256k",
        "ctx": 262144,
        "model": "Qwen3.6-35B-A3B UD-IQ4_NL",
        "model_slug": "qwen3.6-35b-a3b-ud-iq4_nl",
    }
    base.update(kwargs)
    return base


# --------------------------------------------------------------- registre
class TestRegistre(unittest.TestCase):

    def test_kinds_couvre_les_familles_attendues(self):
        for kind in ("speed", "context", "battery", "humaneval", "duel",
                     "vitesse", "memoire", "intelligence"):
            self.assertIn(kind, metrics.KINDS, f"famille {kind} absente du registre")

    def test_kindspec_et_metricspec_ont_exactement_les_cles_du_contrat(self):
        for kind, spec in metrics.KINDS.items():
            self.assertEqual(set(spec), {"label", "metrics", "duration"}, kind)
            self.assertTrue(spec["label"], kind)
            for m in metrics.metric_specs(kind):
                self.assertEqual(set(m), {"key", "label", "unit", "better"}, m)
                self.assertIn(m["better"], ("high", "low"), m)
                self.assertTrue(m["key"] and m["label"], m)
            d = metrics.duration_spec(kind)
            self.assertEqual(set(d), {"key", "label", "unit"}, d)

    def test_metriques_chiffrees_des_familles_historiques(self):
        attendu = {
            "speed": {"pp", "tg", "vram", "load_s"},
            "context": {"prefill_tps", "tokens", "needle", "peak_vram"},
            "battery": {"score_avg", "lat_median", "lat_max", "fails"},
            "humaneval": {"score_pct", "avg_latency_s", "avg_tokens_per_sec",
                          "passed", "total"},
            "duel": {"tok_s", "prompts_ok", "prompts_total", "tokens_out"},
            "vitesse": {"pp_tps", "tg_tps"},
            "memoire": {"needle", "prefill_tps", "tokens"},
            "intelligence": {"score_pct", "score_avg", "passed", "total"},
        }
        for kind, cles in attendu.items():
            presentes = {m["key"] for m in metrics.metric_specs(kind)}
            self.assertTrue(cles <= presentes, f"{kind} : {cles - presentes} manquant")

    def test_durees_par_famille(self):
        self.assertEqual(metrics.duration_spec("speed")["key"], "load_s")
        self.assertEqual(metrics.duration_spec("context")["key"], "elapsed_s")
        self.assertEqual(metrics.duration_spec("battery")["key"], "total_wall_s")
        self.assertEqual(metrics.duration_spec("humaneval")["key"], "wall_s")
        self.assertEqual(metrics.duration_spec("duel")["key"], "duration_s")
        self.assertEqual(metrics.duration_spec("vitesse")["key"], "duree_s")

    def test_famille_inconnue_ne_leve_pas(self):
        self.assertEqual(metrics.metric_specs("bidon"), [])
        self.assertEqual(set(metrics.duration_spec("bidon")), {"key", "label", "unit"})
        self.assertEqual(metrics.metric_specs("batterie"),
                         metrics.metric_specs("battery"), "alias FR")

    def test_metric_specs_renvoie_des_copies(self):
        metrics.metric_specs("speed")[0]["label"] = "pollue"
        self.assertEqual(metrics.metric_specs("speed")[0]["label"], "Prefill")


# --------------------------------------------------------------- identifiants
class TestEntryId(unittest.TestCase):

    def test_derive_de_kind_et_source(self):
        e = ent()
        self.assertEqual(metrics.entry_id(e), f"speed:{e['source']}")

    def test_unicite_sur_les_68_entrees_reelles(self):
        ids = [metrics.entry_id(e) for e in vault_entries()]
        self.assertEqual(len(vault_entries()), 68)
        self.assertEqual(len(set(ids)), len(ids), "entry_id en collision")

    def test_stable_quand_on_reordonne_ou_recopie(self):
        entries = vault_entries()
        avant = {metrics.entry_id(e) for e in entries}
        melange = list(reversed(entries))
        apres = {metrics.entry_id(e) for e in melange}
        self.assertEqual(avant, apres)
        # la normalisation ne change pas l'identifiant (idempotence)
        for norm, raw in zip(metrics.normalize_entries(entries), entries):
            self.assertEqual(norm["entry_id"], metrics.entry_id(raw))

    def test_ne_depend_pas_de_la_date_de_generation_ni_de_la_position(self):
        e = ent()
        ids = {metrics.entry_id(e) for _ in range(3)}
        self.assertEqual(len(ids), 1)
        # meme entree a la position 0 d'une liste de 1, puis au milieu d'une
        # liste de 5 : l'identifiant ne bouge pas (aucun index de liste)
        seul = [e]
        melange = [ent(source="x.log"), ent(source="y.log"), e,
                   ent(source="z.log"), ent(source="w.log")]
        self.assertEqual(metrics.entry_id(seul[0]), metrics.entry_id(melange[2]))
        self.assertEqual(metrics.entry_id(melange[2]), metrics.entry_id(e))

    def test_repli_deterministe_sans_source(self):
        a = {"kind": "speed", "label": "x", "model": "M", "ctx": 8192}
        b = dict(a)
        self.assertEqual(metrics.entry_id(a), metrics.entry_id(b))
        self.assertTrue(metrics.entry_id(a).startswith("speed:"))
        self.assertNotEqual(metrics.entry_id(a), metrics.entry_id({"kind": "speed"}))
        # pas de compteur : l'ordre d'une liste n'entre pas dans le calcul
        self.assertEqual(metrics.entry_id(a), metrics.entry_id(dict(a)))


# --------------------------------------------------------------- variantes
class TestVariante(unittest.TestCase):

    def test_exemple_du_contrat(self):
        self.assertEqual(
            metrics.variant_key({"mtp": True, "ctx": 262144, "temperature": 0.37}),
            "mtp-on_ctx256k_t037")

    def test_mtp_on_et_off_donnent_deux_series(self):
        on = ent(config=None, method="bench 128 tok (draft MTP)", source="a.log")
        off = ent(config=None, method="bench 128 tok (nomtp)", source="b.log")
        neutre = ent(config=None, method="bench 128 tok", source="c.log")
        k_on, k_off, k_neutre = (metrics.entry_variant(x)["key"] for x in (on, off, neutre))
        self.assertNotEqual(k_on, k_off)
        self.assertNotEqual(k_on, k_neutre)
        self.assertNotEqual(k_off, k_neutre)
        self.assertIn("mtp-on", k_on)
        self.assertIn("mtp-off", k_off)
        self.assertNotIn("mtp", k_neutre)
        self.assertIsNone(metrics.entry_variant(neutre)["mtp"])
        self.assertIs(metrics.entry_variant(on)["mtp"], True)
        self.assertIs(metrics.entry_variant(off)["mtp"], False)

    def test_mtp_via_le_texte_de_config_humaneval(self):
        base = {"kind": "humaneval", "model": "Qwen3.8-27B UD-Q4_K_XL", "total": 164,
                "passed": 150, "score_pct": 91.5, "date": "2026-09-16T10:00:00"}
        on = dict(base, source="humaneval/run-mtp/humaneval-summary.json",
                  config="ctx262144 KV q8_0 n_max3 split0.55 draft MTP | 2 GPU .224")
        off = dict(base, source="humaneval/run-nomtp/humaneval-summary.json",
                   config="ctx262144 KV q8_0 n_max3 split0.55 nomtp | 2 GPU .224")
        sans_info = dict(base, source="humaneval/run-plain/humaneval-summary.json",
                         config="ctx262144 KV q8_0 n_max3 split0.55")
        v_on, v_off, v_plain = (metrics.entry_variant(x) for x in (on, off, sans_info))
        self.assertIn("mtp-on", v_on["key"])
        self.assertIn("mtp-off", v_off["key"])
        self.assertNotIn("mtp", v_plain["key"])
        self.assertEqual(len({v_on["key"], v_off["key"], v_plain["key"]}), 3)
        self.assertIn("ctx256k", v_on["key"])
        self.assertIn("n-max3", v_on["flags"])

    def test_aucune_invention_quand_l_info_manque(self):
        v = metrics.entry_variant({"kind": "speed", "source": "x.log"})
        self.assertEqual(v["key"], "standard")
        self.assertEqual(v["label"], "standard")
        self.assertIsNone(v["mtp"])
        self.assertIsNone(v["thinking"])
        self.assertIsNone(v["ctx"])
        self.assertEqual(v["flags"], [])

    def test_cle_sans_accents_sans_espaces_et_lisible(self):
        for e in vault_entries():
            key = metrics.entry_variant(e)["key"]
            self.assertTrue(key, e.get("source"))
            self.assertEqual(key, metrics.strip_accents(key))
            self.assertNotIn(" ", key)
            self.assertEqual(key, key.lower())
            self.assertLessEqual(len(key), 120)

    def test_ctx_flags_thinking_temperature(self):
        duel = next(e for e in vault_entries() if e["kind"] == "duel"
                    and e.get("temperature") == 0.2)
        v = metrics.entry_variant(duel)
        self.assertEqual(v["ctx"], duel["ctx"])
        self.assertIn("ctx", v["key"])
        self.assertIn("_t02", v["key"])
        self.assertIs(v["thinking"], False)
        battery = next(e for e in vault_entries() if e["kind"] == "battery"
                       and e.get("mode") == "think")
        self.assertIs(metrics.entry_variant(battery)["thinking"], True)
        self.assertIn("think", metrics.entry_variant(battery)["key"])

    def test_le_modele_distingue_les_series(self):
        a = ent(source="a.log", model="Qwen3.8-27B UD-Q2_K_XL",
                model_slug="qwen3.8-27b-ud-q2_k_xl")
        b = ent(source="b.log", model="Qwen3.8-27B UD-Q4_K_XL",
                model_slug="qwen3.8-27b-ud-q4_k_xl")
        self.assertNotEqual(metrics.entry_variant(a)["key"],
                            metrics.entry_variant(b)["key"])

    def test_deux_quantifications_du_meme_modele_restent_distinctes(self):
        a = {"kind": "battery", "model": "Qwen3.6-35B-A3B UD-IQ4_NL"}
        b = {"kind": "battery", "model": "Qwen3.6-35B-A3B UD-Q4_K_XL"}
        self.assertNotEqual(metrics.entry_variant(a)["key"],
                            metrics.entry_variant(b)["key"])

    def test_modele_tronque_garde_une_empreinte(self):
        long_a = metrics.entry_variant({"kind": "humaneval", "model": "A" * 40 + "-Q4"})
        long_b = metrics.entry_variant({"kind": "humaneval", "model": "A" * 40 + "-Q5"})
        self.assertNotEqual(long_a["key"], long_b["key"])

    def test_contrat_de_cles_de_la_variante(self):
        v = metrics.entry_variant(ent())
        for cle in ("key", "label", "ctx", "flags", "thinking", "model"):
            self.assertIn(cle, v)
        self.assertIsInstance(v["flags"], list)
        self.assertTrue(all(isinstance(f, str) for f in v["flags"]))

    def test_label_neutre_et_libelle_fr(self):
        self.assertEqual(metrics.variant_label({}), "standard")
        v = metrics.entry_variant(ent(temperature=0.2, thinking=False, max_tokens=16384))
        self.assertLessEqual(len(v["label"]), 60)
        self.assertIn("nothink", v["label"])


# --------------------------------------------------------------- normalisation
class TestNormalisation(unittest.TestCase):

    def test_ajoute_les_champs_sans_toucher_aux_entrees(self):
        entries = extract_entries()
        original = copy.deepcopy(entries)
        norm = metrics.normalize_entries(entries)
        self.assertEqual(len(norm), len(entries))
        self.assertEqual(entries, original, "les entrees recues ont ete modifiees")
        for brut, propre in zip(entries, norm):
            self.assertIsNot(brut, propre)
            self.assertEqual(propre["entry_id"], metrics.entry_id(brut))
            self.assertIn("variant", propre)
            self.assertIn("metric_values", propre)
            self.assertIn("duration_s", propre)
            for cle, valeur in brut.items():
                self.assertEqual(propre[cle], valeur, f"cle {cle} alteree")

    def test_idempotence(self):
        une = metrics.normalize_entries(vault_entries())
        deux = metrics.normalize_entries(une)
        self.assertEqual([e["entry_id"] for e in une], [e["entry_id"] for e in deux])
        self.assertEqual([e["variant"]["key"] for e in une],
                         [e["variant"]["key"] for e in deux])

    def test_ordre_conserve(self):
        entries = vault_entries()
        norm = metrics.normalize_entries(entries)
        self.assertEqual([e["source"] for e in norm], [e["source"] for e in entries])

    def test_valeurs_metriques_normalisees(self):
        speed = next(e for e in vault_entries() if e["kind"] == "speed" and e.get("vram"))
        mv = metrics.entry_metric_values(speed)
        self.assertEqual(mv["vram"], 12.8, "chaine '12.4 / 12.8 Go' -> le pic")
        self.assertEqual(mv["pp"], float(speed["pp"]))
        if speed.get("load_s") is None:
            self.assertIsNone(mv["load_s"])
        else:
            self.assertEqual(mv["load_s"], float(speed["load_s"]))
        battery = next(e for e in vault_entries() if e["kind"] == "battery")
        self.assertEqual(metrics.entry_metric_values(battery)["fails"],
                         float(len(battery["fails"])))
        hum = next(e for e in vault_entries() if e["kind"] == "humaneval")
        self.assertEqual(metrics.entry_metric_values(hum)["score_pct"], 96.0)

    def test_parse_number(self):
        self.assertEqual(metrics.parse_number(True), 1.0)
        self.assertEqual(metrics.parse_number(False), 0.0)
        self.assertEqual(metrics.parse_number("0,846"), 0.846)
        self.assertEqual(metrics.parse_number("11.7 / 11.9 Go"), 11.9)
        self.assertIsNone(metrics.parse_number("-"))
        self.assertIsNone(metrics.parse_number(None))
        self.assertEqual(metrics.parse_number(["a", "b", "c"]), 3.0)

    def test_duree_par_famille(self):
        entries = {e["kind"]: e for e in extract_entries()}
        self.assertEqual(metrics.entry_duration_s(entries["speed"]),
                         float(entries["speed"]["load_s"]))
        self.assertEqual(metrics.entry_duration_s(entries["context"]),
                         float(entries["context"]["elapsed_s"]))
        self.assertEqual(metrics.entry_duration_s(entries["battery"]),
                         float(entries["battery"]["total_wall_s"]))
        self.assertEqual(metrics.entry_duration_s(entries["humaneval"]),
                         float(entries["humaneval"]["wall_s"]))
        self.assertEqual(metrics.entry_duration_s(entries["duel"]),
                         float(entries["duel"]["duration_s"]))
        self.assertEqual(metrics.entry_duration_s(
            {"kind": "vitesse", "duree_s": 4.5}), 4.5)
        self.assertIsNone(metrics.entry_duration_s({"kind": "duel"}))

    def test_needle_booleen_devient_0_ou_1(self):
        ctx = next(e for e in extract_entries() if e["kind"] == "context")
        self.assertEqual(metrics.entry_metric_values(ctx)["needle"], 1.0)
        faux = dict(ctx, needle=False)
        self.assertEqual(metrics.entry_metric_values(faux)["needle"], 0.0)


# --------------------------------------------------------------- graphique
class TestSeriesForChart(unittest.TestCase):

    def test_une_entree_sans_la_metrique_est_ignoree(self):
        entries = [ent(pp=100.0, score_pct=42.0)]
        data = metrics.series_for_chart(entries, metric_key="score_pct")
        self.assertEqual(data["series"], [])
        self.assertEqual(data["groups"], [])
        data = metrics.series_for_chart(entries, metric_key="pp")
        self.assertEqual(len(data["series"]), 1)
        self.assertEqual(data["series"][0]["points"][0]["value"], 100.0)

    def test_un_point_par_serie_groupe_le_plus_recent_gagne(self):
        vieux = ent(pp=100.0, date="2026-01-01T00:00:00", source="vieux.log")
        recent = ent(pp=200.0, date="2026-02-01T00:00:00", source="recent.log")
        data = metrics.series_for_chart([vieux, recent], metric_key="pp")
        self.assertEqual(len(data["series"]), 1, "meme serie attendue")
        points = data["series"][0]["points"]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["value"], 200.0)
        self.assertEqual(points[0]["entry_id"], metrics.entry_id(recent))
        self.assertEqual(len(data["duplicates"]), 1)
        doublon = data["duplicates"][0]
        self.assertEqual(doublon["kept"], metrics.entry_id(recent))
        self.assertEqual(doublon["dropped"], [metrics.entry_id(vieux)])
        self.assertEqual(doublon["group"], points[0]["group"])

    def test_doublon_ordre_inverse_donne_le_meme_resultat(self):
        vieux = ent(pp=100.0, date="2026-01-01T00:00:00", source="vieux.log")
        recent = ent(pp=200.0, date="2026-02-01T00:00:00", source="recent.log")
        a = metrics.series_for_chart([vieux, recent], metric_key="pp")
        b = metrics.series_for_chart([recent, vieux], metric_key="pp")
        self.assertEqual(a["series"][0]["points"], b["series"][0]["points"])

    def test_deux_variantes_mtp_font_deux_series(self):
        on = ent(pp=150.0, method="bench 128 tok (draft MTP)", source="a.log")
        off = ent(pp=100.0, method="bench 128 tok (nomtp)", source="b.log")
        data = metrics.series_for_chart([on, off], metric_key="pp")
        self.assertEqual(len(data["series"]), 2)
        for serie in data["series"]:
            self.assertEqual(len(serie["points"]), 1)

    def test_structure_du_contrat(self):
        data = metrics.series_for_chart(vault_entries(), metric_key="pp")
        self.assertEqual(set(data),
                         {"metric", "groups", "series", "duplicates"})
        self.assertEqual(set(data["metric"]), {"key", "label", "unit", "better"})
        for groupe in data["groups"]:
            self.assertEqual(set(groupe), {"key", "label"})
        for serie in data["series"]:
            self.assertEqual(set(serie), {"key", "label", "hidden", "points"})
            self.assertIsInstance(serie["hidden"], bool)
            for point in serie["points"]:
                self.assertEqual(set(point), {"group", "value", "duration_s",
                                              "entry_id", "date", "status"})

    def test_points_du_vault_sur_pp(self):
        entries = vault_entries()
        data = metrics.series_for_chart(entries, metric_key="pp")
        par_id = {metrics.entry_id(e): e for e in entries}
        total = 0
        for serie in data["series"]:
            for point in serie["points"]:
                total += 1
                source_entry = par_id[point["entry_id"]]
                self.assertEqual(point["value"], float(source_entry["pp"]))
                self.assertEqual(point["group"], "vitesse-128k")
                self.assertIsInstance(point["duration_s"], (float, int, type(None)))
                self.assertEqual(point["status"], "ok")
        self.assertEqual(total, len({p["entry_id"] for s in data["series"]
                                     for p in s["points"]}))
        self.assertGreaterEqual(len(data["groups"]), 1)

    def test_score_pct_ne_melange_pas_les_familles(self):
        entries = vault_entries()
        par_id = {metrics.entry_id(e): e for e in entries}
        data = metrics.series_for_chart(entries, metric_key="score_pct")
        vus = {par_id[p["entry_id"]]["kind"] for s in data["series"] for p in s["points"]}
        self.assertEqual(vus, {"humaneval"})
        self.assertTrue(all(0 <= p["value"] <= 100
                            for s in data["series"] for p in s["points"]))

    def test_filtre_par_kind(self):
        entries = vault_entries()
        data = metrics.series_for_chart(entries, metric_key="score_avg", kind="battery")
        par_id = {metrics.entry_id(e): e for e in entries}
        for serie in data["series"]:
            for point in serie["points"]:
                self.assertEqual(par_id[point["entry_id"]]["kind"], "battery")
        self.assertTrue(data["series"])

    def test_metrique_inconnue(self):
        data = metrics.series_for_chart(extract_entries(), metric_key="bidon")
        self.assertEqual(data["series"], [])
        self.assertEqual(data["metric"]["key"], "bidon")
        self.assertEqual(data["metric"]["unit"], "")

    def test_serie_masquee(self):
        a = ent(pp=1.0, hidden=True, source="a.log")
        data = metrics.series_for_chart([a], metric_key="pp")
        self.assertTrue(data["series"][0]["hidden"])
        data = metrics.series_for_chart([ent(pp=1.0, source="b.log")], metric_key="pp")
        self.assertFalse(data["series"][0]["hidden"])

    def test_groupes_tries_et_libelles_fr(self):
        data = metrics.series_for_chart(vault_entries(), metric_key="score_pct")
        cles = [g["key"] for g in data["groups"]]
        # ordre stable : famille, puis taille croissante, puis cle
        self.assertEqual(cles, ["humaneval-5", "humaneval-23", "humaneval-38",
                                "humaneval-50", "humaneval-164"])
        self.assertEqual(data["groups"][0]["label"], "HumanEval · 5 problèmes")
        for groupe in data["groups"]:
            self.assertTrue(groupe["label"].strip())
        # le meme appel redonne exactement le meme ordre
        self.assertEqual([g["key"] for g in
                          metrics.series_for_chart(vault_entries(),
                                                   metric_key="score_pct")["groups"]],
                         cles)

    def test_groupes_des_benches_de_l_app(self):
        entrees = [
            ent(kind="vitesse", test_id="vitesse-256k", pp_tps=899.2, duree_s=275.2,
                source="benches/x/bench.json#vitesse-256k"),
            ent(kind="memoire", test_id="memoire-128k", prefill_tps=1171.0,
                duree_s=104.0, source="benches/x/bench.json#memoire-128k"),
            ent(kind="intelligence", test_id="batterie-13", score_avg=0.846,
                source="benches/x/bench.json#batterie-13"),
        ]
        data = metrics.series_for_chart(entrees, metric_key="pp_tps")
        self.assertEqual([g["key"] for g in data["groups"]], ["vitesse-256k"])
        data = metrics.series_for_chart(entrees, metric_key="score_avg")
        self.assertEqual([g["key"] for g in data["groups"]], ["batterie FR"])
        self.assertEqual(data["series"][0]["points"][0]["value"], 0.846)

    def test_entrees_vides_ou_invalides(self):
        self.assertEqual(metrics.series_for_chart([], metric_key="pp")["series"], [])
        self.assertEqual(metrics.series_for_chart(None, metric_key="pp")["series"], [])
        self.assertEqual(metrics.normalize_entries(None), [])
        self.assertEqual(metrics.normalize_entries([None, 3, "x"]), [])


if __name__ == "__main__":
    unittest.main()
