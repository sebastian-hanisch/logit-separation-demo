"""Presets: Vollständigkeit, gültige Werte, das erste Preset gleich den Standardeinstellungen, Permalink-Begrenzung, und je Preset die erwartete Verdikt-Zeile (§11a Punkt 7)."""

import numpy as np
import pytest

import lsp_constants as C
import lsp_evaluation as ev
import lsp_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.SETTING_SPECS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_allowed_values():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_defaults_are_allowed_values():
    for key, spec in P.SETTING_SPECS.items():
        assert P._begrenzen(spec, spec.default) == spec.default, key


def test_first_preset_is_the_default_settings_and_all_presets_differ():
    voll = [{k: p.get(k, spec.default) for k, spec in P.SETTING_SPECS.items()} for p in C.PRESETS.values()]
    assert voll[0] == {k: spec.default for k, spec in P.SETTING_SPECS.items()}
    assert len({tuple(sorted(v.items(), key=lambda kv: kv[0])) for v in voll}) == len(voll), "jedes Preset unterscheidet sich von jedem anderen (sonst wäre eines nie hervorgehoben)"


def test_url_params_are_unique_and_state_keys_equal_preset_keys():
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_permalink_begrenzt_und_verwirft_unerlaubte_werte():
    assert P._begrenzen(P.SETTING_SPECS["e1_steps"], "99") == C.E1_STEPS_MAX
    assert P._begrenzen(P.SETTING_SPECS["e1_steps"], "-5") == C.E1_STEPS_MIN
    assert P._begrenzen(P.SETTING_SPECS["e2_steps"], "1000") == C.E2_STEPS_MAX
    assert P._begrenzen(P.SETTING_SPECS["e3_n"], "100") == 100 and P._begrenzen(P.SETTING_SPECS["e3_n"], "101") is None
    assert P._begrenzen(P.SETTING_SPECS["e2_skala"], "0.5") == 0.5 and P._begrenzen(P.SETTING_SPECS["e2_skala"], "3") is None
    assert P._begrenzen(P.SETTING_SPECS["e2_regel"], "unbekannt") is None
    assert P._begrenzen(P.SETTING_SPECS["e3_n"], "abc") is None
    assert P._begrenzen(P.SETTING_SPECS["seed_input"], "123456") == C.SEED_MAX


def _verdikt_e3(preset):
    p = C.PRESETS[preset]
    fz = ev.load_frozen()
    n = p.get("e3_n", P.SETTING_SPECS["e3_n"].default)
    noise = C.E3_NOISE[p.get("e3_streuung", C.E3_STREUUNG[0])]
    return ev.verdict_exp3(ev.haeufigkeit_zeile(fz, n, noise), fz["haeufigkeit_zweiter_strom"], noise)


def test_preset_container_n_1000_meldet_keine_getrennte_stichprobe_und_keine_wirkung():
    """Standardstreuung, n = 1000: 0 getrennte Stichproben in den vorgerechneten Stichproben; das Verdikt darf keine Häufigkeit melden (Art info, Null-Satz)."""
    kind, text = _verdikt_e3("Container n = 1000")
    assert kind == "info" and "waren 0 von 200 Stichproben getrennt" in text and "Null getrennte Stichproben von 200" in text


def test_preset_container_n_100_und_halbe_streuung_melden_eine_haeufigkeit_ueber_zwanzig_prozent():
    for name in ("Container n = 100", "Halbe Streuung"):
        kind, text = _verdikt_e3(name)
        assert kind == "warning", name
    assert "bei halber Streuung" in _verdikt_e3("Halbe Streuung")[1]


def test_preset_trennung_und_quasi_melden_trennung_aber_von_hand_nicht():
    for name, art in (("Trennung (6 Punkte)", "vollständig"), ("Quasi-Trennung (8 Punkte)", "quasi")):
        p = C.PRESETS[name]
        r = ev.exp2(p["e2_daten"], 1.0, p["e2_steps"], C.E2_REGELN[0])
        assert r["art"] == art and ev.verdict_exp2(r)[0] == "warning"
    r1 = ev.exp1(C.E1_BEISPIELE[0])
    assert np.isfinite(r1["pfad"]).all() and r1["k_konv"] is not None, "das Handbeispiel hat einen Schätzer: keine Trennungswarnung"


def test_preset_viele_merkmale_meldet_die_aufblaehung():
    z = ev.faustregel_zeile(ev.load_frozen(), C.PRESETS["Viele Merkmale (p/n = 0,2)"]["e5_pr"])
    kind, text = ev.verdict_exp5(z)
    assert kind == "warning" and z["wald_ablehnung"] > 0.05 and z["sd_ueber_se"] > 1
    z0 = ev.faustregel_zeile(ev.load_frozen(), 0)
    assert ev.verdict_exp5(z0)[0] == "info", "ohne Rauschmerkmale keine Warnung"
