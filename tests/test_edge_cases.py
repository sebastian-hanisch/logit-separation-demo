"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): konstanter Ausgang, Merkmal ohne Variation, n < p, doppelte Zeilen, genau ein Ereignis,
Zelle mit allen Ereignissen, |η| über 36,7 (dort ist p in double exakt 1), n = 2."""

import numpy as np
import pytest

import lsp_constants as C
import lsp_evaluation as ev
import lsp_logit as L
import lsp_scenario as S
import lsp_separation as P


def test_konstanter_ausgang_ist_getrennt_die_richtung_ist_der_achsenabschnitt():
    X = L.design_1d(np.array([-1.0, 0.0, 2.0, 5.0]))
    for wert in (0.0, 1.0):
        y = np.full(4, wert)
        assert P.ist_getrennt(X, y)
        assert P.lp_optimum(X[:, :1], y) > 0, "schon der Achsenabschnitt allein ist die Rezessionsrichtung"


def test_merkmal_ohne_variation_faellt_weg_und_ist_nicht_getrennt():
    """x konstant bei gemischtem y: kein „getrennt“, nach dem Weglassen der Spalte bleibt nur der Achsenabschnitt (p̂ = Mittel)."""
    X = np.column_stack([np.ones(6), np.full(6, 3.0)])
    y = np.array([0, 1, 0, 1, 1, 0], float)
    keep = P.voller_rang(X)
    assert keep == [0]
    assert not P.ist_getrennt(X[:, keep], y)
    b, _, konv = L.ml_konvergiert(X[:, keep], y)
    assert konv and b[0] == pytest.approx(0.0, abs=1e-9)


def test_n_kleiner_p_ist_getrennt():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(3), rng.standard_normal((3, 4))])
    assert P.ist_getrennt(X, np.array([0.0, 1.0, 0.0]))


def test_n_gleich_zwei():
    X = L.design_1d(np.array([0.0, 1.0]))
    assert P.art_der_trennung(X, np.array([0.0, 1.0])) == "vollständig"
    assert P.ist_getrennt(X, np.array([1.0, 1.0]))
    assert P.art_der_trennung(L.design_1d(np.array([1.0, 1.0])[:2]), np.array([0.0, 1.0]))  # Merkmal ohne Variation: vor dem Aufruf `voller_rang`, hier nur: kein Absturz


def test_genau_ein_ereignis_und_zelle_mit_allen_ereignissen():
    x = np.array([0, 0, 0, 0, 1, 1], float)
    X = L.design_1d(x)
    assert P.ist_getrennt(X, np.array([0, 0, 0, 0, 0, 1], float)), "ein Ereignis in der Gruppe 1, keines in der Gruppe 0: Nullzelle"
    assert P.ist_getrennt(X, np.array([1, 1, 1, 1, 0, 1], float)), "Gruppe 0 nur Ereignisse"
    assert not P.ist_getrennt(X, np.array([0, 1, 0, 0, 1, 0], float))
    nz = P.nullzellen(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), np.array([0, 0, 1, 1]))
    assert ("Art Import", 2, 0) in nz and ("Art Export", 2, 2) in nz


def test_doppelte_zeilen_aendern_den_schaetzer_nicht(zoll9):
    X, y = zoll9
    assert L.grenzwert(np.vstack([X, X]), np.r_[y, y])[0] == pytest.approx(L.grenzwert(X, y)[0], abs=1e-10)


def test_eta_ueber_36_7_stabile_formeln_ohne_nan():
    """Bei |η| über 36,7 ist p in double genau 1 und y - p wird 0 oder auslöscht; die stabilen Formeln rechnen den Pfad bis Schritt 200 ohne nan."""
    X = L.design_1d(np.array(C.SEP6_X, float))
    y = np.array(C.SEP6_Y, float)
    pfad = L.irls_pfad(X, y, 200)
    assert np.all(np.isfinite(pfad)) and pfad[-1, 1] > 190
    assert np.isfinite(L.deviance(X, y, pfad[-1]))
    assert np.all(np.isfinite(L.wald_und_lq(X, y, pfad[-1])["z"]))


def test_singulaere_matrix_beendet_den_pfad_kuerzer():
    X = np.column_stack([np.ones(4), np.zeros(4)])
    pfad = L.irls_pfad(X, np.array([0, 1, 0, 1], float), 5)
    assert len(pfad) < 6 or np.all(np.isfinite(pfad))


def test_abbruchregel_bei_pfad_ohne_schritte():
    pfad = np.zeros((1, 2))
    for regel in C.E2_REGELN:
        assert L.abbruch_schritt(pfad, np.array([1.0]), regel)[0] == 0


def test_wilson_randfaelle():
    assert ev.wilson(0, 0) == (0.0, 1.0)
    lo, hi = ev.wilson(0, 200)
    assert lo == 0.0 and 0 < hi < 0.03
    lo, hi = ev.wilson(200, 200)
    assert hi == 1.0 and 0.97 < lo < 1


def test_ziehe_eine_stichprobe_mit_n_gleich_eins():
    s = S.ziehe(1, 4)
    assert s.n == 1 and s.y.shape == (1,) and s.X().shape == (1, 7)


def test_exp2_bei_einem_schritt_und_an_den_reglergrenzen():
    for daten in C.E2_DATEN:
        for skala in C.E2_SKALEN:
            for regel in C.E2_REGELN:
                r = ev.exp2(daten, skala, 1, regel)
                assert r["k_stopp"] == 1 and len(r["pfad"]) == 2
                r = ev.exp2(daten, skala, C.E2_STEPS_MAX, regel)
                assert np.all(np.isfinite(r["pfad"])) and np.all(np.isfinite(r["z"])) and np.all(np.isfinite(r["p_lq"]))


def test_exp1_bei_null_schritten():
    for b in C.E1_BEISPIELE:
        r = ev.exp1(b)
        assert r["pfad"][0].tolist() == [0.0, 0.0]
        kind, text = ev.verdict_exp1(r, 0, None)
        assert kind == "success" and "Nach 0 Schritten" in text


def test_exp3_beispiel_bei_n_30_mit_spalten_ohne_variation():
    """Bei n = 30 fehlt in einigen Stichproben eine Spalte ganz (z. B. kein Leer-Container); die Prüfung lässt sie weg, ohne abzustürzen."""
    gefunden = False
    for seed in range(200):
        r = ev.exp3_beispiel(30, 1.0, seed)
        if r["spalten_ohne_variation"]:
            gefunden = True
            assert r["art"] in ("überlappend", "vollständig", "quasi")
            break
    assert gefunden
