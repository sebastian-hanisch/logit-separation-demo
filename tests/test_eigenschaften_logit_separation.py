"""Eigenschaftstests (metamorph, ORACLE-PLAYBOOK §5.2): Beziehungen, die ohne Kenntnis des richtigen Ergebnisses gelten müssen. Je Eigenschaft 20 Zufallsinstanzen mit festem Zufalls-Seed.

Jede Eigenschaft ist eine belegte Aussage über das Logit (ML, wo er existiert), nicht nur das heutige Verhalten des Codes."""

import numpy as np
import pytest

import lsp_logit as L
import lsp_scenario as S
import lsp_separation as P


def _instanzen(anzahl=20, n=60, seed0=1000):
    """Kleine Instanzen: Merkmale N(0, 1) mit zwei Spalten, Ausgang aus einem logistischen Modell, nur nicht getrennte."""
    rng = np.random.default_rng(seed0)
    out = []
    while len(out) < anzahl:
        x = rng.standard_normal((n, 2))
        eta = 0.3 + x @ np.array([0.8, -0.5])
        y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
        X = np.column_stack([np.ones(n), x])
        if not P.ist_getrennt(X, y):
            out.append((X, y))
    return out


def _ml(X, y):
    return L.ml_konvergiert(X, y, tol=1e-12)[0]


def test_vertauschen_der_beobachtungen_aendert_nichts():
    rng = np.random.default_rng(1)
    for X, y in _instanzen():
        perm = rng.permutation(len(y))
        assert _ml(X[perm], y[perm]) == pytest.approx(_ml(X, y), abs=1e-8)


def test_spiegeln_des_ausgangs_kehrt_beta_um():
    for X, y in _instanzen():
        assert _ml(X, 1 - y) == pytest.approx(-_ml(X, y), abs=1e-8)


def test_verschieben_eines_merkmals_aendert_nur_den_achsenabschnitt():
    for X, y in _instanzen():
        b = _ml(X, y)
        X2 = X.copy()
        X2[:, 1] += 2.5
        b2 = _ml(X2, y)
        assert b2[1:] == pytest.approx(b[1:], abs=1e-7)
        assert b2[0] == pytest.approx(b[0] - 2.5 * b[1], abs=1e-7)


def test_skalieren_eines_merkmals_skaliert_den_koeffizienten_mit_eins_durch_c():
    for X, y in _instanzen():
        b = _ml(X, y)
        X2 = X.copy()
        X2[:, 2] *= 4.0
        b2 = _ml(X2, y)
        assert b2[2] == pytest.approx(b[2] / 4.0, abs=1e-8) and b2[:2] == pytest.approx(b[:2], abs=1e-8)


def test_der_score_ist_am_ergebnis_null():
    for X, y in _instanzen():
        b = _ml(X, y)
        assert np.max(np.abs(X.T @ (y - 1 / (1 + np.exp(-(X @ b)))))) < 1e-8


def test_verdoppeln_aller_zeilen_aendert_beta_nicht_und_teilt_den_standardfehler_durch_wurzel_zwei():
    for X, y in _instanzen(10):
        b, b2 = _ml(X, y), _ml(np.vstack([X, X]), np.r_[y, y])
        assert b2 == pytest.approx(b, abs=1e-8)
        se1, se2 = L.wald_und_lq(X, y, b)["se"], L.wald_und_lq(np.vstack([X, X]), np.r_[y, y], b2)["se"]
        assert se2 == pytest.approx(se1 / np.sqrt(2), rel=1e-8)


def test_ridge_grenzwerte():
    """λ → 0 gibt ML, λ → ∞ gibt Koeffizienten → 0 (der Achsenabschnitt bleibt frei)."""
    for X, y in _instanzen(10):
        b = _ml(X, y)
        assert L.ridge(X, y, 1e-9) == pytest.approx(b, abs=1e-5)
        gross = L.ridge(X, y, 1e9)
        assert np.max(np.abs(gross[1:])) < 1e-6


def test_deviance_nimmt_in_diesen_instanzen_ab_schritt_zwei_in_jedem_schritt_ab():
    """Newton ohne Halbierung muss die Deviance nicht in jedem Schritt senken; bei diesen gutartigen Instanzen (n = 60, 3 Spalten) nimmt sie ab Schritt 2 monoton ab (gemessene Eigenschaft dieser 20 Instanzen)."""
    for X, y in _instanzen():
        d = L.devianzen(X, y, L.irls_pfad(X, y, 8))
        assert np.all(np.diff(d[1:]) <= 1e-10)


def test_firth_schaetzer_sind_betragsmaessig_kleiner_als_ml_in_diesen_instanzen():
    """Aussage nach Kosmidis und Firth („in der Regel“); hier nur als gemessene Eigenschaft dieser 20 Instanzen: der größte |β| bei Firth ist nicht größer als bei ML."""
    for X, y in _instanzen():
        assert np.max(np.abs(L.firth(X, y)[0])) <= np.max(np.abs(_ml(X, y))) + 1e-9


def test_gleicher_zufalls_seed_gleiche_stichprobe():
    a, b = S.ziehe(80, 5, 1.0), S.ziehe(80, 5, 1.0)
    assert np.array_equal(a.dwell, b.dwell) and np.array_equal(a.y, b.y)
    c = S.ziehe(80, 6, 1.0)
    assert not np.array_equal(a.dwell, c.dwell)


def test_die_ersten_container_haengen_nicht_von_n_ab():
    """Keine Zukunftsinformation: die ersten 30 Container einer Stichprobe aus 100 sind die einer aus 30."""
    a, b = S.ziehe(100, 9, 1.0), S.ziehe(30, 9, 1.0)
    assert np.array_equal(a.art[:30], b.art) and np.allclose(a.dwell[:30], b.dwell, rtol=1e-12)
