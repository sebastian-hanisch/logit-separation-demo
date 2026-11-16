"""Handrechnung der Einheiten (Mini-Instanzen von Hand, in Brüchen mit `fractions` oder in geschlossener Form): IRLS, Wald, Abbruchregeln, Firth, Cover.

Erwartungswerte stehen als Rechnung im Test, nicht als abgeschriebene Ausgabe des Codes. Alle Zahlen sind eigene Nachrechnung in diesem Modell, bei diesen gesetzten Daten."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest
from scipy import stats

import lsp_constants as C
import lsp_logit as L
import lsp_scenario as S
import lsp_separation as P


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Neun Container, Zollhalt
# ---------------------------------------------------------------------------------------------------------------------------------------------
def test_erster_schritt_in_bruechen():
    """Bei β = 0 ist p = 1/2, w = 1/4, z = 4y - 2. (X'X) = [[9, 5], [5, 5]], X'(y - 1/2) = (1/2, 3/2): β = 4 (X'X)^-1 X'(y - 1/2) = (-1, 11/5)."""
    assert L.erster_schritt_bruch([[1, x] for x in C.ZOLL9_X], list(C.ZOLL9_Y)) == [Fr(-1), Fr(11, 5)]


def test_erster_schritt_gleitkomma_gleich_bruch(zoll9):
    X, y = zoll9
    assert L.irls_pfad(X, y, 1)[1] == pytest.approx([-1.0, 2.2], abs=1e-14)


def test_grenzwert_geschlossene_form(zoll9):
    """Gruppenanteile 1/4 und 4/5 (Score null: 1 - 4 (1/4) = 0, 4 - 5 (4/5) = 0), Odds 1/3 und 4, Odds-Ratio 12."""
    f = L.zwei_gruppen_grenzwert(4, 1, 5, 4)
    assert (f["p0"], f["p1"]) == (Fr(1, 4), Fr(4, 5))
    assert (f["odds0"], f["odds1"], f["ratio"]) == (Fr(1, 3), Fr(4), Fr(12))
    assert f["beta"] == pytest.approx((math.log(1 / 3), math.log(12)), abs=1e-15)
    X, y = zoll9
    b, k = L.grenzwert(X, y)
    assert b == pytest.approx(f["beta"], abs=1e-11)
    assert 5 <= k <= 7, "6 Schritte, Band ±1 (letzte Stelle plattformabhängig)"


def test_irls_pfad_gleich_newton_mit_numerischer_hesse(zoll9):
    X, y = zoll9
    assert np.max(np.abs(L.irls_pfad(X, y, 6) - L.newton_pfad(X, y, 6))) < 1e-7


def test_quadratische_konvergenz(zoll9):
    X, y = zoll9
    pfad = L.irls_pfad(X, y, 6)
    g = np.array([math.log(1 / 3), math.log(12)])
    fehler = [float(np.max(np.abs(b - g))) for b in pfad]
    quot = [fehler[i + 1] / fehler[i] ** 2 for i in range(len(fehler) - 1) if fehler[i + 1] > 1e-9]
    assert len(quot) >= 2 and max(quot) < 10


def test_information_wald_und_likelihood_quotient_in_bruechen(zoll9):
    """Gewichte p (1 - p) = 3/16 und 4/25; Information [[31/20, 4/5], [4/5, 4/5]] mit Determinante 3/5; Var(β₁) = 31/12; Wald z = ln 12 / sqrt(31/12) = 1,546."""
    w0, w1 = Fr(1, 4) * Fr(3, 4), Fr(4, 5) * Fr(1, 5)
    info = [[4 * w0 + 5 * w1, 5 * w1], [5 * w1, 5 * w1]]
    det = info[0][0] * info[1][1] - info[0][1] ** 2
    assert info == [[Fr(31, 20), Fr(4, 5)], [Fr(4, 5), Fr(4, 5)]] and det == Fr(3, 5)
    assert info[0][0] / det == Fr(31, 12)
    X, y = zoll9
    b, _ = L.grenzwert(X, y)
    r = L.wald_und_lq(X, y, b)
    assert r["se"][1] == pytest.approx(math.sqrt(31 / 12), abs=1e-9)
    assert r["z"][1] == pytest.approx(math.log(12) / math.sqrt(31 / 12), abs=1e-9)
    assert r["p_wald"][1] == pytest.approx(0.122, abs=5e-4)
    assert r["lq"] == pytest.approx(2.863, abs=5e-4) and r["p_lq"] == pytest.approx(0.091, abs=5e-4)
    assert L.deviance_null(y) == pytest.approx(12.365, abs=5e-4) and L.deviance(X, y, b) == pytest.approx(9.503, abs=5e-4)


def test_gerade_im_vorlauf_beispiel_in_bruechen():
    """x = 1 bis 8, y = 0, 0, 0, 1, 0, 1, 1, 1: Steigung Σ(x - 4,5) y / Σ(x - 4,5)² = 7/42 = 1/6, Achsenabschnitt 1/2 - 4,5/6 = -1/4; bei x = 1 der Wert -1/12, bei x = 8 der Wert 13/12."""
    x, y = S.handbeispiel(C.E1_BEISPIELE[1])
    b = L.gerade_ols(L.design_1d(x), y)
    assert b == pytest.approx([-0.25, 1 / 6], abs=1e-12)
    pred = L.design_1d(x) @ b
    assert pred[0] == pytest.approx(-1 / 12, abs=1e-12) and pred[-1] == pytest.approx(13 / 12, abs=1e-12)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Trennung: sechs Punkte, acht Punkte
# ---------------------------------------------------------------------------------------------------------------------------------------------
def test_sechs_punkte_erster_schritt_in_bruechen(sechs):
    """Bei β = 0: Information diag(6/4, 14/4 · 2) = diag(3/2, 7), Score (0, 6) (Σ x (y - 1/2) = 6): β = (0, 6/7). Deviance 12 ln 2."""
    X, y = sechs
    assert L.erster_schritt_bruch([[1, int(v)] for v in C.SEP6_X], list(C.SEP6_Y)) == [Fr(0), Fr(6, 7)]
    assert L.deviance(X, y, np.zeros(2)) == pytest.approx(12 * math.log(2), abs=1e-12)


def test_wachstum_je_schritt_ist_eins_ohne_kappung(sechs):
    X, y = sechs
    pfad = L.irls_pfad(X, y, 120)
    zuw = np.diff(pfad[:, 1])
    assert zuw[39:100].min() > 0.999999 and zuw[39:100].max() < 1.000001
    assert np.max(np.abs(pfad[:31, 0])) < 1e-9
    assert np.all(np.isfinite(pfad))


@pytest.mark.parametrize("faktor", [0.5, 1.0, 2.0])
def test_zuwachs_ist_eins_durch_kleinstes_abs_x(faktor):
    X = L.design_1d(np.array(C.SEP6_X, float) * faktor)
    pfad = L.irls_pfad(X, np.array(C.SEP6_Y, float), 70)
    z = np.diff(pfad[:, 1])[30:60]
    assert z.min() > 1 / faktor - 1e-6 and z.max() < 1 / faktor + 1e-6


def test_stuetzpunkte_asymmetrisch():
    """x = -3, -2, -1, 2, 3, 4: Stützpunkte -1 (y = 0) und 2 (y = 1). Je Schritt wächst der Abstand beider um 1: -(d0 - d1) = 1 und d0 + 2 d1 = 1, also (d0, d1) = (-1/3, 2/3)."""
    sol = L.loese_bruch([[1, 2], [-1, 1]], [1, 1])
    assert sol == [Fr(-1, 3), Fr(2, 3)]
    X = L.design_1d(np.array([-3, -2, -1, 2, 3, 4], float))
    pfad = L.irls_pfad(X, np.array([0, 0, 0, 1, 1, 1], float), 80)
    assert np.allclose(np.diff(pfad, axis=0)[40:70], [-1 / 3, 2 / 3], atol=1e-6)
    assert abs(-pfad[70, 0] / pfad[70, 1] - 0.5) < 0.01


def test_quasi_trennung_vollstaendige_pruefung_sieht_sie_nicht(quasi8):
    X, y = quasi8
    assert P.ist_getrennt(X, y) and not P.ist_vollstaendig_getrennt(X, y)
    assert P.art_der_trennung(X, y) == "quasi"
    pfad = L.irls_pfad(X, y, 60)
    assert pfad[40, 1] > 30 and abs(pfad[40, 0]) < 1e-6


def test_ueberlappung_nach_umdrehen_eines_punktes(quasi8):
    X, y = quasi8
    y2 = y.copy()
    y2[2] = 1.0                                   # x = -1 mit y = 1: größtes x mit y = 0 ist 0, kleinstes mit y = 1 ist -1
    assert not P.ist_getrennt(X, y2)
    b, schritte, konv = L.ml_konvergiert(X, y2)
    assert konv and schritte < 60 and np.max(np.abs(b)) < 5


def test_wald_gegen_likelihood_quotient_bei_trennung(sechs):
    """Wald z fällt nach dem Höchstwert gegen 0; der Likelihood-Quotient wächst gegen 12 ln 2 = 8,318 (p = 0,0039)."""
    X, y = sechs
    pfad = L.irls_pfad(X, y, 80)
    r1, r80 = L.wald_und_lq(X, y, pfad[1]), L.wald_und_lq(X, y, pfad[80])
    assert r1["z"][1] == pytest.approx(1.39, abs=0.01)
    assert abs(r80["z"][1]) < 1e-6
    assert r80["lq"] == pytest.approx(12 * math.log(2), abs=1e-6) and r80["p_lq"] == pytest.approx(stats.chi2.sf(12 * math.log(2), 1), abs=1e-9)
    zmax = max(abs(L.wald_und_lq(X, y, b)["z"][1]) for b in pfad[:60])
    assert zmax < 1.45


def test_abbruchregeln_auf_den_sechs_punkten(sechs):
    X, y = sechs
    pfad = L.irls_pfad(X, y, 200)
    dev = L.devianzen(X, y, pfad)
    k, greift = L.abbruch_schritt(pfad[:26], dev[:26], "Deviance-Regel")
    assert greift and 23 <= k <= 25 and pfad[k, 1] == pytest.approx(22.96, abs=0.05)
    k2, g2 = L.abbruch_schritt(pfad[:26], dev[:26], "Feste Schrittzahl")
    assert (k2, g2) == (25, True) and pfad[25, 1] == pytest.approx(23.96, abs=0.005)
    assert pfad[100, 1] == pytest.approx(98.96, abs=0.005) and pfad[200, 1] == pytest.approx(198.96, abs=0.005)
    k3, g3 = L.abbruch_schritt(pfad, dev, "Koeffizientenänderung")
    assert (k3, g3) == (200, False), "die Koeffizientenregel greift in 200 Schritten nie (der Koeffizient ändert sich je Schritt um 1)"


def test_gekappter_nachbau_haelt_bei_schritt_24(sechs):
    X, y = sechs
    b, k, regulaer = L.irls_gekappt(X, y)
    assert regulaer and 23 <= k <= 25 and b[1] == pytest.approx(22.956, abs=0.02)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Firth
# ---------------------------------------------------------------------------------------------------------------------------------------------
def test_firth_zwei_gruppen_ein_halbes_in_jede_zelle():
    """0 von 5 gegen 3 von 5: p_g = (s_g + 1/2) / (n_g + 1) = 1/12 und 7/12; Odds 1/11 und 7/5; Odds-Ratio 77/5."""
    f = L.firth_zwei_gruppen(5, 0, 5, 3)
    assert (f["p0"], f["p1"]) == (Fr(1, 12), Fr(7, 12)) and (f["odds0"], f["odds1"], f["ratio"]) == (Fr(1, 11), Fr(7, 5), Fr(77, 5))
    X = np.column_stack([np.ones(10), np.array([0] * 5 + [1] * 5, float)])
    y = np.array([0] * 5 + [1, 1, 1, 0, 0], float)
    assert P.ist_getrennt(X, y)                   # Nullzelle: kein ML-Schätzer
    b, _, konv = L.firth(X, y, tol=1e-13)
    assert konv and b == pytest.approx([math.log(1 / 11), math.log(77 / 5)], abs=1e-9)


def test_firth_bei_den_sechs_punkten_ist_endlich(sechs):
    X, y = sechs
    b, _, konv = L.firth(X, y, tol=1e-13)
    assert konv and abs(b[0]) < 1e-9 and b[1] == pytest.approx(0.9384, abs=1e-3)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Cover-Formel durch Aufzählung
# ---------------------------------------------------------------------------------------------------------------------------------------------
def test_cover_formel_gleich_aufzaehlung():
    X6 = np.column_stack([np.ones(6), [0.3, 1.1, 1.9, 2.4, 3.7, 5.2]])
    assert P.aufzaehlung_trennbar(X6) == 12 and P.cover_anteil(6, 2) == Fr(12, 64)
    X6b = np.column_stack([np.ones(6), np.random.default_rng(7).standard_normal((6, 2))])
    assert P.aufzaehlung_trennbar(X6b) == 32 and P.cover_anteil(6, 3) == Fr(32, 64)


def test_cover_randwerte():
    assert P.cover_anteil(5, 5) == 1 and P.cover_anteil(10, 5) == Fr(1, 2) and P.cover_anteil(14, 7) == Fr(1, 2)
    assert P.cover_anteil(10, 2) == Fr(20, 1024)
