"""Claims: jede Zahl der README wird hier nachgerechnet oder gegen `data/lsp_frozen.json` geprüft (DEMO-PLAYBOOK §11a, Tabelle „Zahl → Test“ in der README).

Drei Arten von Prüfung:
- **exakt**: in Brüchen oder geschlossener Form (test_logit.py), hier nur ergänzend;
- **Band**: Neu-Rechnung mit Toleranz (Zählungen ± 2 je 40 Stichproben, Floats relativ 1e-4 bis 1e-9), weil Zufallsströme, HiGHS, BLAS und `numpy.exp` an der letzten Stelle plattformabhängig sind;
- **vorgerechnet**: die Zahl steht in der README und in `data/lsp_frozen.json`; `tools/lsp_make_frozen.py --check` rechnet Teile neu (`test_vorgerechnetes_stimmt_mit_neu_rechnung_als_band`).
Mit LSP_FULL=1 rechnet der letzte Test alle vorgerechneten Messungen ganz neu (einige Minuten)."""

import importlib.util
import json
import os
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import lsp_constants as C
import lsp_evaluation as ev
import lsp_logit as L
import lsp_scenario as S

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
FZ = ev.load_frozen()


def in_readme(*stuecke):
    for s in stuecke:
        assert s in README, f"steht nicht in der README: {s!r}"


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Handbeispiele
# ---------------------------------------------------------------------------------------------------------------------------------------------
def test_h1_newton_unterschied_und_wald_lq_fisher():
    in_readme("unter 1e-7 und über 1e-11", "z = 1,546 (p = 0,122)", "2,863 (p = 0,091)", "p = 0,206")
    r0, r1 = ev.exp1(C.E1_BEISPIELE[0]), ev.exp1(C.E1_BEISPIELE[1])
    assert 1e-11 < r0["diff_newton"] < 1e-7 and 1e-11 < r1["diff_newton"] < 1e-7
    # Nur das Band: der Differenzenquotient der Hesse-Matrix rundet je Plattform anders (Windows 3,3e-10, Linux 7,7e-10 beim Vorlauf).
    w = r0["wald"]
    assert w["z"][1] == pytest.approx(1.546, abs=5e-4) and w["p_wald"][1] == pytest.approx(0.122, abs=5e-4)
    assert w["lq"] == pytest.approx(2.863, abs=5e-4) and w["p_lq"] == pytest.approx(0.091, abs=5e-4)
    assert stats.fisher_exact([[4, 1], [1, 3]])[1] == pytest.approx(0.2063, abs=5e-4)


def test_h2_sechs_punkte_abbruchregeln_und_tests():
    in_readme("β₁ = 22,96", "23,96", "98,96", "198,96", "22,956", "bei Schritt 24 gleich 0,0003 (p = 1,00)", "(p = 0,0039)", "β₁ = 38,96 nach 40 Schritten")
    r = ev.exp2(C.E2_DATEN[0], 1.0, 200, C.E2_REGELN[0])
    k = r["stopp"]["Deviance-Regel"][0]
    assert 23 <= k <= 25 and r["pfad"][k, 1] == pytest.approx(22.96, abs=0.05)
    assert r["z"][24] == pytest.approx(0.0003, abs=5e-5) and r["p_wald"][24] > 0.999
    assert r["lq"][200] == pytest.approx(12 * np.log(2), abs=1e-9) and r["p_lq"][200] == pytest.approx(0.0039, abs=5e-5)
    q = ev.exp2(C.E2_DATEN[1], 1.0, 40, C.E2_REGELN[1])
    assert q["art"] == "quasi" and q["pfad"][40, 1] == pytest.approx(38.96, abs=0.05) and abs(q["pfad"][40, 0]) < 1e-6


def test_h3_skalierung_zuwachs():
    in_readme("2 / 1 / 0,5")
    for faktor, soll in ((0.5, 2.0), (1.0, 1.0), (2.0, 0.5)):
        r = ev.exp2(C.E2_DATEN[0], faktor, 70, C.E2_REGELN[1])
        assert r["formel_zuwachs"] == pytest.approx(soll)
        assert np.diff(r["pfad"][:, 1])[30:60] == pytest.approx(soll, abs=1e-6)


def test_scikit_learn_gibt_drei_verschiedene_werte_ohne_konvergenzwarnung():
    lin = pytest.importorskip("sklearn.linear_model")
    in_readme("β₁ = 8,35", "17,36", "27,06", "keine Konvergenzwarnung")
    x = np.array(C.SEP6_X, float).reshape(-1, 1)
    y = np.array(C.SEP6_Y, float)
    werte = []
    for it, tol in ((100, 1e-4), (1000, 1e-8), (100000, 1e-12)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                sk = lin.LogisticRegression(penalty=None, solver="lbfgs", max_iter=it, tol=tol).fit(x, y)
            except TypeError:
                pytest.skip("penalty=None wird in dieser scikit-learn-Version nicht mehr angenommen")
        assert not any("ConvergenceWarning" in type(m.message).__name__ for m in w), "keine Konvergenzwarnung"
        werte.append(float(sk.coef_[0, 0]))
    assert all(5 < v < 40 for v in werte) and werte[1] - werte[0] > 3 and werte[2] - werte[1] > 3
    import sklearn

    if sklearn.__version__ == "1.9.1":
        assert werte == pytest.approx([8.35, 17.36, 27.06], abs=0.6), "Band: die README nennt die Werte für Version 1.9.1; andere Versionen dürfen abweichen, solange die drei Werte verschieden bleiben"


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Vorgerechnet: Zahl in der README = Zahl in der Datei
# ---------------------------------------------------------------------------------------------------------------------------------------------
HAEUFIGKEIT_README = {
    (1.0, 30): (287, 300, 272, 280), (1.0, 60): (229, 300, 229, 216), (1.0, 100): (151, 300, 151, 147), (1.0, 200): (71, 300, 71, 71), (1.0, 500): (5, 200, 5, 5),
    (1.0, 1000): (0, 200, 0, 0), (1.0, 3000): (0, 100, 0, 0),
    (0.5, 30): (300, 300, 240, 298), (0.5, 60): (297, 300, 294, 292), (0.5, 100): (288, 300, 288, 284), (0.5, 200): (259, 300, 259, 256), (0.5, 500): (139, 200, 139, 139),
    (0.5, 1000): (105, 200, 105, 105), (0.5, 3000): (13, 100, 13, 13),
}


def test_haeufigkeit_zahlen_der_readme_gleich_der_datei():
    for (noise, n), (k, r, quasi, nz) in HAEUFIGKEIT_README.items():
        z = ev.haeufigkeit_zeile(FZ, n, noise)
        assert (z["getrennt"], z["reps"], z["quasi"], z["nullzelle"]) == (k, r, quasi, nz), (noise, n)
        if k:
            in_readme(f"| {k} von {r} (" if k else f"| {k} von {r} |")
        else:
            in_readme(f"| {k} von {r} |")
    in_readme("287 von 300 (95,7 %)", "151 von 300 (50,3 %)", "71 von 300 (23,7 %)", "5 von 200 (2,5 %)", "139 von 200 (69,5 %)", "105 von 200 (52,5 %)", "13 von 100 (13,0 %)", "288 von 300 (96,0 %)")


def test_haeufigkeit_ab_n_100_ist_jede_getrennte_stichprobe_nur_quasi():
    in_readme("Ab n = 100 ist jede getrennte Stichprobe nur quasi getrennt")
    for e in FZ["haeufigkeit"]:
        if e["n"] >= 100:
            assert e["quasi"] == e["getrennt"], (e["noise"], e["n"])


def test_haeufigkeit_spalte_ohne_variation_zweiter_strom_und_band():
    in_readme("in 32 von 300 Stichproben eine ganze Spalte", "bei n = 60 in 1 von 300", "136 von 300 (45,3 %)", "bei 1,9 %")
    assert ev.haeufigkeit_zeile(FZ, 30, 1.0)["konstant"] == 32 and ev.haeufigkeit_zeile(FZ, 60, 1.0)["konstant"] == 1
    z2 = FZ["haeufigkeit_zweiter_strom"]
    assert (z2["getrennt"], z2["reps"]) == (136, 300) and z2["stream"] != ev.haeufigkeit_zeile(FZ, 100, 1.0)["stream"]
    lo, hi = ev.wilson(0, 200)
    assert round(100 * hi, 1) == 1.9
    assert round(100 * np.sqrt(0.5 * 0.5 / 300), 1) == 2.9


def test_lp_luecke_in_600_stichproben():
    in_readme("mindestens 4,0 (365 Stichproben)", "null (235 Stichproben)")
    g = FZ["lp_luecke"]
    assert (g["stichproben"], g["getrennt"], g["nicht_getrennt"]) == (600, 365, 235)
    assert round(g["min_getrennt"], 1) == 4.0 and g["max_betrag_nicht_getrennt"] < 1e-12


def test_wahre_raten_je_art():
    in_readme("Import 23,3 %", "Export 25,9 %", "Umschlag 3,3 %", "Leer 50,5 %", "insgesamt 21,8 %")
    r = ev.wahre_rate_je_art()
    assert [round(100 * r[a], 1) for a in C.ART_LABELS] == [23.3, 25.9, 3.3, 50.5]
    assert round(100 * FZ["meta"]["standgeld_anteil_test"], 1) == 21.8


def test_hilfe_tabelle_der_readme_gleich_der_datei():
    erwartet = {
        60: {"ML, Abbruchregel": (0.711, 0), "ML, 25 Schritte": (0.982, 0), "Firth": (0.074, None), "Ridge 0,1": (0.105, 18), "Ridge 1": (0.092, 19), "Streichen": (0.170, 4)},
        100: {"ML, Abbruchregel": (0.184, 0), "ML, 25 Schritte": (0.241, 0), "Firth": (0.055, None), "Ridge 0,1": (0.067, 6), "Ridge 1": (0.071, 15), "Streichen": (0.110, 9)},
    }
    for n, tab in erwartet.items():
        rows = {r["verfahren"]: r for r in ev.exp4_tabelle(FZ, n, 150)}
        for m, (med, besser) in tab.items():
            assert round(rows[m]["median"], 3) == med, (n, m)
            assert rows[m]["besser"] == besser, (n, m)
            in_readme(ev.de(med, 3))
    w60, w100 = FZ["was_hilft"]["60"], FZ["was_hilft"]["100"]
    assert (w60["gezogen"], w60["getrennt"], w100["gezogen"], w100["getrennt"]) == (200, 150, 306, 150)
    in_readme("200 Stichproben (n = 60; 150 getrennt, 75,0 %)", "306 (n = 100; 150 getrennt, 49,0 %)")
    r60 = {r["verfahren"]: r for r in ev.exp4_tabelle(FZ, 60, 150)}
    assert ev.de_sci(r60["ML, Abbruchregel"]["mittel"], 1) == "7,4e+08" and round(r60["ML, 25 Schritte"]["mittel"], 2) == 2.40
    in_readme("7,4e+08", "2,40", "0,711 und 0,982")


def test_hilfe_ml_ist_in_keiner_getrennten_stichprobe_besser_als_firth():
    in_readme("ML in keiner der 150 getrennten Stichproben besser")
    for n in (60, 100):
        w = FZ["was_hilft"][str(n)]
        firth = np.array(w["mehrverlust"]["Firth"])
        for m in ("ML, Abbruchregel", "ML, 25 Schritte"):
            assert int(np.sum(np.array(w["mehrverlust"][m]) < firth)) == 0, (n, m)
        assert len(firth) == 150


def test_bezug_ohne_trennung_und_grenze():
    in_readme("ML 0,062, Firth 0,057; Firth besser als ML in 107 von 150", "Grenze der Logit-Form: 0,024")
    b = FZ["bezug_nicht_getrennt"]
    assert (round(b["ml_median"], 3), round(b["firth_median"], 3), b["firth_besser"], b["stichproben"]) == (0.062, 0.057, 107, 150)
    assert round(FZ["meta"]["mehrverlust_grenze"], 3) == 0.024
    assert all(r["median"] > FZ["meta"]["mehrverlust_grenze"] for n in (60, 100) for r in ev.exp4_tabelle(FZ, n, 150)), "keine Methode erreicht die Grenze der Logit-Form"


def test_beispiel_nullzelle_ab_zufalls_seed_1():
    in_readme("Umschlag 20 Container, kein Standgeld", "−19,65", "−26,65", "−3,95", "−5,50", "−2,67", "z = −0,01 (p = 0,995;", "nach 18 Schritten", "8,31 (p = 0,0040)", "ML 0,161", "0,219", "Firth 0,039", "0,053 und 0,059", "Streichen 0,101")
    b = ev.exp4_beispiel(100, 1)
    assert b["seed"] == 1 and b["nullzellen"] == [("Art Umschlag", 20, 0)] and b["spalte"] == "Umschlag"
    j = C.SPALTEN.index("Umschlag")
    for m, soll, tol in (("ML, Abbruchregel", -19.65, 0.05), ("ML, 25 Schritte", -26.65, 0.05), ("Firth", -3.95, 0.02), ("Ridge 0,1", -5.50, 0.02), ("Ridge 1", -2.67, 0.02)):
        assert b["koef"][m][j] == pytest.approx(soll, abs=tol), m
    assert b["z"] == pytest.approx(-0.01, abs=0.01) and b["p_wald"] == pytest.approx(0.995, abs=0.005)
    assert b["lq"] == pytest.approx(8.31, abs=0.02) and b["p_lq"] == pytest.approx(0.00395, abs=5e-5)
    for m, soll in (("ML, Abbruchregel", 0.161), ("ML, 25 Schritte", 0.219), ("Firth", 0.039), ("Ridge 0,1", 0.053), ("Ridge 1", 0.059), ("Streichen", 0.101)):
        assert b["mehrverlust"][m] == pytest.approx(soll, abs=0.002), m


def test_faustregel_tabelle():
    erwartet = {0: (None, None, 0.98), 50: (1.07, 0.061, 1.07), 100: (1.12, 0.076, 1.23), 200: (1.38, 0.140, 1.85)}
    for p_r, (sd, wald, zoll) in erwartet.items():
        z = ev.faustregel_zeile(FZ, p_r)
        assert round(z["zollhalt_verhaeltnis_median"], 2) == zoll and z["nicht_konvergiert"] == 0
        if sd is not None:
            assert round(z["sd_ueber_se"], 2) == sd and round(z["wald_ablehnung"], 3) == wald
    assert sum(ev.faustregel_zeile(FZ, p)["reps"] for p in C.E5_PR) == 520
    in_readme("in allen 520 Stichproben", "| 200 | 207 | 0,207 | 100 | 1,38 | 14,0 % | 1,85 |", "| 50 | 57 | 0,057 | 150 | 1,07 | 6,1 % | 1,07 |", "(1,925)", "1,38 bei p/n = 0,207")
    assert round(FZ["meta"]["beta_grenze"][4], 3) == 1.925


def test_bruecke_gerade_ausserhalb():
    in_readme("n = 100: 16,7 % (100 Stichproben), n = 500: 15,6 % (60), n = 3000: 15,5 % (40)", "für 15,5 % der Testcontainer")
    for n, reps, soll in ((100, 100, 16.7), (500, 60, 15.6), (3000, 40, 15.5)):
        z = ev.bruecke_zeile(FZ, n)
        assert z["reps"] == reps and round(100 * z["ausserhalb_mittel"], 1) == soll
        assert z["rmse_firth"] < z["rmse_ols"]


def test_korrekturen_frueherer_messungen_stehen_in_der_readme():
    in_readme("117 von 300 (39,0 %)", "127 von 300 (42,3 %)", "0 von 200, hier 5 von 200", "0,0498, hier 0,024", "14 und 18 von 150", "13,7 %", "1,74-fach", "15,0 % bei n = 3000")


def test_readme_nennt_keine_planungswoerter():
    import re

    ohne_fuss = README.rsplit("\n---\n", 1)[0]  # der Footer („Mehr zur Reihe: …“) stammt vom Footer-Werkzeug
    for wort in ("Stück", "Linie", "Reihe", "Vormessung", "Phase 0", "Plan Z", "geplant", "Saatwert", "trägt", "tragen"):
        assert not re.search(rf"\b{wort}\b", ohne_fuss), wort


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Neu-Rechnung eines Teils als Band
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _lade_tool():
    spec = importlib.util.spec_from_file_location("lsp_make_frozen", ROOT / "tools" / "lsp_make_frozen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vorgerechnetes_stimmt_mit_neu_rechnung_als_band():
    """`tools/lsp_make_frozen.py --check`: die ersten 40 Stichproben bei n = 60 und 100 (höchstens 2 weichen ab), die ersten 6 getrennten Stichproben des Mehrverlusts (relativ 1e-4) und die Logit-Grenze (relativ 1e-6)."""
    tool = _lade_tool()
    fehler = tool.pruefe_teile()
    assert not fehler, fehler


def test_bruecke_kleine_neu_rechnung_als_band():
    from lsp_messungen import bruecke

    neu = bruecke(500, 4)
    alt = ev.bruecke_zeile(FZ, 500)
    assert neu["ausserhalb_mittel"] == pytest.approx(alt["ausserhalb_mittel"], abs=0.02)
    assert neu["rmse_ols"] == pytest.approx(alt["rmse_ols"], abs=0.01)


def test_faustregel_kleine_neu_rechnung_als_band():
    from lsp_messungen import faustregel

    neu = faustregel(50, 8)
    alt = ev.faustregel_zeile(FZ, 50)
    assert neu["nicht_konvergiert"] == 0
    assert neu["sd_ueber_se"] == pytest.approx(alt["sd_ueber_se"], abs=0.08)
    assert neu["zollhalt_verhaeltnis_median"] == pytest.approx(alt["zollhalt_verhaeltnis_median"], abs=0.15)


@pytest.mark.skipif(not os.environ.get("LSP_FULL"), reason="LSP_FULL=1 rechnet alle vorgerechneten Messungen neu (einige Minuten)")
def test_voll_neu_rechnung():
    tool = _lade_tool()
    neu = tool.M.alles(verbose=False)
    fehler = tool._bandvergleich(json.loads(tool.PFAD.read_text(encoding="utf-8")), neu)
    assert not fehler, fehler[:10]




def test_cover_ohne_signal_bis_n_14_mindestens_ein_halb_bei_sieben_spalten():
    import lsp_separation as P

    assert P.cover_anteil(14, 7) == Fraction(1, 2) and P.cover_anteil(15, 7) < Fraction(1, 2)
    assert all(P.cover_anteil(n, 7) > Fraction(1, 2) for n in range(7, 14))


def test_nullzelle_bei_n_100_in_147_von_151():
    in_readme("147 von 151 getrennten Stichproben")
    z = ev.haeufigkeit_zeile(FZ, 100, 1.0)
    assert (z["nullzelle"], z["getrennt"]) == (147, 151)
    in_readme("151 von 300 und in 136 von 300 Stichproben aus zwei Zufallsströmen")


def test_nullzelle_je_gruppe_bei_n_100_neu_gerechnet_als_band():
    """„Vor allem Umschlag“: die Zählung je Gruppe steht nicht in der Datei, sondern wird hier neu gerechnet (Band: Zählungen ± 2)."""
    in_readme("bei Umschlag in 138 getrennten Stichproben", "bei Zollhalt 1 in 11", "bei Leer in 6", "bei Export in 1", "bei Import und Wochenende in 0")
    from lsp_messungen import nullzellen_je_gruppe

    z = nullzellen_je_gruppe(1.0, 100, 300)
    soll = ev.haeufigkeit_zeile(FZ, 100, 1.0)
    assert (soll["getrennt"], soll["nullzelle"]) == (151, 147)
    assert (z["getrennt"], z["mit_nullzelle"]) == pytest.approx((151, 147), abs=2)
    je = z["je_gruppe"]
    for name, erwartet in (("Art Umschlag", 138), ("Zollhalt", 11), ("Art Leer", 6), ("Art Export", 1), ("Art Import", 0), ("Wochenende", 0)):
        assert je.get(name, 0) == pytest.approx(erwartet, abs=2), name
    assert je["Art Umschlag"] > 10 * max(v for k, v in je.items() if k != "Art Umschlag")


def test_app_nennt_fuer_die_koeffizientenregel_200_schritte_wie_die_readme():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    in_readme("die Koeffizientenregel greift in 200 Schritten nicht")
    assert "greift im Pfad der App von 200 Schritten nie" in app and "700 Schritten" not in app
    r = ev.exp2(C.E2_DATEN[0], 1.0, 200, C.E2_REGELN[2])
    assert r["stopp"]["Koeffizientenänderung"] == (200, False)


def test_beispiel_wald_gehoert_zur_regel_mit_kappung_nach_18_schritten():
    import lsp_separation as P

    st = S.ziehe(100, 1, 1.0, stream=11)
    X = st.X()
    Xk = X[:, P.voller_rang(X)]
    assert P.ist_getrennt(Xk, st.y)
    _b, schritte, regulaer = L.irls_gekappt(Xk, st.y)
    assert (schritte, regulaer) == (18, True)


def test_firth_ridge_streichen_wie_in_der_readme_beschrieben():
    """Code-Verhalten, das die README nennt: Firth (Start 0, Schritt auf 5 begrenzt, 40 Halbierungen, 1e-9, 300 Schritte), Ridge (standardisiert mit Nenner n, Summe, Achsenabschnitt frei),
    Streichen (genau eine Spalte), gängige Regel (höchstens 25 Schritte)."""
    import inspect

    import lsp_separation as P
    from scipy import optimize, special

    in_readme("auf höchstens 5 zurückgeschnitten", "höchstens 40 Halbierungen", "unter 1e-9", "höchstens 300 Schritte", "Nenner n", "als Summe ein", "Achsenabschnitt bleibt unbestraft", "genau eine Spalte", "höchstens 25 Schritte")
    assert (C.FIRTH_SCHRITT_MAX, C.FIRTH_HALBIERUNGEN, C.FIRTH_TOL, C.FIRTH_MAX_SCHRITTE, C.GAENGIG_MAX_SCHRITTE) == (5.0, 40, 1e-9, 300, 25)
    assert inspect.signature(L.firth).parameters["maxit"].default == 300 and inspect.signature(L.firth).parameters["tol"].default == 1e-9
    assert inspect.signature(L.irls_gekappt).parameters["maxit"].default == 25

    # Firth: Start β = 0 (ein Schritt ist der Schritt aus β = 0), die Schrittkomponente ist auf 5 begrenzt
    x6 = np.array(C.SEP6_X, float)
    X6, y6 = L.design_1d(x6), np.array(C.SEP6_Y, float)
    u, inv = L.firth_score(X6, y6, np.zeros(2))
    assert L.firth(X6, y6, maxit=1)[0] == pytest.approx(inv @ u, abs=1e-12)
    X_klein, y_klein = L.design_1d(0.01 * x6), y6
    u, inv = L.firth_score(X_klein, y_klein, np.zeros(2))
    assert np.max(np.abs(inv @ u)) > 5.0, "ohne Begrenzung wäre der erste Schritt größer als 5"
    assert np.max(np.abs(L.firth(X_klein, y_klein, maxit=1)[0])) <= 5.0 + 1e-12

    st = S.ziehe(100, 1, 1.0, stream=11)
    X = st.X()
    Xk = X[:, P.voller_rang(X)]
    y = st.y

    # Ridge: Summe der Log-Likelihood, Standardisierung mit Nenner n (np.std), Achsenabschnitt unbestraft, Rückrechnung auf die Originalskala
    lam = 0.1
    mu, sd = Xk[:, 1:].mean(axis=0), Xk[:, 1:].std(axis=0)
    Z = np.column_stack([np.ones(len(y)), (Xk[:, 1:] - mu) / sd])

    def zielfunktion(b):
        eta = Z @ b
        wert = -np.sum(y * eta - np.logaddexp(0.0, eta)) + 0.5 * lam * np.sum(b[1:] ** 2)
        grad = -Z.T @ (y - special.expit(eta)) + lam * np.r_[0.0, b[1:]]
        return wert, grad

    bz = optimize.minimize(zielfunktion, np.zeros(Z.shape[1]), jac=True, method="BFGS", options={"gtol": 1e-10, "maxiter": 1000}).x
    erwartet = np.r_[bz[0] - np.sum(bz[1:] * mu / sd), bz[1:] / sd]
    assert L.ridge(Xk, y, lam) == pytest.approx(erwartet, abs=1e-4)

    # Streichen: genau eine Spalte (nicht der Achsenabschnitt) bekommt den Koeffizienten 0
    b, j = L.streichen(Xk, y)
    assert j >= 1 and b[j] == 0.0 and int(np.sum(b == 0.0)) == 1
