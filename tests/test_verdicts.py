"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier. Ein Verdikt ist eine reine Funktion (Art, Text)."""

import numpy as np

import lsp_evaluation as ev


# --- Anzeige ---------------------------------------------------------------------------------------------------------------------------------
def test_anzeigefunktionen():
    assert ev.de(-1.0986, 3) == "−1,099" and ev.de(0.5, 2, sign=True) == "+0,50"
    assert ev.de_sci(2.1e-10) == "2,1e−10"
    assert ev.abstand_text(0.0) == "unter 1e−15" and ev.abstand_text(3e-16) == "unter 1e−15" and ev.abstand_text(2.7e-5) == "2,7e−05"
    assert ev.pct(0.1523) == "15,2 %" and ev.pct(0.4, 0) == "40 %"
    assert ev.fmt_int(3000) == "3 000"
    assert ev.de_mv(0.0554) == "0,055" and ev.de_mv(7.35e8) == "7,35e+08" and ev.de_mv(99.9) == "99,900"


# --- Exp. 1 ----------------------------------------------------------------------------------------------------------------------------------
def _res1(formel=True, aussen=()):
    return {
        "pfad": np.array([[0, 0], [-1.0, 2.2], [-1.0963, 2.4733]]), "grenz": np.array([-1.0986, 2.4849]), "fehler": np.array([2.5, 0.285, 0.0116]), "diff_newton": 3.1e-10, "k_konv": 6,
        "formel": ev.L.zwei_gruppen_grenzwert(4, 1, 5, 4) if formel else None, "aussen": list(aussen),
    }


def test_verdikt_exp1_nennt_pfad_grenzwert_newton_und_formel():
    kind, text = ev.verdict_exp1(_res1(), 2, None)
    assert kind == "success"
    assert "Nach 2 Schritten liegt β bei (−1,0963; 2,4733)" in text and "Grenzwert ist (−1,0986; 2,4849)" in text and "Abstand beträgt 1,2e−02" in text
    assert "Unterschied über alle 8 Schritte 3,1e−10" in text and "nach 6 Schritten unter 1e-13" in text
    assert "β = (ln 1/3, ln 12)" in text and "Odds 1/3 ohne und 4 mit Zollhalt, Odds-Ratio 12" in text
    assert "liegt die Gerade genau auf den Gruppenanteilen" in text


def test_verdikt_exp1_gerade_ausserhalb_und_bruecke():
    bruecke = {"n": 3000, "ausserhalb_mittel": 0.155, "reps": 40}
    kind, text = ev.verdict_exp1(_res1(formel=False, aussen=[(1.0, -0.0833), (8.0, 1.0833)]), 1, bruecke)
    assert "bei x = 1 den Wert −0,08, x = 8 den Wert 1,08 voraus" in text and "β = (ln" not in text
    assert "bei n = 3 000 für 15,5 % der Testcontainer" in text and "Mittel über 40 Stichproben" in text


# --- Exp. 2 ----------------------------------------------------------------------------------------------------------------------------------
def _res2(art="vollständig", pw=1.0, greift=True, formel=1.0, regel="Deviance-Regel"):
    pfad = np.zeros((41, 2))
    pfad[:, 1] = np.arange(41) * 1.0 - 1.04
    return {"art": art, "lp_opt": 12.0, "regel": regel, "k_stopp": 24, "regel_greift": greift, "steps": 40, "pfad": pfad, "z": np.full(41, 0.0003), "p_wald": np.full(41, pw), "lq": np.full(41, 8.32),
            "p_lq": np.full(41, 0.0039), "formel_zuwachs": formel, "zuwachs": np.ones(40), "daten": "vollständig (6 Punkte)"}


def test_verdikt_exp2_getrennt_wald_findet_keinen_effekt():
    kind, text = ev.verdict_exp2(_res2())
    assert kind == "warning"
    assert "sind vollständig getrennt (Rezessions-LP: Optimum 12,0)" in text and "ML-Schätzer existiert nicht" in text
    assert "Die Deviance-Regel hält bei Schritt 24 mit β₁ = 22,96; nach 40 Schritten wäre es β₁ = 38,96" in text
    assert "z = 0,0003 (p = 1,00)" in text and "Likelihood-Quotient bei 8,32 (p = 0,0039)" in text and "findet keinen Effekt" in text
    assert "Schritte 31 bis 40 1,0000; die Formel 1/min|x| nennt 1,0000" in text


def test_verdikt_exp2_schwelle_wald_p_ist_0_05():
    assert "findet keinen Effekt" in ev.verdict_exp2(_res2(pw=0.0501))[1]
    assert "erkennen beide Tests den Effekt" in ev.verdict_exp2(_res2(pw=0.05))[1]


def test_verdikt_exp2_sagt_vollstaendig_nur_bei_vollstaendiger_trennung():
    assert "obwohl die Daten vollständig trennen" in ev.verdict_exp2(_res2())[1]
    text = ev.verdict_exp2(_res2(art="quasi"))[1]
    assert "quasi getrennt" in text and "obwohl die Daten trennen." in text and "vollständig trennen" not in text and "vollkommen" not in text


def test_verdikt_exp2_regel_greift_nicht_und_ueberlappung():
    assert "greift in den gerechneten Schritten nicht" in ev.verdict_exp2(_res2(greift=False, regel="Koeffizientenänderung"))[1]
    assert "Die Koeffizientenregel hält" in ev.verdict_exp2(_res2(regel="Koeffizientenänderung"))[1]
    assert "Die feste Schrittzahl endet" in ev.verdict_exp2(_res2(regel="Feste Schrittzahl"))[1]
    kind, text = ev.verdict_exp2(_res2(art="überlappend"))
    assert kind == "info" and "überlappen" in text and "der Schätzer existiert" in text
    assert "Formel" not in ev.verdict_exp2(_res2(formel=None))[1]


# --- Exp. 3 ----------------------------------------------------------------------------------------------------------------------------------
def test_verdikt_exp3_schwelle_zwanzig_prozent_und_nullzahl():
    zeile = {"n": 100, "reps": 300, "getrennt": 60, "quasi": 60, "nullzelle": 57}
    kind, text = ev.verdict_exp3(zeile, {"getrennt": 136, "reps": 300}, 1.0)
    assert kind == "warning"                                        # genau 20 %
    assert "Bei n = 100 Containern bei Standardstreuung waren 60 von 300 Stichproben getrennt (20,0 %" in text
    assert "davon 60 nur quasi-getrennt und 57 mit einer Nullzelle" in text and "gab 136 von 300 (45,3 %)" in text
    assert ev.verdict_exp3({**zeile, "getrennt": 59}, None, 1.0)[0] == "info"
    kind, text = ev.verdict_exp3({"n": 1000, "reps": 200, "getrennt": 0, "quasi": 0, "nullzelle": 0}, None, 0.5)
    assert kind == "info" and "bei halber Streuung" in text and "Null getrennte Stichproben von 200 heißt nicht" in text and "obere Rand des Bandes liegt bei 1,9 %" in text
    assert "zweiter Zufallsstrom" not in text


# --- Exp. 4 ----------------------------------------------------------------------------------------------------------------------------------
def _rows4():
    namen = ["ML, Abbruchregel", "ML, 25 Schritte", "Firth", "Ridge 0,1", "Ridge 1", "Streichen"]
    med = [0.184, 0.241, 0.0554, 0.0674, 0.0708, 0.1103]
    return [{"verfahren": m, "median": v, "mittel": v, "p90": v + 0.1, "rmse_median": 0.1, "besser": None if m == "Firth" else 3, "k": 150} for m, v in zip(namen, med)]


def test_verdikt_exp4_nennt_mediane_besser_als_firth_und_grenze():
    ref = {"stichproben": 150, "n": 100, "ml_median": 0.062, "firth_median": 0.057, "firth_besser": 107}
    kind, text = ev.verdict_exp4(_rows4(), ref, 0.0239, 100, 150, "Firth")
    assert kind == "info"
    assert "n = 100, 150 getrennte Stichproben" in text and "im Median 0,184 (ML mit Abbruchregel) gegen 0,055 (Firth)" in text
    assert "Ridge 0,1 0,067, Ridge 1 0,071, Streichen 0,110" in text and "ML war in 3 von 150 Stichproben besser als Firth" in text
    assert "Bezug, 150 Stichproben mit n = 100) liegen ML bei 0,062 und Firth bei 0,057 im Median, Firth war in 107 besser" in text
    assert "Grenze der Logit-Form (0,024)" in text and "Gewählt" not in text
    assert "Gewählt: Ridge 1, Median 0,071, 90. Perzentil 0,171" in ev.verdict_exp4(_rows4(), ref, 0.0239, 100, 150, "Ridge 1")[1]


# --- Exp. 5 ----------------------------------------------------------------------------------------------------------------------------------
def test_verdikt_exp5_ohne_und_mit_rauschmerkmalen():
    z = {"p_r": 200, "spalten": 207, "p_n": 0.207, "reps": 100, "nicht_konvergiert": 0, "sd_ueber_se": 1.379, "wald_ablehnung": 0.14, "zollhalt_verhaeltnis_median": 1.85}
    kind, text = ev.verdict_exp5(z)
    assert kind == "warning"
    assert "p/n = 0,207 (207 Spalten, n = 1000, 100 Stichproben)" in text and "streuen 1,38-fach" in text and "14,0 % der Rauschmerkmale gelten bei nominal 5 %" in text
    assert "beim 1,85-fachen des Grenzwerts" in text and "In keiner Stichprobe fand ML keinen Schätzer" in text
    assert "In 3 Stichproben fand ML" in ev.verdict_exp5({**z, "nicht_konvergiert": 3})[1]
    kind, text = ev.verdict_exp5({**z, "p_r": 0, "spalten": 7, "p_n": 0.007, "zollhalt_verhaeltnis_median": 0.976, "sd_ueber_se": None, "wald_ablehnung": None})
    assert kind == "info" and "Mit nur den 7 Container-Spalten (p/n = 0,007)" in text and "beim 0,98-fachen" in text


# --- Kernfragen ------------------------------------------------------------------------------------------------------------------------------
def test_kernfragen_nennen_ihre_zahlen():
    kind, text = ev.kernfrage_1({"diff_newton": 3.1e-10, "k_konv": 6}, {"n": 3000, "ausserhalb_mittel": 0.155})
    assert kind == "success" and "Unterschied 3,1e−10, 6 Schritte" in text and "n = 3 000 für 15,5 %" in text
    kind, text = ev.kernfrage_2(_res2())
    assert kind == "warning" and "hält die Regel „Deviance-Regel“ bei Schritt 24 mit β₁ = 22,96, nach 40 Schritten wäre es 38,96" in text and "z = 0,0003, der Likelihood-Quotient bei p = 0,0039" in text
    kind, text = ev.kernfrage_3({"n": 200, "getrennt": 71, "reps": 300}, _rows4(), 100, 150)
    assert kind == "info" and "Bei n = 200 Containern waren 71 von 300" in text and "0,184 (ML mit Abbruchregel) gegen 0,055 (Firth); ML war in 3 von 150" in text
