"""Auswertung: die fünf Experimente als reine Rechnungen und die Verdikte als reine Funktionen (Art, Text).

Live gerechnet wird, was klein ist (Handbeispiele, eine frisch gezogene Stichprobe, ein Pfad von höchstens 200 Schritten). Die Messungen über Hunderte Stichproben liegen
vorgerechnet in `data/lsp_frozen.json` (`tools/lsp_make_frozen.py`); die App kennzeichnet sie als „vorgerechnet“."""

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import special

import lsp_constants as C
import lsp_logit as L
import lsp_messungen as M
import lsp_scenario as S
import lsp_separation as P

ROOT = Path(__file__).resolve().parent
FROZEN_PATH = ROOT / "data" / "lsp_frozen.json"
NB = " "                                                   # geschütztes Leerzeichen als Tausendertrenner


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Anzeige
# ---------------------------------------------------------------------------------------------------------------------------------------------
def de(x: float, nd: int = 2, sign: bool = False) -> str:
    """Zahl mit Dezimalkomma (Anzeige)."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def de_sci(x: float, nd: int = 1) -> str:
    """Zahl in Exponentenschreibweise mit Dezimalkomma (2,1e-10)."""
    return f"{x:.{nd}e}".replace(".", ",").replace("-", "−")


def abstand_text(x: float) -> str:
    """Abstand zum Grenzwert: unter der Rechengenauigkeit „unter 1e-15“, sonst in Exponentenschreibweise."""
    return "unter 1e−15" if x < 1e-15 else de_sci(x)


def pct(x: float, nd: int = 1) -> str:
    return de(100.0 * x, nd) + " %"


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", NB)


@lru_cache(maxsize=1)
def load_frozen() -> dict:
    return json.loads(FROZEN_PATH.read_text(encoding="utf-8"))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson-Intervall (zweiseitig 95 %) für den Anteil k von n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, m - h), min(1.0, m + h)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: von Hand, IRLS gleich Newton, Gerade gegen Logit
# ---------------------------------------------------------------------------------------------------------------------------------------------
def exp1(beispiel: str) -> dict:
    x, y = S.handbeispiel(beispiel)
    X = L.design_1d(x)
    pfad = L.irls_pfad(X, y, C.E1_STEPS_MAX)
    newton = L.newton_pfad(X, y, C.E1_STEPS_MAX)
    grenz, k_konv = L.grenzwert(X, y)
    b_ols = L.gerade_ols(X, y)
    pred = X @ b_ols
    formel = L.zwei_gruppen_grenzwert(4, 1, 5, 4) if beispiel == C.E1_BEISPIELE[0] else None
    return {
        "beispiel": beispiel, "x": x, "y": y, "X": X, "pfad": pfad, "dev": L.devianzen(X, y, pfad), "newton": newton,
        "diff_newton": float(np.max(np.abs(pfad - newton))), "grenz": grenz, "k_konv": k_konv,
        "fehler": np.max(np.abs(pfad - grenz), axis=1), "bruch": L.erster_schritt_bruch([[1, int(v)] for v in x], [int(v) for v in y]),
        "b_ols": b_ols, "pred": pred, "aussen": [(float(a), float(b)) for a, b in zip(x, pred) if b < 0 or b > 1], "p_logit": special.expit(X @ grenz),
        "formel": formel, "wald": L.wald_und_lq(X, y, grenz),
    }


def bruch_text(f) -> str:
    return (str(f) if f.denominator != 1 else str(f.numerator)).replace("-", "−")


def verdict_exp1(res: dict, steps: int, bruecke_row: dict | None) -> tuple[str, str]:
    """Verdikt zu Exp. 1: der Pfad nach `steps` Schritten, der Grenzwert, IRLS gegen Newton und die Gerade."""
    b, g = res["pfad"][steps], res["grenz"]
    teile = [f"Nach {steps} Schritten liegt β bei ({de(b[0], 4)}; {de(b[1], 4)}), der Grenzwert ist ({de(g[0], 4)}; {de(g[1], 4)}); der größte Abstand beträgt {abstand_text(res['fehler'][steps])}. "
             f"Der IRLS-Pfad gleicht dem Newton-Pfad mit numerisch abgeleiteter Hesse-Matrix (größter Unterschied über alle {C.E1_STEPS_MAX} Schritte {de_sci(res['diff_newton'])}); "
             f"die Änderung sinkt nach {res['k_konv']} Schritten unter 1e-13."]
    f = res["formel"]
    if f is not None:
        teile.append(f"Der Grenzwert hat hier die geschlossene Form β = (ln {bruch_text(f['odds0'])}, ln {bruch_text(f['ratio'])}): Odds {bruch_text(f['odds0'])} ohne und {bruch_text(f['odds1'])} mit Zollhalt, Odds-Ratio {bruch_text(f['ratio'])}.")
    if res["aussen"]:
        a = res["aussen"]
        teile.append("Die Gerade auf y ∈ {0, 1} sagt bei " + ", ".join(f"x = {de(xx, 0)} den Wert {de(v, 2)}" for xx, v in a) + " voraus, also außerhalb von [0, 1]; das Logit bleibt dazwischen.")
    else:
        teile.append("Bei diesem einzigen 0/1-Merkmal liegt die Gerade genau auf den Gruppenanteilen und bleibt in [0, 1]; ein stetiges Merkmal (zweites Beispiel) bricht das.")
    if bruecke_row is not None:
        teile.append(f"Im Container-Vehikel sagt die Gerade bei n = {fmt_int(bruecke_row['n'])} für {pct(bruecke_row['ausserhalb_mittel'])} der Testcontainer einen Wert außerhalb [0, 1] voraus "
                     f"(vorgerechnet, Mittel über {bruecke_row['reps']} Stichproben).")
    return "success", " ".join(teile)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: Trennung, Wachstum, Abbruchregeln
# ---------------------------------------------------------------------------------------------------------------------------------------------
def exp2(daten: str, skala: float, steps: int, regel: str) -> dict:
    x, y = S.handbeispiel(daten)
    X = L.design_1d(x * skala)
    pfad = L.irls_pfad(X, y, steps)
    dev = L.devianzen(X, y, pfad)
    wal = [L.wald_und_lq(X, y, b) for b in pfad]
    opt = P.lp_optimum(X, y)
    stopp = {r: L.abbruch_schritt(pfad, dev, r) for r in C.E2_REGELN}
    return {
        "daten": daten, "skala": skala, "steps": steps, "regel": regel, "x": x * skala, "y": y, "X": X, "pfad": pfad, "dev": dev, "zuwachs": np.diff(pfad[:, 1]),
        "z": np.array([w["z"][1] for w in wal]), "p_wald": np.array([w["p_wald"][1] for w in wal]), "lq": np.array([w["lq"] for w in wal]), "p_lq": np.array([w["p_lq"] for w in wal]),
        "lp_opt": opt, "art": P.art_der_trennung(X, y), "stopp": stopp, "k_stopp": stopp[regel][0], "regel_greift": stopp[regel][1],
        "formel_zuwachs": 1.0 / float(np.min(np.abs(x * skala))) if np.all(x != 0) and daten == C.E2_DATEN[0] else None,
        "dev_null": L.deviance_null(y),
    }


def verdict_exp2(res: dict) -> tuple[str, str]:
    k = res["k_stopp"]
    b1 = res["pfad"][k, 1]
    z, pw, lq, plq = res["z"][k], res["p_wald"][k], res["lq"][k], res["p_lq"][k]
    if res["art"] == "überlappend":
        return "info", f"Die Daten überlappen (Rezessions-LP: Optimum {de(res['lp_opt'], 2)}); der Schätzer existiert, β₁ = {de(b1, 3)} nach {k} Schritten."
    wo = {"Deviance-Regel": "Die Deviance-Regel hält", "Feste Schrittzahl": "Die feste Schrittzahl endet", "Koeffizientenänderung": "Die Koeffizientenregel hält"}[res["regel"]]
    ohne = "" if res["regel_greift"] else " (die Regel greift in den gerechneten Schritten nicht; es zählt das Ende des Pfads)"
    teile = [f"Die Daten sind {res['art']} getrennt (Rezessions-LP: Optimum {de(res['lp_opt'], 1)}): der ML-Schätzer existiert nicht, jeder angezeigte Wert ist ein Abbruchwert.",
             f"{wo} bei Schritt {k}{ohne} mit β₁ = {de(b1, 2)}; nach {res['steps']} Schritten wäre es β₁ = {de(res['pfad'][-1, 1], 2)}.",
             f"Dort steht der Wald-Test bei z = {de(z, 4)} (p = {de(pw, 2)}), der Likelihood-Quotient bei {de(lq, 2)} (p = {de(plq, 4)}): "
             + ("der Wald-Test findet keinen Effekt, obwohl die Daten " + ("vollständig " if res["art"] == "vollständig" else "") + "trennen." if pw > 0.05 else "hier erkennen beide Tests den Effekt; mit mehr Schritten sinkt z des Wald-Tests wieder.")]
    if res["formel_zuwachs"] is not None and len(res["zuwachs"]) > 30:
        teile.append(f"Der Zuwachs von β₁ je Schritt beträgt im Mittel der Schritte 31 bis {len(res['zuwachs'])} {de(float(np.mean(res['zuwachs'][30:])), 4)}; die Formel 1/min|x| nennt {de(res['formel_zuwachs'], 4)} "
                     "(gilt nur für symmetrische Daten mit zwei Stützpunkten).")
    return "warning", " ".join(teile)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3: wie oft trennen die Daten?
# ---------------------------------------------------------------------------------------------------------------------------------------------
def haeufigkeit_zeile(fz: dict, n: int, noise: float) -> dict:
    return next(e for e in fz["haeufigkeit"] if e["n"] == n and e["noise"] == noise)


def haeufigkeit_je_n(fz: dict, noise: float) -> list[dict]:
    rows = []
    for e in sorted((e for e in fz["haeufigkeit"] if e["noise"] == noise), key=lambda e: e["n"]):
        lo, hi = wilson(e["getrennt"], e["reps"])
        rows.append({**e, "anteil": e["getrennt"] / e["reps"], "lo": lo, "hi": hi})
    return rows


def cover_je_n(p: int, ns) -> list[float]:
    """Anteil der trennbaren Beschriftungen nach der Cover-Formel (kein Zusammenhang zwischen Merkmalen und Ausgang)."""
    return [float(P.cover_anteil(n, p)) for n in ns]


def exp3_beispiel(n: int, noise: float, seed: int) -> dict:
    """Eine frisch gezogene Stichprobe: Trennung ja/nein, Art der Trennung, Container und Standgeld je Art, Nullzellen."""
    st = S.ziehe(n, seed, noise, stream=10)
    X = st.X()
    keep = P.voller_rang(X)
    Xk = X[:, keep]
    art = P.art_der_trennung(Xk, st.y)
    je_art = [(C.ART_LABELS[a], int(np.sum(st.art == a)), int(np.sum(st.y[st.art == a]))) for a in range(4)]
    return {"n": n, "noise": noise, "seed": seed, "art": art, "je_art": je_art, "nullzellen": P.nullzellen(st.art, st.zoll, st.we, st.y), "anteil_standgeld": float(st.y.mean()),
            "spalten_ohne_variation": [C.SPALTEN[j] for j in range(X.shape[1]) if j not in keep], "lp_opt": P.lp_optimum(Xk, st.y)}


def verdict_exp3(zeile: dict, zweiter: dict | None, noise: float) -> tuple[str, str]:
    k, r = zeile["getrennt"], zeile["reps"]
    lo, hi = wilson(k, r)
    wo = "bei Standardstreuung" if noise == 1.0 else "bei halber Streuung der Log-Verweilzeit"
    teile = [f"Bei n = {fmt_int(zeile['n'])} Containern {wo} waren {k} von {r} Stichproben getrennt ({pct(k / r)}; 95-%-Band {pct(lo, 0)} bis {pct(hi, 0)}), "
             f"davon {zeile['quasi']} nur quasi-getrennt und {zeile['nullzelle']} mit einer Nullzelle in Art, Zollhalt oder Wochenende."]
    if k == 0:
        teile.append(f"Null getrennte Stichproben von {r} heißt nicht „kann nicht vorkommen“: der obere Rand des Bandes liegt bei {pct(hi, 1)}.")
    if zweiter is not None and zeile["n"] == 100 and noise == 1.0:
        teile.append(f"Ein zweiter Zufallsstrom mit 300 anderen Stichproben gab {zweiter['getrennt']} von {zweiter['reps']} ({pct(zweiter['getrennt'] / zweiter['reps'])}); die Zahl ist also ein Band, keine Stelle genau.")
    return ("warning" if k / r >= 0.2 else "info"), " ".join(teile)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4: was hilft?
# ---------------------------------------------------------------------------------------------------------------------------------------------
def exp4_tabelle(fz: dict, n: int, k: int) -> list[dict]:
    """Je Verfahren Median, Mittel und 90. Perzentil des Mehrverlusts über die ersten k getrennten Stichproben, dazu in wie vielen besser als Firth."""
    w = fz["was_hilft"][str(n)]
    firth = np.array(w["mehrverlust"]["Firth"][:k])
    rows = []
    for m in C.E4_VERFAHREN:
        a = np.array(w["mehrverlust"][m][:k])
        rows.append({"verfahren": m, "median": float(np.median(a)), "mittel": float(a.mean()), "p90": float(np.percentile(a, 90)), "rmse_median": float(np.median(w["rmse"][m][:k])),
                     "besser": None if m == "Firth" else int(np.sum(a < firth)), "k": len(a)})
    return rows


def exp4_beispiel(n: int, seed: int) -> dict | None:
    """Erste getrennte Stichprobe ab dem Zufalls-Seed (höchstens SEARCH_MAX Versuche): Koeffizienten und Mehrverlust je Verfahren, Wald und Likelihood-Quotient der Spalte mit dem größten |β|."""
    for v in range(C.SEARCH_MAX):
        st = S.ziehe(n, seed + v, 1.0, stream=11)
        X = st.X()
        keep = P.voller_rang(X)
        Xk = X[:, keep]
        if not P.ist_getrennt(Xk, st.y):
            continue
        koef = {m: M.voll(b, keep) for m, b in M.verfahren(Xk, st.y).items()}
        b_ml = koef["ML, Abbruchregel"]
        wal = L.wald_und_lq(Xk, st.y, b_ml[keep])
        jmax = int(np.argmax(np.abs(b_ml[keep][1:]))) + 1
        red = np.delete(Xk, jmax, axis=1)
        b_red, _, konv = L.ml_konvergiert(red, st.y)
        lq = L.deviance(red, st.y, b_red) - L.deviance(Xk, st.y, b_ml[keep])
        from scipy import stats

        return {"n": n, "seed": seed + v, "versuche": v + 1, "koef": koef, "mehrverlust": {m: S.mehrverlust(b) for m, b in koef.items()},
                "je_art": [(C.ART_LABELS[a], int(np.sum(st.art == a)), int(np.sum(st.y[st.art == a]))) for a in range(4)], "nullzellen": P.nullzellen(st.art, st.zoll, st.we, st.y),
                "spalte": C.SPALTEN[keep[jmax]], "z": float(wal["z"][jmax]), "p_wald": float(wal["p_wald"][jmax]), "lq": float(lq), "p_lq": float(stats.chi2.sf(max(lq, 0.0), 1)),
                "lq_konv": bool(konv), "grenze": S.logit_grenze()}
    return None


def verdict_exp4(rows: list[dict], ref: dict, grenze_mv: float, n: int, k: int, gewaehlt: str) -> tuple[str, str]:
    r = {x["verfahren"]: x for x in rows}
    ml, fi = r["ML, Abbruchregel"], r["Firth"]
    schnitt = ", ".join(f"{m} {de(r[m]['median'], 3)}" for m in ("Ridge 0,1", "Ridge 1", "Streichen"))
    teile = [f"n = {n}, {k} getrennte Stichproben: Mehrverlust gegen die wahre Wahrscheinlichkeit im Median {de(ml['median'], 3)} (ML mit Abbruchregel) gegen {de(fi['median'], 3)} (Firth); {schnitt}. "
             f"ML war in {ml['besser']} von {k} Stichproben besser als Firth.",
             f"Ohne Trennung (Bezug, {ref['stichproben']} Stichproben mit n = {ref['n']}) liegen ML bei {de(ref['ml_median'], 3)} und Firth bei {de(ref['firth_median'], 3)} im Median, Firth war in {ref['firth_besser']} besser.",
             f"Keine Methode erreicht die Grenze der Logit-Form ({de(grenze_mv, 3)}); Ridge hängt an der Stärke (nur zwei feste Werte gemessen), „Streichen“ ist nur als Streichen der Spalte mit dem größten |β| gemessen."]
    if gewaehlt != "Firth":
        g = r[gewaehlt]
        teile.append(f"Gewählt: {gewaehlt}, Median {de(g['median'], 3)}, 90. Perzentil {de(g['p90'], 3)}.")
    return "info", " ".join(teile)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 5: Faustregel n >= 5p
# ---------------------------------------------------------------------------------------------------------------------------------------------
def faustregel_zeile(fz: dict, p_r: int) -> dict:
    return next(e for e in fz["faustregel"] if e["p_r"] == p_r)


def verdict_exp5(z: dict) -> tuple[str, str]:
    if z["p_r"] == 0:
        return "info", (f"Mit nur den {z['spalten']} Container-Spalten (p/n = {de(z['p_n'], 3)}) liegt der Zollhalt-Koeffizient im Median beim {de(z['zollhalt_verhaeltnis_median'], 2)}-fachen des Grenzwerts; "
                        f"Ausgangslage ohne Rauschmerkmale ({z['reps']} Stichproben, in {z['nicht_konvergiert']} kein Schätzer nach 30 Schritten).")
    return "warning", (f"p/n = {de(z['p_n'], 3)} ({z['spalten']} Spalten, n = 1000, {z['reps']} Stichproben): die Rausch-Schätzer streuen {de(z['sd_ueber_se'], 2)}-fach so stark, wie der mittlere Standardfehler angibt; "
                       f"{pct(z['wald_ablehnung'])} der Rauschmerkmale gelten bei nominal 5 % als signifikant; der Zollhalt-Koeffizient liegt im Median beim {de(z['zollhalt_verhaeltnis_median'], 2)}-fachen des Grenzwerts. " + (
                       "In keiner Stichprobe" if z["nicht_konvergiert"] == 0 else f"In {z['nicht_konvergiert']} Stichproben") + " fand ML keinen Schätzer in 30 Schritten. Container-Merkmale, nicht die Gauß-Anordnung der Quelle.")


def bruecke_zeile(fz: dict, n: int) -> dict:
    return next(e for e in fz["bruecke"] if e["n"] == n)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Kernfragen (kurze Fassung, aus denselben Zahlen)
# ---------------------------------------------------------------------------------------------------------------------------------------------
def kernfrage_1(res1: dict, bruecke_row: dict) -> tuple[str, str]:
    return "success", (f"Beim Ja/Nein-Ausgang passt man ein Logit an; der IRLS-Pfad ist beim kanonischen Link der Newton-Pfad (Handbeispiel: Unterschied {de_sci(res1['diff_newton'])}, "
                       f"{res1['k_konv']} Schritte). Eine Gerade sagt dagegen im Container-Vehikel bei n = {fmt_int(bruecke_row['n'])} für {pct(bruecke_row['ausserhalb_mittel'])} der Testcontainer einen Wert außerhalb [0, 1] voraus.")


def kernfrage_2(res2: dict) -> tuple[str, str]:
    k = res2["k_stopp"]
    return "warning", (f"Trennen die Daten, gibt es keinen Schätzer: bei den Daten „{res2['daten']}“ hält die Regel „{res2['regel']}“ bei Schritt {k} mit β₁ = {de(res2['pfad'][k, 1], 2)}, nach {res2['steps']} Schritten wäre es {de(res2['pfad'][-1, 1], 2)}; "
                       f"der Wald-Test steht dabei bei z = {de(res2['z'][k], 4)}, der Likelihood-Quotient bei p = {de(res2['p_lq'][k], 4)}.")


def kernfrage_3(zeile: dict, rows4: list[dict], n4: int, k4: int) -> tuple[str, str]:
    r = {x["verfahren"]: x for x in rows4}
    return "info", (f"Bei n = {zeile['n']} Containern waren {zeile['getrennt']} von {zeile['reps']} Stichproben getrennt (vorgerechnet). Bei n = {n4} und {k4} getrennten Stichproben liegt der Mehrverlust im Median bei "
                    f"{de(r['ML, Abbruchregel']['median'], 3)} (ML mit Abbruchregel) gegen {de(r['Firth']['median'], 3)} (Firth); ML war in {r['ML, Abbruchregel']['besser']} von {k4} besser als Firth.")


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Kennzahlen des Vehikels
# ---------------------------------------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def wahre_rate_je_art() -> dict:
    """Mittlere wahre Standgeld-Wahrscheinlichkeit je Containerart auf der Testmenge (100 000 Container, nur Merkmale)."""
    X, p, _ = S.testmenge()
    art = np.select([X[:, 1] == 1, X[:, 2] == 1, X[:, 3] == 1], [1, 2, 3], default=0)
    return {C.ART_LABELS[a]: float(p[art == a].mean()) for a in range(4)}


def de_mv(x: float) -> str:
    """Mehrverlust in nat mit drei Stellen; ab 100 (einzelne Ausreißer bestimmen das Mittel) in Exponentenschreibweise."""
    return de_sci(x, 2) if x >= 100 else de(x, 3)
