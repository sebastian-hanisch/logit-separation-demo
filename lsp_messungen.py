"""Die vorgerechneten Messungen (Exp. 3 bis 5 und die Brücke vom stetigen Ausgang) und die kleinen Neu-Rechnungen zur Prüfung.

Jede Messung ist eine Funktion mit festen Zufalls-Seeds; `tools/lsp_make_frozen.py` ruft sie auf und schreibt `data/lsp_frozen.json`. Die App liest nur die Datei
(schwere Messungen: Hunderte Stichproben mit bis zu 3000 Containern, 100 Stichproben mit 207 Spalten). Ein Test rechnet Teile neu und vergleicht sie als Band
(`tests/test_claims.py`, Abschnitt „vorgerechnet“). Zähleinheit überall: die Stichprobe (ein Datensatz aus n Containern), wo nichts anderes steht."""

import numpy as np
from scipy import special

import lsp_constants as C
import lsp_logit as L
import lsp_rngvec as V
import lsp_scenario as S
import lsp_separation as P

# Seeds und Ströme der Messungen (feste Zufalls-Seeds; ein anderer Strom = andere Stichproben)
H_SEED, H_STREAM = 20000, 1             # Häufigkeit der Trennung, erster Zufallsstrom
H2_SEED, H2_STREAM = 50000, 2           # Gegenprobe mit anderem Zufallsstrom
LP_SEED, LP_STREAM = 70000, 3           # Lücke der LP-Optimalwerte
D_SEED, D_STREAM = 90000, 4             # getrennte Stichproben (n = 60, 100)
D0_SEED, D0_STREAM = 31000, 5           # nicht getrennte Stichproben als Bezug
E_SEED, E_STREAM, E_NOISE_STREAM = 300000, 6, 7
F_SEED, F_STREAM = 400000, 8

H_GRID = ((30, 300), (60, 300), (100, 300), (200, 300), (500, 200), (1000, 200), (3000, 100))
D_ZIEL = 150
E_GRID = ((0, 150), (50, 150), (100, 120), (200, 100))
F_GRID = ((100, 100), (500, 60), (3000, 40))


def _entscheide(st):
    """Getrennt? Dazu Art der Trennung, Nullzelle, fehlende Variation. st: Stichprobe."""
    X = st.X()
    keep = P.voller_rang(X)
    Xk = X[:, keep]
    opt = P.lp_optimum(Xk, st.y)
    getrennt = opt > C.LP_SCHWELLE
    quasi = getrennt and not P.ist_vollstaendig_getrennt(Xk, st.y)
    nz = bool(P.nullzellen(st.art, st.zoll, st.we, st.y)) if getrennt else False
    return {"getrennt": bool(getrennt), "quasi": bool(quasi), "nullzelle": bool(nz), "konstant": len(keep) < X.shape[1], "opt": opt}


def haeufigkeit(noise: float, n: int, reps: int, seed: int = H_SEED, stream: int = H_STREAM) -> dict:
    """Anteil getrennter Stichproben: reps Stichproben aus n Containern bei Streuungsfaktor noise (Zufalls-Seed seed + k im Strom stream)."""
    bits, quasi, nz, konst = [], 0, 0, 0
    for k in range(reps):
        e = _entscheide(S.ziehe(n, seed + k, noise, stream))
        bits.append("1" if e["getrennt"] else "0")
        quasi += e["quasi"]
        nz += e["nullzelle"]
        konst += e["konstant"]
    return {"noise": noise, "n": n, "reps": reps, "seed": seed, "stream": stream, "getrennt": bits.count("1"), "quasi": quasi, "nullzelle": nz, "konstant": konst, "bits": "".join(bits)}


def lp_luecke(reps: int = 150) -> dict:
    """Optimalwerte des Rezessions-LP über n = 100 und 200, beide Streuungen: kleinster Wert der getrennten, größter Betrag der nicht getrennten Stichproben."""
    pos, nul = [], []
    for noise in (1.0, 0.5):
        for n in (100, 200):
            for k in range(reps):
                st = S.ziehe(n, LP_SEED + k, noise, LP_STREAM)
                X = st.X()
                v = P.lp_optimum(X[:, P.voller_rang(X)], st.y)
                (pos if v > C.LP_SCHWELLE else nul).append(v)
    return {"reps": reps, "stichproben": len(pos) + len(nul), "getrennt": len(pos), "nicht_getrennt": len(nul), "min_getrennt": float(min(pos)), "max_betrag_nicht_getrennt": float(max(abs(v) for v in nul))}


def verfahren(Xk, y):
    """Koeffizienten (auf den behaltenen Spalten) je Verfahren der Tabelle in Exp. 4."""
    out = {}
    out["ML, Abbruchregel"] = L.irls_gekappt(Xk, y)[0]
    out["ML, 25 Schritte"] = L.ml_stabil(Xk, y, 25)
    out["Firth"] = L.firth(Xk, y)[0]
    out["Ridge 0,1"] = L.ridge(Xk, y, 0.1)
    out["Ridge 1"] = L.ridge(Xk, y, 1.0)
    out["Streichen"] = L.streichen(Xk, y)[0]
    return out


def voll(b, keep, p=7):
    """Koeffizienten auf die sieben Spalten zurückfüllen (nicht behaltene Spalten: 0)."""
    full = np.zeros(p)
    full[keep] = b
    return full


def was_hilft(n: int, ziel: int = D_ZIEL, max_zug: int = 1500) -> dict:
    """Mehrverlust gegen die wahre Wahrscheinlichkeit je Verfahren in `ziel` getrennten Stichproben aus n Containern (Standardstreuung)."""
    mv = {m: [] for m in C.E4_VERFAHREN}
    rm = {m: [] for m in C.E4_VERFAHREN}
    zug = 0
    seed0 = D_SEED + 1000 * n
    while len(mv["Firth"]) < ziel and zug < max_zug:
        st = S.ziehe(n, seed0 + zug, 1.0, D_STREAM)
        zug += 1
        X = st.X()
        keep = P.voller_rang(X)
        Xk = X[:, keep]
        if not P.ist_getrennt(Xk, st.y):
            continue
        for m, b in verfahren(Xk, st.y).items():
            bf = voll(b, keep)
            mv[m].append(S.mehrverlust(bf))
            rm[m].append(S.rmse_wahr(bf))
    return {"n": n, "ziel": ziel, "gezogen": zug, "getrennt": len(mv["Firth"]), "seed": seed0, "stream": D_STREAM,
            "mehrverlust": {m: [round(v, 6) for v in mv[m]] for m in mv}, "rmse": {m: [round(v, 6) for v in rm[m]] for m in rm}}


def bezug_nicht_getrennt(n: int = 100, ziel: int = D_ZIEL, max_zug: int = 1500) -> dict:
    """Mehrverlust von ML und Firth in `ziel` NICHT getrennten Stichproben (n Container): der Bezug ohne Trennung."""
    ml, fi = [], []
    zug = 0
    while len(ml) < ziel and zug < max_zug:
        st = S.ziehe(n, D0_SEED + zug, 1.0, D0_STREAM)
        zug += 1
        X = st.X()
        keep = P.voller_rang(X)
        Xk = X[:, keep]
        if P.ist_getrennt(Xk, st.y):
            continue
        ml.append(S.mehrverlust(voll(L.ml_konvergiert(Xk, st.y)[0], keep)))
        fi.append(S.mehrverlust(voll(L.firth(Xk, st.y)[0], keep)))
    ml, fi = np.array(ml), np.array(fi)
    return {"n": n, "stichproben": len(ml), "ml_median": float(np.median(ml)), "ml_mittel": float(ml.mean()), "firth_median": float(np.median(fi)), "firth_mittel": float(fi.mean()),
            "firth_besser": int(np.sum(fi < ml))}


def faustregel(p_r: int, reps: int, n: int = 1000) -> dict:
    """n Container plus p_r reine Rauschmerkmale N(0, 1) (Wirkung null): ML mit höchstens 30 Schritten. Rausch-Schätzer gegen Standardfehler, Wald-Ablehnung, Aufblähung."""
    binf = S.logit_grenze()
    schaetz, ses, zs, vz, ve, nk = [], [], [], [], [], 0
    for k in range(reps):
        st = S.ziehe(n, E_SEED + k, 1.0, E_STREAM)
        Fc = st.X()
        F = np.column_stack([Fc, V.normal_matrix(S.stream_seed(160000 + 1000 * p_r + k, E_NOISE_STREAM), n, p_r)]) if p_r else Fc
        pfad = L.irls_pfad(F, st.y, 30)
        kk = L.konvergiert_bei(pfad, 1e-8)
        if kk is None:
            nk += 1
            continue
        b = pfad[kk]
        w, _ = L.gewicht_residuum(F, st.y, b)
        se = np.sqrt(np.diag(np.linalg.inv(F.T @ (w[:, None] * F))))
        if p_r:
            schaetz.append(b[7:])
            ses.append(se[7:])
            zs.append(b[7:] / se[7:])
        vz.append(b[4] / binf[4])
        ve.append(b[1] / binf[1])
    row = {"p_r": p_r, "spalten": 7 + p_r, "p_n": (7 + p_r) / n, "reps": reps, "nicht_konvergiert": nk, "zollhalt_verhaeltnis_median": float(np.median(vz)),
           "export_verhaeltnis_median": float(np.median(ve))}
    if p_r:
        sch = np.concatenate(schaetz)
        row["sd_ueber_se"] = float(sch.std() / np.mean(np.concatenate(ses)))
        row["wald_ablehnung"] = float(np.mean(np.abs(np.concatenate(zs)) > 1.96))
    else:
        row["sd_ueber_se"] = None
        row["wald_ablehnung"] = None
    return row


def bruecke(n: int, reps: int) -> dict:
    """Gerade auf y in {0, 1}: Anteil der Testcontainer mit Vorhersage unter 0 oder über 1; RMSE gegen die wahre Wahrscheinlichkeit (Gerade auf [0, 1] gekappt) und Firth."""
    Xt, pt, _ = S.testmenge()
    aus, r_ols, r_firth = [], [], []
    for k in range(reps):
        st = S.ziehe(n, F_SEED + k, 1.0, F_STREAM)
        X = st.X()
        keep = P.voller_rang(X)
        b = L.gerade_ols(X[:, keep], st.y)
        pr = Xt[:, keep] @ b
        aus.append(float(np.mean((pr < 0) | (pr > 1))))
        r_ols.append(float(np.sqrt(np.mean((np.clip(pr, 0, 1) - pt) ** 2))))
        r_firth.append(S.rmse_wahr(voll(L.firth(X[:, keep], st.y)[0], keep)))
    return {"n": n, "reps": reps, "ausserhalb_mittel": float(np.mean(aus)), "ausserhalb_sd": float(np.std(aus)), "rmse_ols": float(np.mean(r_ols)), "rmse_firth": float(np.mean(r_firth))}


def alles(verbose: bool = True) -> dict:
    """Alle Messungen; Laufzeit mehrere Minuten (Normalzahlen der Container aus Dezimalrechnung)."""
    import time

    t0 = time.time()

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)

    out = {"meta": {}}
    binf = S.logit_grenze()
    out["meta"] = {"beta_grenze": [float(v) for v in binf], "mehrverlust_grenze": S.mehrverlust(binf), "rmse_grenze": S.rmse_wahr(binf), "entropie": S.testmenge()[2],
                   "test_n": C.TEST_N, "standgeld_anteil_test": float(S.testmenge()[1].mean())}
    out["haeufigkeit"] = []
    for noise in (1.0, 0.5):
        for n, reps in H_GRID:
            out["haeufigkeit"].append(haeufigkeit(noise, n, reps))
            log(f"Häufigkeit noise {noise} n {n}: {out['haeufigkeit'][-1]['getrennt']} von {reps}")
    out["haeufigkeit_zweiter_strom"] = haeufigkeit(1.0, 100, 300, H2_SEED, H2_STREAM)
    log(f"zweiter Strom: {out['haeufigkeit_zweiter_strom']['getrennt']} von 300")
    out["lp_luecke"] = lp_luecke()
    log(f"LP-Lücke: {out['lp_luecke']}")
    out["was_hilft"] = {}
    for n in C.E4_N:
        out["was_hilft"][str(n)] = was_hilft(n)
        log(f"was hilft n = {n}: gezogen {out['was_hilft'][str(n)]['gezogen']}")
    out["bezug_nicht_getrennt"] = bezug_nicht_getrennt()
    log(f"Bezug: {out['bezug_nicht_getrennt']}")
    out["faustregel"] = []
    for p_r, reps in E_GRID:
        out["faustregel"].append(faustregel(p_r, reps))
        log(f"Faustregel p_R {p_r}: {out['faustregel'][-1]}")
    out["bruecke"] = []
    for n, reps in F_GRID:
        out["bruecke"].append(bruecke(n, reps))
        log(f"Brücke n {n}: {out['bruecke'][-1]}")
    return out
