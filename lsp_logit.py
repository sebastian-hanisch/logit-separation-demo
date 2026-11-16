"""Kern des Logit: IRLS (gleich Newton beim kanonischen Link), Wald und Likelihood-Quotient, die Abbruchregeln, Firth, Ridge, Streichen und der Bruch-Schritt für das Handbeispiel.

Konvention: Ausgang y in {0, 1}; die Spalten von X enthalten den Achsenabschnitt zuerst; p ist die Zahl der Spalten inklusive Achsenabschnitt.
Logit: p_i = 1 / (1 + e^(-η_i)), η = Xβ. Gewicht w_i = p_i (1 - p_i). Ein IRLS-Schritt ist β⁺ = β + (X'WX)⁻¹ X'(y - p).
Stabile Formeln: w = σ(η) σ(-η) und das Residuum r = y σ(-η) - (1 - y) σ(η) vermeiden die Auslöschung von y - p bei |η| über 36 (dort ist p in double gleich 1)."""

import math
from fractions import Fraction

import numpy as np
from scipy import special, stats

import lsp_constants as C


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Grundgrößen
# ---------------------------------------------------------------------------------------------------------------------------------------------
def deviance(X, y, b) -> float:
    """D = 2 Σ [log(1 + e^η) - y η]."""
    eta = X @ b
    return float(2.0 * np.sum(np.logaddexp(0.0, eta) - y * eta))


def deviance_null(y) -> float:
    """Deviance des Modells nur mit Achsenabschnitt (p̂ = Mittel von y)."""
    n, s = len(y), float(np.sum(y))
    return float(-2.0 * (special.xlogy(s, s / n) + special.xlogy(n - s, (n - s) / n)))


def gewicht_residuum(X, y, b):
    eta = X @ b
    w = special.expit(eta) * special.expit(-eta)
    r = y * special.expit(-eta) - (1.0 - y) * special.expit(eta)
    return w, r


def irls_schritt(X, y, b, strafe=None):
    """Ein IRLS-Schritt (Newton-Schritt): Änderung Δ mit (X'WX + P) Δ = X'(y - p) - P b. Wirft numpy.linalg.LinAlgError bei singulärer Matrix."""
    w, r = gewicht_residuum(X, y, b)
    a = X.T @ (w[:, None] * X)
    g = X.T @ r
    if strafe is not None:
        a = a + strafe
        g = g - strafe @ b
    return np.linalg.solve(a, g)


def irls_pfad(X, y, schritte: int, beta0=None, strafe=None):
    """Pfad der Koeffizienten, Zeile k = nach k Schritten (Zeile 0 = Start, voreingestellt 0). Bricht bei singulärer Matrix früher ab (dann kürzer als schritte + 1)."""
    p = X.shape[1]
    b = np.zeros(p) if beta0 is None else np.array(beta0, dtype=float)
    pfad = [b.copy()]
    for _ in range(schritte):
        try:
            b = b + irls_schritt(X, y, b, strafe)
        except np.linalg.LinAlgError:
            break
        pfad.append(b.copy())
    return np.array(pfad)


def devianzen(X, y, pfad):
    return np.array([deviance(X, y, b) for b in pfad])


def newton_pfad(X, y, schritte: int, h: float = 1e-6):
    """Newton auf der Log-Likelihood mit der Hesse-Matrix aus der zweiten Ableitung, hier per Differenzenquotient des Gradienten X'(y - p) (ohne die Formel w = p (1 - p))."""
    p = X.shape[1]
    b = np.zeros(p)
    pfad = [b.copy()]

    def grad(bb):
        return X.T @ (y - special.expit(X @ bb))

    for _ in range(schritte):
        hesse = np.empty((p, p))
        for j in range(p):
            e = np.zeros(p)
            e[j] = h
            hesse[:, j] = (grad(b + e) - grad(b - e)) / (2.0 * h)
        try:
            b = b + np.linalg.solve(-hesse, grad(b))
        except np.linalg.LinAlgError:
            break
        pfad.append(b.copy())
    return np.array(pfad)


def konvergiert_bei(pfad, tol: float = C.CONV_TOL):
    """Erster Schritt k (1, 2, …), dessen Änderung der Koeffizienten unter tol liegt; None, wenn keiner im Pfad."""
    for k in range(1, len(pfad)):
        if np.max(np.abs(pfad[k] - pfad[k - 1])) < tol:
            return k
    return None


def grenzwert(X, y, max_schritte: int = 60):
    """ML-Schätzer als Grenzwert der IRLS-Schritte (nur sinnvoll, wenn er existiert); gibt (β, Schritte bis zur Konvergenz oder None)."""
    pfad = irls_pfad(X, y, max_schritte)
    k = konvergiert_bei(pfad, 1e-13)
    return pfad[-1] if k is None else pfad[k], k


def gerade_ols(X, y):
    """Gerade auf y in {0, 1} (lineares Wahrscheinlichkeitsmodell), kleinste Quadrate."""
    return np.linalg.lstsq(X, y, rcond=None)[0]


def wald_und_lq(X, y, b):
    """Wald-Standardfehler (Modellvarianz (X'WX)⁻¹), z und p je Koeffizient; dazu der Likelihood-Quotient gegen das Modell nur mit Achsenabschnitt (Freiheitsgrade p - 1) mit p-Wert."""
    w, _ = gewicht_residuum(X, y, b)
    cov = np.linalg.inv(X.T @ (w[:, None] * X))
    se = np.sqrt(np.diag(cov))
    z = b / se
    pw = 2.0 * stats.norm.sf(np.abs(z))
    lq = deviance_null(y) - deviance(X, y, b)
    df = X.shape[1] - 1
    return {"se": se, "z": z, "p_wald": pw, "lq": lq, "p_lq": float(stats.chi2.sf(max(lq, 0.0), df)) if df > 0 else float("nan")}


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Abbruchregeln (Exp. 2): wo hält ein Programm an, dessen Pfad nicht konvergiert?
# ---------------------------------------------------------------------------------------------------------------------------------------------
def abbruch_schritt(pfad, dev, regel: str):
    """Schritt k, bei dem die Regel den Pfad anhält (k ≤ Länge des Pfads - 1). Deviance-Regel: |ΔD| / (|D| + 0,1) < 1e-8; feste Schrittzahl: das Ende des Pfads;
    Koeffizientenänderung: größte Änderung < 1e-8. Greift die Regel im Pfad nicht, ist es das Ende des Pfads (zweiter Wert False)."""
    ende = len(pfad) - 1
    if regel == "Feste Schrittzahl":
        return ende, True
    for k in range(1, len(pfad)):
        if regel == "Deviance-Regel":
            hit = abs(dev[k] - dev[k - 1]) / (abs(dev[k]) + 0.1) < C.DEV_RULE_EPS
        else:
            hit = np.max(np.abs(pfad[k] - pfad[k - 1])) < C.COEF_RULE_EPS
        if hit:
            return k, True
    return ende, False


def irls_gekappt(X, y, maxit: int = 25, eps: float = C.DEV_RULE_EPS):
    """Nachbau einer gängigen Programmregel: Wahrscheinlichkeiten aus |η| ≤ 30 gekappt, Abbruch nach |ΔD| / (|D| + 0,1) < eps, höchstens maxit Schritte.
    Aus dem Gedächtnis nachgebaut, nicht gegen eine bestimmte Software geprüft. Rückgabe (β, Schritte, regulär abgebrochen)."""
    p = X.shape[1]
    b = np.zeros(p)

    def dev_of(bb):
        pr = special.expit(np.clip(X @ bb, -C.CAP_ETA, C.CAP_ETA))
        return float(-2.0 * np.sum(y * np.log(pr) + (1.0 - y) * np.log1p(-pr)))

    d_alt = dev_of(b)
    for it in range(1, maxit + 1):
        pr = special.expit(np.clip(X @ b, -C.CAP_ETA, C.CAP_ETA))
        w = pr * (1.0 - pr)
        try:
            b = b + np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (y - pr))
        except np.linalg.LinAlgError:
            return b, it - 1, False
        d_neu = dev_of(b)
        if abs(d_neu - d_alt) / (abs(d_neu) + 0.1) < eps:
            return b, it, True
        d_alt = d_neu
    return b, maxit, False


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Firth, Ridge, Streichen (Exp. 4)
# ---------------------------------------------------------------------------------------------------------------------------------------------
def gestrafte_ll(X, y, b):
    eta = X @ b
    w = special.expit(eta) * special.expit(-eta)
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)) + 0.5 * np.linalg.slogdet(X.T @ (w[:, None] * X))[1])


def firth_score(X, y, b):
    """Modifizierte Score-Gleichung U* = X'(y - p + h (1/2 - p)) mit den Hebeln h_i der gewichteten Hat-Matrix (Jeffreys-Strafe beim kanonischen Link)."""
    eta = X @ b
    pr = special.expit(eta)
    w = special.expit(eta) * special.expit(-eta)
    inv = np.linalg.inv(X.T @ (w[:, None] * X))
    h = w * np.einsum("ij,jk,ik->i", X, inv, X)
    return X.T @ (y - pr + h * (0.5 - pr)), inv


def firth(X, y, maxit: int = 300, tol: float = 1e-9):
    """Firth-Schätzer: Newton auf der gestraften Log-Likelihood ℓ + ½ log|X'WX| mit Schrittweitenhalbierung. Rückgabe (β, Schritte, konvergiert)."""
    p = X.shape[1]
    b = np.zeros(p)
    cur = gestrafte_ll(X, y, b)
    for it in range(1, maxit + 1):
        u, inv = firth_score(X, y, b)
        step = inv @ u
        mx = float(np.max(np.abs(step)))
        if mx > 5.0:
            step = step * 5.0 / mx
        k = 0
        while gestrafte_ll(X, y, b + step) < cur - 1e-12 and k < 40:
            step = step / 2.0
            k += 1
        b = b + step
        cur = gestrafte_ll(X, y, b)
        if float(np.max(np.abs(step))) < tol:
            return b, it, True
    return b, maxit, False


def ridge(X, y, lam: float, maxit: int = 100, tol: float = 1e-10):
    """Gestraftes Logit (Strafe ½ λ Σ β_j² ohne Achsenabschnitt) auf standardisierten Spalten; Koeffizienten in der Skala der Originalspalten."""
    mu, sd = X[:, 1:].mean(axis=0), X[:, 1:].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.column_stack([np.ones(len(y)), (X[:, 1:] - mu) / sd])
    P = np.diag(np.r_[0.0, np.full(Z.shape[1] - 1, float(lam))])
    pfad = irls_pfad(Z, y, maxit, strafe=P)
    k = konvergiert_bei(pfad, tol)
    bz = pfad[-1] if k is None else pfad[k]
    return np.r_[bz[0] - np.sum(bz[1:] * mu / sd), bz[1:] / sd]


def streichen(X, y):
    """Die Spalte mit dem größten |β| (ohne Achsenabschnitt) beim Abbruch der gängigen Regel streichen, dann dieselbe Regel; die gestrichene Spalte bekommt den Koeffizienten 0."""
    b, _, _ = irls_gekappt(X, y)
    j = int(np.argmax(np.abs(b[1:]))) + 1
    b2, _, _ = irls_gekappt(np.delete(X, j, axis=1), y)
    return np.insert(b2, j, 0.0), j


def ml_stabil(X, y, schritte: int = 25):
    """ML mit stabilen Formeln, genau `schritte` Schritte (ohne Kappung, ohne Abbruchregel)."""
    return irls_pfad(X, y, schritte)[-1]


def ml_konvergiert(X, y, max_schritte: int = 100, tol: float = 1e-10):
    """ML bis zur Konvergenz; Rückgabe (β, Schritte, konvergiert)."""
    pfad = irls_pfad(X, y, max_schritte)
    k = konvergiert_bei(pfad, tol)
    return (pfad[-1], len(pfad) - 1, False) if k is None else (pfad[k], k, True)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel in Brüchen
# ---------------------------------------------------------------------------------------------------------------------------------------------
def loese_bruch(a, b):
    """Gauß-Elimination mit Brüchen (kleine Systeme, Spaltenwahl nach erstem Eintrag ungleich 0)."""
    n = len(b)
    m = [[Fraction(v) for v in row] + [Fraction(bi)] for row, bi in zip(a, b)]
    for c in range(n):
        piv = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[piv] = m[piv], m[c]
        m[c] = [v / m[c][c] for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [vr - f * vc for vr, vc in zip(m[r], m[c])]
    return [m[r][n] for r in range(n)]


def erster_schritt_bruch(X_int, y_int):
    """Erster IRLS-Schritt von β = 0 in Brüchen: dort ist p = 1/2, w = 1/4 und die Arbeitsantwort z = 4y - 2, also (X'X) β = 4 X'(y - 1/2)."""
    p = len(X_int[0])
    xtx = [[sum(Fraction(r[a]) * Fraction(r[b]) for r in X_int) for b in range(p)] for a in range(p)]
    rhs = [4 * sum(Fraction(r[a]) * (Fraction(yy) - Fraction(1, 2)) for r, yy in zip(X_int, y_int)) for a in range(p)]
    return loese_bruch(xtx, rhs)


def zwei_gruppen_grenzwert(n0: int, s0: int, n1: int, s1: int):
    """Grenzwert des Logit mit einem 0/1-Merkmal in geschlossener Form: p̂_g = s_g / n_g, Odds und Odds-Ratio als Brüche; β = (ln Odds_0, ln Odds-Ratio). None bei Nullzelle (dann existiert er nicht)."""
    if s0 in (0, n0) or s1 in (0, n1):
        return None
    p0, p1 = Fraction(s0, n0), Fraction(s1, n1)
    o0, o1 = p0 / (1 - p0), p1 / (1 - p1)
    return {"p0": p0, "p1": p1, "odds0": o0, "odds1": o1, "ratio": o1 / o0, "beta": (math.log(o0), math.log(o1 / o0))}


def firth_zwei_gruppen(n0: int, s0: int, n1: int, s1: int):
    """Firth im Zwei-Gruppen-Modell in geschlossener Form: p_g = (s_g + 1/2) / (n_g + 1), „ein halbes in jede Zelle“."""
    p0, p1 = Fraction(2 * s0 + 1, 2 * (n0 + 1)), Fraction(2 * s1 + 1, 2 * (n1 + 1))
    o0, o1 = p0 / (1 - p0), p1 / (1 - p1)
    return {"p0": p0, "p1": p1, "odds0": o0, "odds1": o1, "ratio": o1 / o0, "beta": (math.log(o0), math.log(o1 / o0))}


def design_1d(x):
    """Spalte 1 und Merkmal (Achsenabschnitt zuerst)."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([np.ones(len(x)), x])
