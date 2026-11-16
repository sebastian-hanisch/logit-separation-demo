"""Orakel-Tests logit-separation-demo: Kernroutinen gegen einen anderen Rechenweg (ORACLE-PLAYBOOK §5, DEMO-PLAYBOOK §12).

Orakel: `scipy.optimize.minimize` mit analytischem Gradienten (nicht ohne `jac`: Vorwärtsdifferenzen sind nur auf etwa 1e-4 genau), `scikit-learn` (`LogisticRegression` mit
`C=np.inf` und Newton-Cholesky, `importorskip`; nicht `penalty=None`, das ab scikit-learn 1.8 veraltet ist), die geschlossene Form des Rezessions-LP für ein Merkmal, die
Differenzenquotienten der gestraften Log-Likelihood, die Hesse-Matrix per Differenzenquotient für die Standardfehler und die Aufzählung für die Cover-Formel (in test_logit.py).
Die Orakel selbst sind an den Handbeispielen in test_logit.py geprüft (Brüche, geschlossene Form)."""

import numpy as np
import pytest
from scipy import optimize, special, stats

import lsp_logit as L
import lsp_scenario as S
import lsp_separation as P


def _nicht_getrennt(n, seed0, anzahl, spalten_ohne=()):
    """Container-Stichproben (Designmatrix, y) ohne Trennung und ohne Spalten ohne Variation: die ersten `anzahl` ab dem Zufalls-Seed seed0."""
    out, seed = [], seed0
    while len(out) < anzahl:
        st = S.ziehe(n, seed, 1.0, stream=20)
        seed += 1
        X = st.X()
        X = X[:, P.voller_rang(X)]
        if not P.ist_getrennt(X, st.y):
            out.append((X, st.y))
    return out


def _neg_ll_und_grad(b, X, y):
    eta = X @ b
    return float(np.sum(np.logaddexp(0.0, eta) - y * eta)), X.T @ (special.expit(eta) - y)


def test_irls_gleich_scipy_mit_analytischem_gradienten():
    for X, y in _nicht_getrennt(150, 100, 12):
        b, _, konv = L.ml_konvergiert(X, y)
        assert konv
        res = optimize.minimize(_neg_ll_und_grad, np.zeros(X.shape[1]), args=(X, y), jac=True, method="BFGS", options={"gtol": 1e-10})
        assert np.max(np.abs(b - res.x)) < 1e-5
        assert np.max(np.abs(X.T @ (y - special.expit(X @ b)))) < 1e-7, "der Score ist am Ergebnis null"


def test_irls_gleich_scikit_learn_newton_cholesky():
    lin = pytest.importorskip("sklearn.linear_model")
    for X, y in _nicht_getrennt(150, 200, 8):
        b, _, konv = L.ml_konvergiert(X, y, tol=1e-12)
        sk = lin.LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-12, max_iter=200).fit(X[:, 1:], y)
        assert np.max(np.abs(b - np.r_[sk.intercept_, sk.coef_[0]])) < 1e-7


def test_deviance_gleich_der_summe_der_log_wahrscheinlichkeiten():
    for X, y in _nicht_getrennt(100, 300, 5):
        b = np.random.default_rng(1).normal(0, 0.3, X.shape[1])
        p = special.expit(X @ b)
        assert L.deviance(X, y, b) == pytest.approx(-2 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)), rel=1e-12)


def test_wald_standardfehler_gleich_hesse_matrix_per_differenzenquotient():
    X, y = _nicht_getrennt(200, 400, 1)[0]
    b, _, _ = L.ml_konvergiert(X, y, tol=1e-12)
    p = X.shape[1]

    def grad(bb):
        return -(X.T @ (y - special.expit(X @ bb)))      # Gradient der negativen Log-Likelihood

    h = 1e-5
    H = np.empty((p, p))
    for j in range(p):
        e = np.zeros(p)
        e[j] = h
        H[:, j] = (grad(b + e) - grad(b - e)) / (2 * h)
    se_orakel = np.sqrt(np.diag(np.linalg.inv(H)))
    assert L.wald_und_lq(X, y, b)["se"] == pytest.approx(se_orakel, rel=1e-5)


def test_rezessions_lp_gleich_geschlossener_form_bei_einem_merkmal():
    """Mit einem ganzzahligen Merkmal (Gleichstände erlaubt): getrennt (vollständig oder quasi) genau dann, wenn max x(y=0) <= min x(y=1) oder max x(y=1) <= min x(y=0), auch wenn eine Klasse fehlt."""
    rng = np.random.default_rng(1601)
    gezaehlt = getrennt = 0
    while gezaehlt < 300:
        n = int(rng.integers(3, 9))
        x = rng.integers(-3, 4, n).astype(float)
        y = rng.integers(0, 2, n).astype(float)
        if np.ptp(x) == 0:
            continue
        gezaehlt += 1
        x0, x1 = x[y == 0], x[y == 1]
        form = len(x0) == 0 or len(x1) == 0 or x0.max() <= x1.min() or x1.max() <= x0.min()
        getrennt += form
        assert P.ist_getrennt(L.design_1d(x), y) == form, (x, y)
    assert 0 < getrennt < gezaehlt, "beide Fälle kommen vor"


def test_lp_schwelle_liegt_in_einer_luecke():
    """Optimalwerte der Container-Stichproben: entweder höchstens 1e-12 (überlappend) oder mindestens 1e-3 (getrennt); die Schwelle 1e-9 liegt dazwischen."""
    pos, nul = [], []
    for seed in range(40):
        st = S.ziehe(100, 500 + seed, 1.0, stream=21)
        X = st.X()
        v = P.lp_optimum(X[:, P.voller_rang(X)], st.y)
        (pos if v > 1e-9 else nul).append(v)
    assert pos and nul, "beide Fälle kommen in den 40 Stichproben vor"
    assert min(pos) > 1e-3 and max(abs(v) for v in nul) < 1e-12


def test_firth_score_gleich_gradient_der_gestraften_log_likelihood():
    X, y = _nicht_getrennt(120, 600, 1)[0]
    b = np.random.default_rng(4).normal(0, 0.3, X.shape[1])
    u, _ = L.firth_score(X, y, b)
    h = 1e-6
    num = np.empty(len(b))
    for j in range(len(b)):
        e = np.zeros(len(b))
        e[j] = h
        num[j] = (L.gestrafte_ll(X, y, b + e) - L.gestrafte_ll(X, y, b - e)) / (2 * h)
    assert np.max(np.abs(u - num)) < 1e-6


def test_firth_newton_gleich_bfgs_auf_der_gestraften_log_likelihood():
    X, y = _nicht_getrennt(120, 700, 1)[0]
    b, _, konv = L.firth(X, y, tol=1e-12)
    assert konv
    res = optimize.minimize(lambda bb: -L.gestrafte_ll(X, y, bb), np.zeros(X.shape[1]), jac=lambda bb: -L.firth_score(X, y, bb)[0], method="BFGS", options={"gtol": 1e-9})
    assert np.max(np.abs(b - res.x)) < 1e-5


def test_firth_gleich_ein_halbes_in_jede_zelle_in_200_zaehlungen():
    rng = np.random.default_rng(1601)
    for _ in range(200):
        n0, n1 = (int(v) for v in rng.integers(2, 12, 2))
        s0, s1 = int(rng.integers(0, n0 + 1)), int(rng.integers(0, n1 + 1))
        X = np.column_stack([np.ones(n0 + n1), np.r_[np.zeros(n0), np.ones(n1)]])
        y = np.r_[np.r_[np.ones(s0), np.zeros(n0 - s0)], np.r_[np.ones(s1), np.zeros(n1 - s1)]]
        b, _, _ = L.firth(X, y, tol=1e-12)
        soll = L.firth_zwei_gruppen(n0, s0, n1, s1)["beta"]
        assert b == pytest.approx(soll, abs=1e-8), (n0, s0, n1, s1)


def test_ridge_gleich_scikit_learn_mit_c_gleich_eins_durch_lambda():
    lin = pytest.importorskip("sklearn.linear_model")
    X, y = _nicht_getrennt(100, 800, 1)[0]
    mu, sd = X[:, 1:].mean(axis=0), X[:, 1:].std(axis=0)
    Z = (X[:, 1:] - mu) / sd
    for lam in (0.1, 1.0):
        b = L.ridge(X, y, lam)
        sk = lin.LogisticRegression(C=1.0 / lam, solver="newton-cholesky", tol=1e-12, max_iter=200).fit(Z, y)
        bz = np.r_[sk.intercept_, sk.coef_[0]]
        soll = np.r_[bz[0] - np.sum(bz[1:] * mu / sd), bz[1:] / sd]
        assert np.max(np.abs(b - soll)) < 1e-6


def test_stat_chi2_und_normalverteilung_der_tests():
    """Die p-Werte sind die der Verteilungen: Wald zweiseitig Normal, Likelihood-Quotient Chi-Quadrat mit p - 1 Freiheitsgraden."""
    X, y = _nicht_getrennt(150, 900, 1)[0]
    b, _, _ = L.ml_konvergiert(X, y)
    r = L.wald_und_lq(X, y, b)
    assert r["p_wald"] == pytest.approx(2 * stats.norm.sf(np.abs(r["z"])), rel=1e-12)
    assert r["p_lq"] == pytest.approx(stats.chi2.sf(r["lq"], X.shape[1] - 1), rel=1e-12)
