"""Trennung: Prüfung per Rezessions-Linearprogramm, vollständig gegen quasi, Cover-Formel und Zählung der Zellen.

Begriffe (nach Kosmidis und Firth, Abschn. 1): mit s_i = 2 y_i - 1 heißen die Daten *vollständig getrennt*, wenn ein β mit s_i x_i'β > 0 für alle i existiert,
*quasi-vollständig getrennt*, wenn nur „≥“ gilt und für mindestens einen Punkt Gleichheit, sonst *überlappend*. Bei vollständiger oder quasi-vollständiger Trennung hat der
ML-Schätzer des Logit keinen endlichen Wert (Albert und Anderson 1984, hier nur über Kosmidis/Firth bekannt).
Prüfung hier: Linearprogramm „maximiere Σ s_i x_i'd unter s_i x_i'd ≥ 0, -1 ≤ d_j ≤ 1“; ein Optimum über LP_SCHWELLE heißt: es gibt eine Richtung d, in der die
Log-Likelihood nie abnimmt und für mindestens einen Punkt wächst (Rezessionsrichtung), der Schätzer existiert nicht."""

import math
from fractions import Fraction
from itertools import product

import numpy as np
from scipy import optimize

import lsp_constants as C


def lp_optimum(X, y) -> float:
    """Optimalwert des Rezessions-LP (0: überlappend; positiv: getrennt)."""
    s = 2.0 * np.asarray(y, dtype=float) - 1.0
    a = s[:, None] * X
    res = optimize.linprog(-a.sum(axis=0), A_ub=-a, b_ub=np.zeros(len(s)), bounds=[(-1.0, 1.0)] * X.shape[1], method="highs")
    return float(-res.fun)


def ist_getrennt(X, y) -> bool:
    """Trennung (vollständig oder quasi): kein endlicher ML-Schätzer. Voraussetzung: X hat vollen Spaltenrang (Spalten ohne Variation vorher entfernen, `voller_rang`)."""
    return lp_optimum(X, y) > C.LP_SCHWELLE


def ist_vollstaendig_getrennt(X, y) -> bool:
    """Gibt es β mit s_i x_i'β ≥ 1 für alle i? (Zulässigkeit; sieht quasi-vollständige Trennung nicht.)"""
    s = 2.0 * np.asarray(y, dtype=float) - 1.0
    res = optimize.linprog(np.zeros(X.shape[1]), A_ub=-(s[:, None] * X), b_ub=-np.ones(len(s)), bounds=[(None, None)] * X.shape[1], method="highs")
    return res.status == 0


def art_der_trennung(X, y) -> str:
    """„überlappend“, „vollständig“ oder „quasi“."""
    if not ist_getrennt(X, y):
        return "überlappend"
    return "vollständig" if ist_vollstaendig_getrennt(X, y) else "quasi"


def voller_rang(X):
    """Indizes der Spalten, die bleiben: der Achsenabschnitt und jede Spalte mit Variation."""
    return [0] + [j for j in range(1, X.shape[1]) if np.ptp(X[:, j]) > 0]


def cover_anteil(n: int, p: int) -> Fraction:
    """Anteil der 2^n Beschriftungen, die n Punkte in allgemeiner Lage im R^(p-1) (p Spalten inklusive Achsenabschnitt) vollständig trennen: 2 Σ_{k<p} C(n-1, k) / 2^n
    (Cover 1965, hier nur über Candès und Sur 2020 bekannt; für kleine Fälle durch Aufzählung nachgerechnet, nicht allgemein bewiesen)."""
    return Fraction(2 * sum(math.comb(n - 1, k) for k in range(p)), 2 ** n)


def aufzaehlung_trennbar(X) -> int:
    """Zahl der Beschriftungen y in {0, 1}^n, die X vollständig trennen (Aufzählung aller 2^n; nur für kleine n)."""
    n = X.shape[0]
    return sum(1 for lab in product([0.0, 1.0], repeat=n) if ist_vollstaendig_getrennt(X, np.array(lab)))


def nullzellen(art, zoll, wochenende, y):
    """Die sechs Einzelgruppen (jede der vier Arten, Zollhalt = 1, Wochenende = 1; keine Kreuzzellen) mit mindestens einem Container, in denen alle gleich ausgehen: Liste (Name, Anzahl, Standgeld)."""
    out = []
    zellen = [(f"Art {C.ART_LABELS[a]}", art == a) for a in range(4)] + [("Zollhalt", zoll == 1), ("Wochenende", wochenende == 1)]
    for name, m in zellen:
        k = int(np.sum(m))
        if k > 0:
            s = int(np.sum(np.asarray(y)[m]))
            if s == 0 or s == k:
                out.append((name, k, s))
    return out
