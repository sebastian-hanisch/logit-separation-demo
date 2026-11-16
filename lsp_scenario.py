"""Szenario: der Containerstrom (Vehikel) und die Handbeispiele. Alle Zahlen des Stroms sind erfunden (kein Echtdatenbezug).

Merkmale je Container: Art (Import, Export, Umschlag, Leer), Kunde (12, nicht im Modell), Zollhalt (nur Import), Wochenende, Vorlauf in Tagen (Export und Umschlag).
Die log-Verweilzeit ist normalverteilt mit Median und Streuung nach Art (heteroskedastisch); Standgeld ist fällig, wenn die Verweilzeit über 7 Tage liegt. Die wahre
Wahrscheinlichkeit P(Verweilzeit > 7 Tage | Merkmale) ist bekannt: Φ((μ - ln 7) / s). Sie ist nicht logistisch in den Merkmalen; darum bleibt auch bei beliebig vielen Daten
ein Rest zwischen jedem Logit und der Wahrheit (die „Grenze der Logit-Form“).

Zufall nur aus dem ganzzahligen SplitMix64 (`lsp_rng`): je Container 7 Ziehungen in fester Reihenfolge (Art, Kunde, Zollhalt, Wochentag, Vorlauf, zwei für die Normalzahl der
Verweilzeit). Die Merkmale werden vektorisiert aus `u64_array` gebildet (gleich der Klasse), die Normalzahl je Container aus `SplitMix64.normal()` (Dezimalrechnung, plattformfest).
Neu geschrieben für diese Demo nach den Annahmen der `verweilzeit-demo`; kein Import von dort."""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import special

import lsp_constants as C
import lsp_logit as L
import lsp_rng as R

_TWO53 = float(1 << 53)
LN_STAND = math.log(C.STAND_TAGE)


def stream_seed(seed: int, stream: int = 0) -> int:
    """Zufalls-Seed der Folge zu (Zufalls-Seed, Strom): gemischt, damit benachbarte Seeds keine verschobenen Folgen teilen."""
    return R.SplitMix64((int(seed) * 0x9E3779B97F4A7C15 + int(stream) * 0xD6E8FEB86659FD93 + 0x1234567) & R.MASK).next_u64()


@lru_cache(maxsize=1)
def kunden_effekte():
    """12 Kundeneffekte auf die log-Verweilzeit: N(0, 0,3²) aus SplitMix64(KUNDEN_SEED)."""
    g = R.SplitMix64(C.KUNDEN_SEED)
    return np.array([C.U_SD * g.normal() for _ in range(C.K_KUNDEN)])


def log_median_und_s(art, kunde, zoll, we, vorlauf, noise: float = 1.0):
    """Bedingte Verteilung der log-Verweilzeit: N(μ, s²). Vektorisiert."""
    art = np.asarray(art)
    u = kunden_effekte()[kunde] * np.take(np.array(C.U_SCALE), art)
    med = np.select(
        [art == 0, art == 1, art == 2],
        [3.5 * np.exp(0.7 * zoll + 0.30 * we), 0.6 + 0.85 * vorlauf, 0.3 + 0.8 * vorlauf],
        default=7.0 * np.exp(0.30 * we),
    )
    return np.log(med) + u, np.take(np.array(C.S_BASE), art) * noise


@dataclass(frozen=True)
class Stichprobe:
    art: np.ndarray
    kunde: np.ndarray
    zoll: np.ndarray
    we: np.ndarray
    vorlauf: np.ndarray
    mu: np.ndarray
    s: np.ndarray
    dwell: np.ndarray | None        # Verweilzeit in Tagen (None bei reinen Merkmalen)
    y: np.ndarray | None            # Standgeld fällig (1) oder nicht (0)

    @property
    def n(self) -> int:
        return len(self.art)

    def X(self):
        return design(self.art, self.zoll, self.we, self.vorlauf)

    def p_wahr(self):
        return special.ndtr((self.mu - LN_STAND) / self.s)


def design(art, zoll, we, vorlauf):
    """Spalten: Achsenabschnitt, Export, Umschlag, Leer (Referenz Import), Zollhalt, Wochenende, Vorlauf (Tage)."""
    art = np.asarray(art)
    return np.column_stack([np.ones(len(art)), art == 1, art == 2, art == 3, zoll, we, vorlauf]).astype(float)


def _merkmale(u64, noise):
    uni = (u64 >> np.uint64(11)).astype(np.float64) / _TWO53
    cum = np.cumsum(np.array(C.ART_P))
    art = np.minimum(np.searchsorted(cum, uni[:, 0], side="right"), 3)
    kunde = (u64[:, 1] % np.uint64(C.K_KUNDEN)).astype(np.int64)
    zoll = ((uni[:, 2] < C.P_ZOLL) & (art == 0)).astype(float)
    we = ((u64[:, 3] % np.uint64(7)) >= np.uint64(5)).astype(float)
    r = uni[:, 4]
    vorlauf = np.where(art == 1, 1.0 + 9.0 * r, np.where(art == 2, 0.5 + 5.5 * r, 0.0))
    mu, s = log_median_und_s(art, kunde, zoll, we, vorlauf, noise)
    return art, kunde, zoll, we, vorlauf, mu, s, uni


def merkmale(n: int, seed: int, noise: float = 1.0, stream: int = 0) -> Stichprobe:
    """Nur die Merkmale (und die wahre Verteilung), ohne Verweilzeit: schnell, für Testmengen."""
    u64 = R.u64_array(stream_seed(seed, stream), C.N_DRAWS * n).reshape(n, C.N_DRAWS)
    art, kunde, zoll, we, vorlauf, mu, s, _ = _merkmale(u64, noise)
    return Stichprobe(art, kunde, zoll, we, vorlauf, mu, s, None, None)


def ziehe(n: int, seed: int, noise: float = 1.0, stream: int = 0) -> Stichprobe:
    """n Container aus dem Strom: Merkmale, Verweilzeit (log-normal) und Standgeld (Verweilzeit über 7 Tage). Gleiche Eingabe, gleiche Stichprobe, auf jeder Plattform."""
    u64 = R.u64_array(stream_seed(seed, stream), C.N_DRAWS * n).reshape(n, C.N_DRAWS)
    art, kunde, zoll, we, vorlauf, mu, s, uni = _merkmale(u64, noise)
    u1 = np.maximum(uni[:, 5], 1e-12)
    u2 = uni[:, 6]
    z = np.array([math.sqrt(-2.0 * R.exact_log(a)) * R.exact_cos(2.0 * math.pi * b) for a, b in zip(u1, u2)])
    dwell = np.exp(mu + s * z)
    return Stichprobe(art, kunde, zoll, we, vorlauf, mu, s, dwell, (dwell > C.STAND_TAGE).astype(float))


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Testmenge und Grenze der Logit-Form
# ---------------------------------------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def testmenge():
    """100 000 Testcontainer (nur Merkmale): Designmatrix, wahre Wahrscheinlichkeit, Entropie der wahren Wahrscheinlichkeiten (nat)."""
    t = merkmale(C.TEST_N, C.TEST_SEED, 1.0, stream=90)
    p = np.clip(t.p_wahr(), 1e-12, 1 - 1e-12)
    ent = float(-(p * np.log(p) + (1 - p) * np.log1p(-p)).mean())
    return t.X(), p, ent


def mehrverlust(b, test=None) -> float:
    """Mittlerer Log-Verlust des Modells gegen die wahre Wahrscheinlichkeit minus die Entropie der wahren Wahrscheinlichkeiten (nat); 0 hieße: Wahrheit getroffen."""
    X, p, ent = test or testmenge()
    eta = X @ np.asarray(b, dtype=float)
    return float((p * np.logaddexp(0.0, -eta) + (1.0 - p) * np.logaddexp(0.0, eta)).mean() - ent)


def rmse_wahr(b, test=None) -> float:
    X, p, _ = test or testmenge()
    return float(np.sqrt(np.mean((special.expit(X @ np.asarray(b, dtype=float)) - p) ** 2)))


@lru_cache(maxsize=1)
def logit_grenze():
    """Pseudowahre Koeffizienten: das Logit mit dem kleinsten erwarteten Log-Verlust gegen die wahre Wahrscheinlichkeit (IRLS mit der wahren Wahrscheinlichkeit als Ausgang,
    auf 400 000 Merkmalssätzen). Unabhängig von jeder Stichprobe; der Rest an Mehrverlust ist die Grenze der Logit-Form."""
    t = merkmale(C.LIMIT_N, C.LIMIT_SEED, 1.0, stream=91)
    X, p = t.X(), t.p_wahr()
    pfad = L.irls_pfad(X, p, 40)
    k = L.konvergiert_bei(pfad, 1e-12)
    return pfad[-1] if k is None else pfad[k]


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Handbeispiele
# ---------------------------------------------------------------------------------------------------------------------------------------------
def handbeispiel(name: str):
    """(x, y) als Zahlenfelder; name aus lsp_constants (E1_BEISPIELE, E2_DATEN)."""
    tab = {
        C.E1_BEISPIELE[0]: (C.ZOLL9_X, C.ZOLL9_Y),
        C.E1_BEISPIELE[1]: (C.VORLAUF8_X, C.VORLAUF8_Y),
        C.E2_DATEN[0]: (C.SEP6_X, C.SEP6_Y),
        C.E2_DATEN[1]: (C.QUASI8_X, C.QUASI8_Y),
    }
    x, y = tab[name]
    return np.array(x, dtype=float), np.array(y, dtype=float)
