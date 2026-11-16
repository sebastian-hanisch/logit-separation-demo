"""Schnelle, plattformfeste Normalzahlen für große Felder (die Rauschmerkmale der Messung zur Faustregel n ≥ 5p).

Die Normalzahlen der Container (Verweilzeit) kommen aus `lsp_rng.SplitMix64.normal()` (Dezimalrechnung, 78 bis 143 µs je Zahl). Für die Rauschmerkmale braucht die Messung
mehrere Millionen Normalzahlen; dafür steht hier Box-Muller mit den Gleichverteilten der ganzzahligen SplitMix64-Folge (`lsp_rng.u64_array`) und einem Logarithmus,
Sinus und Kosinus, die nur aus Addition, Multiplikation, Division und `frexp` bestehen. Diese Grundrechenarten sind nach IEEE 754 auf jeder Plattform exakt gerundet
(`numpy.log` und `numpy.cos` nicht: ihre letzte Stelle hängt von der SIMD-Fassung ab); die Zahlen sind also auf Windows und Linux bitgleich, auf etwa 1e-15 relativ genau
(Test gegen `numpy.log`/`numpy.cos`)."""

import numpy as np

import lsp_rng as R

_LN2 = 0.6931471805599453
_PI_2 = 1.5707963267948966
_TWO_PI = 6.283185307179586
_K_LOG = 14                                   # Entwicklung von atanh bis f^29 (f höchstens 0,172: Rest unter 1e-22)
_K_TRIG = 16                                  # Taylor-Entwicklungen bis x^32 (x höchstens π/2)
_LOG_COEF = [1.0 / (2 * k + 1) for k in range(_K_LOG, -1, -1)]
_COS_COEF, _SIN_COEF = [], []
_f = 1.0
for _k in range(0, 2 * _K_TRIG + 3):
    if _k > 0:
        _f *= _k
    if _k % 2 == 0:
        _COS_COEF.append((-1.0) ** (_k // 2) / _f)
    else:
        _SIN_COEF.append((-1.0) ** ((_k - 1) // 2) / _f)


def _horner(coefs_desc, t):
    acc = np.full_like(t, coefs_desc[0])
    for c in coefs_desc[1:]:
        acc = acc * t + c
    return acc


def log_arith(x):
    """Natürlicher Logarithmus für x > 0 nur mit Grundrechenarten und `frexp` (deterministisch)."""
    m, e = np.frexp(np.asarray(x, dtype=np.float64))
    klein = m < 0.7071067811865476
    m = np.where(klein, m * 2.0, m)
    e = np.where(klein, e - 1, e).astype(np.float64)
    f = (m - 1.0) / (m + 1.0)
    return e * _LN2 + 2.0 * f * _horner(_LOG_COEF, f * f)


def _sincos_kurz(r):
    """cos und sin für 0 <= r <= π/2 (Taylor-Entwicklung, Horner)."""
    r2 = r * r
    c = _horner(_COS_COEF[::-1], r2)
    s = r * _horner(_SIN_COEF[::-1], r2)
    return c, s


def cos_sin_2pi(u):
    """cos(2πu) und sin(2πu) für u in [0, 1) (Quadrant über 4u, Entwicklung im Rest)."""
    t = 4.0 * np.asarray(u, dtype=np.float64)
    q = np.floor(t)
    c, s = _sincos_kurz((t - q) * _PI_2)
    qi = q.astype(np.int64)
    cos_ = np.select([qi == 0, qi == 1, qi == 2], [c, -s, -c], default=s)
    sin_ = np.select([qi == 0, qi == 1, qi == 2], [s, c, -s], default=-c)
    return cos_, sin_


def normal_array(seed: int, n: int):
    """n Standardnormalzahlen aus SplitMix64(seed): Box-Muller auf den Paaren (u1, u2) der Folge; je Paar zwei Zahlen (Kosinus- und Sinuszweig)."""
    m = (n + 1) // 2
    u = (R.u64_array(seed, 2 * m) >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    u1, u2 = u[0::2], u[1::2]
    rad = np.sqrt(-2.0 * log_arith(1.0 - u1))                 # 1 - u1 liegt in (0, 1]
    c, s = cos_sin_2pi(u2)
    z = np.empty(2 * m)
    z[0::2], z[1::2] = rad * c, rad * s
    return z[:n]


def normal_matrix(seed: int, rows: int, cols: int):
    return normal_array(seed, rows * cols).reshape(rows, cols)
