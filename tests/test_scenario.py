"""Der Containerstrom und die Zufallsquelle: vektorisierte Folge gleich der Klasse, Merkmale gleich der skalaren Ziehung, wahre Wahrscheinlichkeit gleich der Häufigkeit."""

import math

import numpy as np
import pytest
from scipy import stats

import lsp_constants as C
import lsp_rng as R
import lsp_rngvec as V
import lsp_scenario as S


def test_u64_array_gleich_der_klasse():
    g = R.SplitMix64(123)
    a = R.u64_array(123, 200)
    assert all(int(a[i]) == g.next_u64() for i in range(200))


def test_sehr_grosse_seeds_wie_die_klasse():
    for seed in ((1 << 64) - 1, 2 ** 63 + 5):
        g = R.SplitMix64(seed)
        erwartet = [g.next_u64() for _ in range(3)]
        assert [int(v) for v in R.u64_array(seed, 3)] == erwartet


def test_schnelle_normalzahlen_log_und_trig_gegen_numpy():
    x = np.exp(np.random.default_rng(1).uniform(-30, 30, 20000))
    assert np.max(np.abs(V.log_arith(x) - np.log(x))) < 1e-13
    u = np.random.default_rng(2).random(20000)
    c, s = V.cos_sin_2pi(u)
    assert np.max(np.abs(c - np.cos(2 * np.pi * u))) < 1e-14 and np.max(np.abs(s - np.sin(2 * np.pi * u))) < 1e-14


def test_schnelle_normalzahlen_verteilung():
    z = V.normal_array(5, 200_000)
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01
    assert stats.kstest(z[:50_000], "norm").pvalue > 0.001
    assert V.normal_array(5, 1000)[:10] == pytest.approx(V.normal_array(5, 10), abs=0), "gleiche Folge, egal wie viele Zahlen"


def test_merkmale_gleich_der_skalaren_ziehung_mit_splitmix64():
    """Je Container 7 Ziehungen: Art, Kunde, Zollhalt, Wochentag, Vorlauf, zwei für die Normalzahl; die Klasse `SplitMix64` liefert dieselben Merkmale und (bis auf Rundung) dieselbe Verweilzeit."""
    n, seed = 60, 5
    s = S.ziehe(n, seed, 1.0)
    g = R.SplitMix64(S.stream_seed(seed, 0))
    for i in range(n):
        art = g.choice(C.ART_P)
        kunde = g.below(C.K_KUNDEN)
        zoll = 1.0 if (g.random() < C.P_ZOLL and art == 0) else 0.0
        we = 1.0 if g.below(7) >= 5 else 0.0
        r = g.random()
        vor = 1 + 9 * r if art == 1 else (0.5 + 5.5 * r if art == 2 else 0.0)
        z = g.normal()
        assert (art, kunde, zoll, we) == (s.art[i], s.kunde[i], s.zoll[i], s.we[i]) and vor == s.vorlauf[i]
        assert s.dwell[i] == pytest.approx(math.exp(s.mu[i] + s.s[i] * z), rel=1e-12)


def test_merkmale_ohne_normalzahl_sind_gleich_den_merkmalen_der_stichprobe():
    a, b = S.merkmale(500, 3), S.ziehe(500, 3)
    assert np.array_equal(a.art, b.art) and np.array_equal(a.vorlauf, b.vorlauf) and np.array_equal(a.mu, b.mu)


def test_wertebereiche_der_merkmale():
    s = S.ziehe(3000, 1)
    assert set(np.unique(s.art)) == {0, 1, 2, 3} and set(np.unique(s.we)) <= {0.0, 1.0}
    assert np.all(s.zoll[s.art != 0] == 0)
    assert np.all(s.vorlauf[s.art == 0] == 0) and np.all(s.vorlauf[s.art == 3] == 0)
    assert s.vorlauf[s.art == 1].min() >= 1 and s.vorlauf[s.art == 1].max() <= 10 and s.vorlauf[s.art == 2].min() >= 0.5 and s.vorlauf[s.art == 2].max() <= 6
    assert np.array_equal(s.y, (s.dwell > 7).astype(float))


def test_artanteile_und_zollhalt_anteil():
    s = S.merkmale(100_000, 11)
    for a, p in enumerate(C.ART_P):
        assert abs(np.mean(s.art == a) - p) < 0.005
    assert abs(np.mean(s.zoll[s.art == 0]) - C.P_ZOLL) < 0.01 and abs(np.mean(s.we) - 2 / 7) < 0.005


def test_wahre_wahrscheinlichkeit_gleich_der_haeufigkeit_im_standgeld():
    """Mittel der wahren Wahrscheinlichkeiten gegen den Anteil mit Standgeld in 20 000 gezogenen Containern: innerhalb von vier Standardfehlern."""
    s = S.ziehe(20_000, 77)
    p = s.p_wahr()
    se = math.sqrt(float(np.sum(p * (1 - p)))) / len(p)
    assert abs(s.y.mean() - p.mean()) < 4 * se


def test_wahre_wahrscheinlichkeit_je_art_gleich_haeufigkeit():
    s = S.ziehe(30_000, 78)
    p = s.p_wahr()
    for a in range(4):
        m = s.art == a
        se = math.sqrt(float(np.sum(p[m] * (1 - p[m])))) / m.sum()
        assert abs(s.y[m].mean() - p[m].mean()) < 4 * se


def test_testmenge_und_grenze():
    X, p, ent = S.testmenge()
    assert X.shape == (C.TEST_N, 7) and 0 < ent < math.log(2)
    b = S.logit_grenze()
    assert 0 < S.mehrverlust(b) < S.mehrverlust(np.zeros(7))
    # kein Logit unterschreitet die Grenze: eine kleine Störung erhöht den erwarteten Verlust
    rng = np.random.default_rng(3)
    for _ in range(5):
        assert S.mehrverlust(b + rng.normal(0, 0.05, 7)) > S.mehrverlust(b)


def test_kundeneffekte_sind_gleich_der_klasse():
    g = R.SplitMix64(C.KUNDEN_SEED)
    assert list(S.kunden_effekte()) == [C.U_SD * g.normal() for _ in range(C.K_KUNDEN)]


def test_handbeispiele():
    x, y = S.handbeispiel(C.E1_BEISPIELE[0])
    assert (x.sum(), y.sum()) == (5, 5) and y[x == 0].sum() == 1 and y[x == 1].sum() == 4
    x, y = S.handbeispiel(C.E2_DATEN[1])
    assert len(x) == 8 and y.sum() == 4
