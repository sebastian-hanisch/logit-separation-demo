"""Gemeinsame Fixtures: die Handbeispiele, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import os
import sys
from pathlib import Path

# Ein BLAS-Faden: die Tests rechnen kleine Matrizen; mehrere Fäden verlangsamen sie unter Last.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lsp_constants as C  # noqa: E402
import lsp_logit as L  # noqa: E402


@pytest.fixture
def zoll9():
    """Neun Container: Zollhalt 0 vier Container mit einem Standgeld, Zollhalt 1 fünf Container mit vier Standgeldern (gesetzte Zahlen)."""
    x, y = np.array(C.ZOLL9_X, float), np.array(C.ZOLL9_Y, float)
    return L.design_1d(x), y


@pytest.fixture
def sechs():
    """Sechs Punkte x = -3, -2, -1, 1, 2, 3 mit y = 0, 0, 0, 1, 1, 1: vollständig getrennt."""
    x, y = np.array(C.SEP6_X, float), np.array(C.SEP6_Y, float)
    return L.design_1d(x), y


@pytest.fixture
def quasi8():
    x, y = np.array(C.QUASI8_X, float), np.array(C.QUASI8_Y, float)
    return L.design_1d(x), y
