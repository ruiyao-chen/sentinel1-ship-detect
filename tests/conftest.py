"""Shared fixtures. Every test builds synthetic inputs with a known answer and
runs offline; nothing here touches Earth Engine, Planetary Computer or AIS
downloads."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from s1ship import clutter  # noqa: E402

L_IW = 4.4


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def gamma_image(shape, rng, L=L_IW, nu=np.inf, mean=0.01):
    """Independent-pixel gamma (nu = inf) or K clutter with the given mean."""
    return mean * clutter.sample(shape, L, nu, rng)


def far_tolerance(n_tested: int, pfa: float, overdispersion: float = 1.5, k: float = 4.0) -> float:
    """Allowed |false alarms - expected|: k binomial sigmas, inflated because
    neighbouring pixels share background windows."""
    return k * overdispersion * np.sqrt(n_tested * pfa)
