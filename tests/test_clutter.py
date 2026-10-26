"""Clutter models, estimators and CFAR thresholds against closed forms and Monte Carlo."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, special, stats

from conftest import L_IW
from s1ship import clutter as C


@pytest.mark.parametrize("nu", [0.8, 3.0, 20.0])
def test_k_pdf_is_a_unit_mean_density(nu):
    total = integrate.quad(lambda x: C.k_pdf(x, L_IW, nu), 0, np.inf, limit=200)[0]
    mean = integrate.quad(lambda x: x * C.k_pdf(x, L_IW, nu), 0, np.inf, limit=200)[0]
    assert total == pytest.approx(1.0, abs=1e-6)
    assert mean == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("t", [1.0, 4.0, 10.0, 20.0])
def test_k_sf_matches_tail_integral_of_pdf(t):
    nu = 3.0
    tail = integrate.quad(lambda x: C.k_pdf(x, L_IW, nu), t, np.inf, limit=200)[0]
    assert C.k_sf(t, L_IW, nu) == pytest.approx(tail, rel=1e-5)


def test_k_sf_matches_monte_carlo(rng):
    nu = 2.0
    x = C.sample(2_000_000, L_IW, nu, rng)
    for t in (3.0, 8.0):
        p = C.k_sf(t, L_IW, nu)
        se = np.sqrt(p * (1 - p) / x.size)
        assert abs((x > t).mean() - p) < 4 * se


def test_k_tends_to_gamma_for_smooth_texture():
    # texture variance 1/nu = 1e-7 changes the tail at t = 6 by ~(L t)^2 / (2 nu) = 1.4e-5
    for t in (2.0, 6.0):
        assert C.k_sf(t, L_IW, 1e7) == pytest.approx(float(C.gamma_sf(t, L_IW)), rel=1e-4)
    assert C.k_isf(1e-6, L_IW, np.inf) == pytest.approx(C.gamma_isf(1e-6, L_IW))


def test_enl_estimators_recover_injected_looks(rng):
    for L in (1.0, 4.4, 10.0):
        x = C.sample(1_000_000, L, np.inf, rng)
        assert C.enl_moments(x) == pytest.approx(L, rel=0.02)
        assert C.enl_logcumulant(x) == pytest.approx(L, rel=0.02)


@pytest.mark.parametrize("nu", [1.5, 5.0])
def test_k_shape_estimators_recover_injected_texture(rng, nu):
    x = C.sample(2_000_000, L_IW, nu, rng)
    assert C.k_shape_logcumulant(x, L_IW) == pytest.approx(nu, rel=0.08)
    assert C.k_shape_moments(x, L_IW) == pytest.approx(nu, rel=0.08)


def test_k_shape_is_infinite_for_gamma_clutter(rng):
    x = C.sample(1_000_000, L_IW, np.inf, rng)
    nu = C.k_shape_logcumulant(x, L_IW)
    assert not np.isfinite(nu) or nu > 100


def test_f_quantile_is_right_where_scipy_breaks():
    # SciPy 1.15's stats.f.isf is right for moderate d2 and wrong for d2 > ~5e4;
    # an 81 x 81 window with a 25 x 25 guard and L = 4.4 gives d2 = 52,237.
    for d2 in (1000.0, 10000.0):
        assert C.f_isf(1e-6, 8.8, d2) == pytest.approx(float(stats.f.isf(1e-6, 8.8, d2)), rel=1e-8)
    for d2 in (52236.8, 1e6):
        a = C.f_isf(1e-6, 8.8, d2)
        assert float(C.f_sf(a, 8.8, d2)) == pytest.approx(1e-6, rel=1e-8)
        assert 5.0 < a < 5.1                      # between the n = 1000 and n -> inf values


def test_ca_multiplier_round_trips_and_converges():
    for nu in (np.inf, 3.0):
        a = C.ca_multiplier(1e-5, L_IW, 2000.0, nu)
        assert C.ca_pfa(a, L_IW, 2000.0, nu) == pytest.approx(1e-5, rel=1e-6)
    # n -> infinity: the background mean is known, the multiplier is the clutter quantile
    assert C.ca_multiplier(1e-6, L_IW, 1e7) == pytest.approx(C.gamma_isf(1e-6, L_IW), rel=1e-4)
    assert C.ca_multiplier(1e-6, L_IW, 1e7, 3.0) == pytest.approx(C.k_isf(1e-6, L_IW, 3.0), rel=1e-3)
    # fewer background cells -> noisier mean -> higher multiplier
    assert C.ca_multiplier(1e-6, L_IW, 200.0) > C.ca_multiplier(1e-6, L_IW, 2000.0)


def test_two_parameter_factor():
    assert C.two_parameter_t(1e-6) == pytest.approx(float(stats.norm.isf(1e-6)))
    q = C.gamma_isf(1e-6, L_IW)
    assert C.two_parameter_t(1e-6, "gamma", L_IW) == pytest.approx(np.sqrt(L_IW) * (q - 1))
    # a heavier tail needs a larger factor
    assert C.two_parameter_t(1e-6, "k", L_IW, 2.0) > C.two_parameter_t(1e-6, "gamma", L_IW)


def test_inverse_trigamma():
    for v in (0.5, 3.0, 40.0):
        assert C._inv_trigamma(float(special.polygamma(1, v))) == pytest.approx(v, rel=1e-8)
