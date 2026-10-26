"""Sea-clutter intensity models, parameter estimators, and CFAR thresholds.

Multi-look SAR intensity over a homogeneous sea is gamma distributed: L looks of
exponential speckle averaged, mean mu, variance mu^2 / L. A rougher sea
modulates the local mean ("texture"). When the texture is itself gamma with
shape nu, the intensity is K distributed (the product model; Jakeman & Pusey
1976, Ward 1981). nu -> infinity recovers the gamma model, and small nu means a
heavier tail, i.e. more bright clutter spikes that a gamma-based detector would
call ships.

Everything here works on *unit-mean* intensity; the CFAR divides by the local
background mean before applying these thresholds.

CA-CFAR thresholds account for the noise in the background estimate. With N
independent gamma background cells the test ratio I / Z_hat is exactly
F(2L, 2NL)-distributed (Z_hat ~ Gamma(NL)), so the multiplier is an F quantile.
For K clutter Z_hat is approximated by a gamma with the same mean and variance,
and P(I > a Z_hat) is averaged over the texture of the cell under test.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import integrate, optimize, special, stats


# --------------------------------------------------------------------------- #
# Distributions (unit mean)
# --------------------------------------------------------------------------- #
def gamma_sf(t, L: float):
    """P(I > t) for unit-mean L-look gamma intensity."""
    return stats.gamma.sf(t, L, scale=1.0 / L)


def gamma_isf(p: float, L: float) -> float:
    return float(stats.gamma.isf(p, L, scale=1.0 / L))


def f_sf(x, d1: float, d2: float):
    """F(d1, d2) survival function via the regularised incomplete beta function.

    Written out because ``scipy.stats.f.isf`` (SciPy 1.15) returns wrong values
    once d2 exceeds ~5e4, which a 81 x 81 background window with L = 4.4 does.
    """
    x = np.asarray(x, dtype=float)
    return special.betaincc(d1 / 2, d2 / 2, d1 * x / (d1 * x + d2))


def f_isf(p: float, d1: float, d2: float) -> float:
    """Inverse of ``f_sf``: x with P(F > x) = p."""
    b = special.betainccinv(d1 / 2, d2 / 2, p)
    return float(d2 / d1 * b / (1.0 - b))


def k_pdf(x, L: float, nu: float):
    """Unit-mean K-distributed intensity density with L looks and texture shape nu.

    p(x) = 2 (L nu)^((L+nu)/2) x^((L+nu)/2 - 1) K_{nu-L}(2 sqrt(L nu x)) / (G(L) G(nu)),
    evaluated in logs with the exponentially scaled Bessel function.
    """
    x = np.asarray(x, dtype=float)
    z = 2.0 * np.sqrt(L * nu * x)
    with np.errstate(divide="ignore"):
        logp = (np.log(2.0) + 0.5 * (L + nu) * np.log(L * nu) + (0.5 * (L + nu) - 1.0) * np.log(x)
                - special.gammaln(L) - special.gammaln(nu) + np.log(special.kve(nu - L, z)) - z)
    return np.exp(logp)


def _texture_average(fun, nu: float) -> float:
    """E_tau[fun(tau)] for unit-mean gamma texture tau ~ Gamma(nu, 1/nu).

    Integrated over s = ln(tau) so both the bulk and the far tail of the
    texture are resolved; the limits cover all but 1e-25 of the texture mass.
    """
    lo = np.log(stats.gamma.ppf(1e-25, nu, scale=1.0 / nu))
    hi = np.log(stats.gamma.isf(1e-25, nu, scale=1.0 / nu))

    def integrand(s):
        tau = np.exp(s)
        return fun(tau) * stats.gamma.pdf(tau, nu, scale=1.0 / nu) * tau

    val, _ = integrate.quad(integrand, lo, hi, limit=400, epsabs=0.0, epsrel=1e-9)
    return float(val)


def k_sf(t: float, L: float, nu: float) -> float:
    """P(I > t) for unit-mean K-distributed intensity."""
    if not np.isfinite(nu):
        return float(gamma_sf(t, L))
    return _texture_average(lambda tau: special.gammaincc(L, L * t / tau), nu)


def k_isf(p: float, L: float, nu: float) -> float:
    """Threshold t with P(I > t) = p for unit-mean K intensity."""
    if not np.isfinite(nu):
        return gamma_isf(p, L)
    return _solve_log(lambda t: k_sf(t, L, nu), p, start=gamma_isf(p, L))


def k_variance(L: float, nu: float) -> float:
    """Variance of unit-mean K intensity: (1 + 1/L)(1 + 1/nu) - 1."""
    inv_nu = 0.0 if not np.isfinite(nu) else 1.0 / nu
    return (1.0 + 1.0 / L) * (1.0 + inv_nu) - 1.0


def sample(shape, L: float, nu: float = np.inf, rng: np.random.Generator | None = None) -> np.ndarray:
    """Unit-mean gamma (nu = inf) or K intensity samples, independent per pixel."""
    rng = np.random.default_rng() if rng is None else rng
    x = rng.gamma(L, 1.0 / L, size=shape)
    if np.isfinite(nu):
        x *= rng.gamma(nu, 1.0 / nu, size=shape)
    return x


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def enl_moments(x) -> float:
    """Equivalent number of looks, mean^2 / variance (gamma model)."""
    x = np.asarray(x, dtype=float).ravel()
    return float(x.mean() ** 2 / x.var())


def log_cumulants(x) -> tuple[float, float, float]:
    """First three sample cumulants of ln(x) (the Mellin-kind statistics)."""
    lx = np.log(np.asarray(x, dtype=float).ravel())
    m = lx.mean()
    d = lx - m
    return float(m), float((d ** 2).mean()), float((d ** 3).mean())


def _inv_trigamma(y: float) -> float:
    """v with psi_1(v) = y (psi_1 is decreasing on v > 0)."""
    f = lambda lv: special.polygamma(1, np.exp(lv)) - y
    return float(np.exp(optimize.brentq(f, np.log(1e-4), np.log(1e7))))


def enl_logcumulant(x) -> float:
    """ENL from the second log-cumulant: var(ln I) = psi_1(L) for gamma intensity."""
    return _inv_trigamma(log_cumulants(x)[1])


def k_shape_logcumulant(x, L: float, nu_max: float = 1e3) -> float:
    """K texture shape nu given L, from var(ln I) = psi_1(L) + psi_1(nu).

    Log-cumulants are far less sensitive to a few bright ships than the
    second moment. Returns inf when the sample is no rougher than L-look gamma.
    """
    excess = log_cumulants(x)[1] - special.polygamma(1, L)
    if excess <= special.polygamma(1, nu_max):
        return np.inf
    return _inv_trigamma(excess)


def k_shape_moments(x, L: float) -> float:
    """K texture shape nu from the normalised second moment, given L."""
    x = np.asarray(x, dtype=float).ravel()
    r = (x ** 2).mean() / x.mean() ** 2
    inv_nu = r / (1.0 + 1.0 / L) - 1.0
    return np.inf if inv_nu <= 0 else float(1.0 / inv_nu)


# --------------------------------------------------------------------------- #
# CFAR thresholds
# --------------------------------------------------------------------------- #
def ca_pfa(alpha: float, L: float, n: float, nu: float = np.inf) -> float:
    """P(I > alpha * Z_hat) when Z_hat averages n independent background cells.

    Gamma clutter: exact, I / Z_hat ~ F(2L, 2nL).
    K clutter: Z_hat ~ Gamma(k) with k = n / var_K (moment match), so given the
    texture tau of the cell under test, I / Z_hat ~ tau * F(2L, 2k).
    """
    if not np.isfinite(nu):
        return float(f_sf(alpha, 2 * L, 2 * n * L))
    k = n / k_variance(L, nu)
    return _texture_average(lambda tau: f_sf(alpha / tau, 2 * L, 2 * k), nu)


@lru_cache(maxsize=4096)
def ca_multiplier(pfa: float, L: float, n: float, nu: float = np.inf) -> float:
    """CA-CFAR multiplier alpha: declare a target where I > alpha * Z_hat."""
    if not np.isfinite(nu):
        return float(f_isf(pfa, 2 * L, 2 * n * L))
    return _solve_log(lambda a: ca_pfa(a, L, n, nu), pfa, start=k_isf(pfa, L, nu))


def ca_multiplier_table(pfa: float, L: float, n_values, nu: float = np.inf) -> np.ndarray:
    """ca_multiplier over many background sizes (masked rings vary per pixel).

    The multiplier changes slowly with n, so it is computed on a log-spaced
    grid and interpolated in 1/n.
    """
    n_values = np.asarray(n_values, dtype=float)
    lo, hi = max(n_values.min(), 4.0), max(n_values.max(), 5.0)
    grid = np.unique(np.round(np.geomspace(lo, hi, 24)))
    a = np.array([ca_multiplier(pfa, L, float(n), nu) for n in grid])
    return np.interp(1.0 / np.clip(n_values, lo, hi), (1.0 / grid)[::-1], a[::-1])


def two_parameter_t(pfa: float, model: str = "gaussian", L: float = 4.4, nu: float = np.inf) -> float:
    """Two-parameter CFAR factor t: declare a target where (I - mean) / sd > t.

    "gaussian" is the textbook choice (t = Phi^-1(1 - pfa)); it assumes
    Gaussian clutter and so under-thresholds skewed speckle. "gamma" / "k" put
    the same test on the clutter model's own quantile.
    """
    if model == "gaussian":
        return float(stats.norm.isf(pfa))
    nu_ = np.inf if model == "gamma" else nu
    q = k_isf(pfa, L, nu_)
    return float((q - 1.0) / np.sqrt(k_variance(L, nu_)))


def _solve_log(sf, p: float, start: float) -> float:
    """Solve sf(t) = p for t > 0 by bracketing in log t (sf decreasing)."""
    g = lambda lt: np.log(sf(np.exp(lt))) - np.log(p)
    lo = hi = np.log(start)
    while g(lo) < 0:
        lo -= 0.5
    while g(hi) > 0:
        hi += 0.5
    return float(np.exp(optimize.brentq(g, lo, hi, xtol=1e-10)))
