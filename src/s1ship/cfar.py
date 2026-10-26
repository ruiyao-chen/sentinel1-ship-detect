"""CFAR ship detection on SAR intensity.

Each pixel (the cell under test) is compared with the clutter around it: a
square background window with a square guard window cut out of its centre, so
that the target's own pixels do not inflate the background. Window sums come
from integral images, so the cost per pixel does not depend on window size.

Land is handled by weights: background statistics are averaged over *valid
water* pixels only, so a coastline or a breakwater in the ring does not raise
the threshold, and land pixels are never tested. A pixel whose ring holds too
few water pixels gets no decision.

Two detectors:

* **Cell-averaging (CA-CFAR).** I > alpha(n) * mean, with alpha from the clutter
  model (gamma or K) and the number n of background cells actually used, so the
  false-alarm rate holds at the design value wherever the ring is partly masked.
* **Two-parameter CFAR.** (I - mean) / sd > t, the classic ship detector, with t
  either Gaussian (textbook) or taken from the clutter model.

Optional censoring: a second pass recomputes the background with first-pass
detections (dilated) removed, so neighbouring ships in a crowded anchorage do
not mask each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from . import clutter, config


# --------------------------------------------------------------------------- #
# Window statistics
# --------------------------------------------------------------------------- #
def box_sum(a: np.ndarray, h: int) -> np.ndarray:
    """Sum of ``a`` over the (2h+1) x (2h+1) window centred on each pixel.

    Pixels outside the image count as zero. Computed from a padded integral
    image in float64.
    """
    H, W = a.shape
    k = 2 * h + 1
    ii = np.zeros((H + 2 * h + 1, W + 2 * h + 1))
    ii[1:, 1:] = np.pad(np.asarray(a, dtype=float), h).cumsum(0).cumsum(1)
    return ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]


def ring_stats(img: np.ndarray, weight: np.ndarray, guard: int, background: int):
    """Weighted mean, standard deviation and cell count over the background ring."""
    w = weight.astype(float)
    x = np.where(weight, img, 0.0)
    ring = lambda a: box_sum(a, background) - box_sum(a, guard)
    n = np.rint(ring(w))
    s1 = ring(x)
    s2 = ring(x * x)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        var = np.maximum(s2 / n - mean * mean, 0.0)
    return mean, np.sqrt(var), n


def ring_size(guard: int, background: int) -> int:
    return (2 * background + 1) ** 2 - (2 * guard + 1) ** 2


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
@dataclass
class CfarResult:
    detections: np.ndarray          # bool, target pixels
    valid: np.ndarray               # bool, pixels where a decision was made
    background: np.ndarray          # local clutter mean (final pass)
    n_background: np.ndarray        # background cells used per pixel
    threshold: np.ndarray           # intensity threshold per pixel (nan where not valid)
    method: str
    pfa: float
    L: float
    nu: float
    info: dict = field(default_factory=dict)

    @property
    def n_tested(self) -> int:
        return int(self.valid.sum())

    @property
    def n_detected(self) -> int:
        return int(self.detections.sum())


def _weights(img, water, exclude):
    w = np.isfinite(img) & (img > 0)
    if water is not None:
        w &= water
    if exclude is not None:
        w &= ~exclude
    return w


def _decide(img, water_ok, mean, sd, n, method, pfa, L, nu, t, min_n):
    valid = water_ok & (n >= min_n) & (mean > 0)
    thr = np.full(img.shape, np.nan)
    if method == "ca":
        alpha = clutter.ca_multiplier_table(pfa, L, n[valid], nu)
        thr[valid] = alpha * mean[valid]
    else:
        thr[valid] = mean[valid] + t * sd[valid]
    det = np.zeros(img.shape, dtype=bool)
    det[valid] = img[valid] > thr[valid]
    return det, valid, thr


def detect(
    img: np.ndarray,
    water: np.ndarray | None = None,
    method: str = "ca",
    pfa: float = config.CFAR.pfa,
    L: float = config.S1_ENL_IW_GRDH,
    nu: float = np.inf,
    guard: int = config.CFAR.guard_px,
    background: int = config.CFAR.background_px,
    min_background_frac: float = config.CFAR.min_background_frac,
    censor: bool = config.CFAR.censor,
    censor_dilate: int = config.CFAR.censor_dilate_px,
    t: float | None = None,
    t_model: str = "gaussian",
) -> CfarResult:
    """Run CA-CFAR (``method="ca"``) or two-parameter CFAR (``method="2p"``).

    ``water`` is True where detection is allowed and where background cells may
    be drawn from (land already buffered). ``nu = inf`` is gamma clutter; finite
    ``nu`` is K clutter with that texture shape. For the two-parameter detector,
    ``t`` defaults to ``clutter.two_parameter_t(pfa, t_model, L, nu)``.
    """
    if method not in ("ca", "2p"):
        raise ValueError(f"unknown CFAR method {method!r}")
    if method == "2p" and t is None:
        t = clutter.two_parameter_t(pfa, t_model, L, nu)
    img = np.asarray(img, dtype=float)
    water_ok = _weights(img, water, None)
    min_n = min_background_frac * ring_size(guard, background)

    mean, sd, n = ring_stats(img, water_ok, guard, background)
    det, valid, thr = _decide(img, water_ok, mean, sd, n, method, pfa, L, nu, t, min_n)
    info = {"first_pass_detections": int(det.sum())}

    if censor and det.any():
        grown = ndimage.binary_dilation(det, iterations=censor_dilate) if censor_dilate > 0 else det
        mean, sd, n = ring_stats(img, water_ok & ~grown, guard, background)
        det, valid, thr = _decide(img, water_ok, mean, sd, n, method, pfa, L, nu, t, min_n)

    info["t"] = t
    return CfarResult(det, valid, mean, n, thr, method, pfa, L, nu, info)


# --------------------------------------------------------------------------- #
# Clutter fitting on a real scene
# --------------------------------------------------------------------------- #
@dataclass
class ClutterFit:
    L: float                 # looks used (nominal, or measured if the sea is smoother)
    nu: float                # K texture shape (inf = gamma)
    enl_measured: float      # ENL of the normalised water intensity (log-cumulant)
    n_pixels: int
    ratio_sample: np.ndarray  # I / local mean on clutter pixels, for tail plots


def fit_clutter(
    img: np.ndarray,
    water: np.ndarray,
    L_nominal: float = config.S1_ENL_IW_GRDH,
    guard: int = config.CFAR.guard_px,
    background: int = config.CFAR.background_px,
    max_sample: int = 2_000_000,
    seed: int = config.SEED,
) -> ClutterFit:
    """Fit the clutter model to the scene's own water pixels.

    The intensity is divided by its local (ring) mean, which removes the
    incidence-angle trend and wind patches. The K shape is fitted with L fixed
    at the nominal looks; log-cumulants barely move for a few ship pixels, so a
    first fit on all water is close. Targets found with that model at Pfa 1e-6
    (dilated) are then removed and the fit is repeated. If the normalised sea
    is smoother than nominal-L gamma (resampling during terrain correction can
    do that), the gamma model with the measured ENL is used instead.
    """
    rng = np.random.default_rng(seed)
    clean = water.copy()
    for it in range(2):
        mean, _, n = ring_stats(img, clean, guard, background)
        ok = clean & (n >= 0.25 * ring_size(guard, background)) & (mean > 0)
        ratio = img[ok] / mean[ok]
        if ratio.size > max_sample:
            ratio = rng.choice(ratio, max_sample, replace=False)
        enl = clutter.enl_logcumulant(ratio)
        nu = clutter.k_shape_logcumulant(ratio, L_nominal)
        L = L_nominal if np.isfinite(nu) else enl
        if it == 0:
            targets = detect(img, water, "ca", pfa=1e-6, L=L, nu=nu, guard=guard,
                             background=background, censor=False).detections
            clean = water & ~ndimage.binary_dilation(targets, iterations=3)
    return ClutterFit(L=float(L), nu=float(nu), enl_measured=float(enl),
                      n_pixels=int(ok.sum()), ratio_sample=ratio)
