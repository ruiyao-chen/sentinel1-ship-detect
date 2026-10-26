"""From detected pixels to targets: clustering, centroid, length, orientation.

A ship spans many pixels and its return is uneven (bright superstructure,
darker hull), so CFAR typically marks it as several nearby fragments.
Detections closer than ``merge_radius_px`` are joined into one target
(connected components of the mask dilated by that radius); each target's
properties are then measured on its *detected* pixels only.

Shape is measured on the target's core, the detected pixels within
``CORE_DB`` (15 dB) of its peak, so that the sidelobe cross of a bright ship
does not count as hull. Length estimators:

* ``length_m``: extent of *all* detected pixels along the core's major axis
  (first to last pixel, plus one), the raw Feret length;
* ``length_core_m``: the same for the core pixels only;
* ``length_moment_m``: sqrt(12 * variance along the axis) of the core, exact
  for a uniformly filled segment;
* ``length_psf_m``: the core length with the impulse-response blur removed.

The blur: each end of the hull stays above the cut T for a distance set by how
far the peak exceeds it. For a Gaussian impulse response of width sigma, a hull
end of power P falls below T at sigma * Phi^-1(1 - T/P) past the true end, so

    length_psf_m = length_core_m - 2 sigma Phi^-1(1 - T/P_peak),

with T the larger of the CFAR threshold and the core cut, and sigma from the
published S1 resolution (not fitted to any data). Using the peak power
over-corrects ships whose ends are dimmer than their brightest point; the
synthetic tests and the AIS comparison measure what is left.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage, stats

from . import config
from .geo import Grid


def _disk(r: int) -> np.ndarray:
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def label_targets(detections: np.ndarray, merge_radius_px: int = config.MERGE_RADIUS_PX):
    """Label detected pixels; pixels within ``merge_radius_px`` share a label."""
    if merge_radius_px > 0:
        joined = ndimage.binary_dilation(detections, structure=_disk(merge_radius_px))
        lab, n = ndimage.label(joined, structure=np.ones((3, 3)))
        lab = np.where(detections, lab, 0)
    else:
        lab, n = ndimage.label(detections, structure=np.ones((3, 3)))
    return lab, n


def _axes(r, c, k, n):
    """Principal axis, second moment along it, and Feret extents of labelled pixel sets."""
    bincount = lambda v: np.bincount(k, weights=v, minlength=n + 1)[1:]
    cnt = np.maximum(np.bincount(k, minlength=n + 1)[1:].astype(float), 1)
    mr, mc = bincount(r.astype(float)) / cnt, bincount(c.astype(float)) / cnt
    dr, dc = r - mr[k - 1], c - mc[k - 1]
    srr, scc, src = bincount(dr * dr) / cnt, bincount(dc * dc) / cnt, bincount(dr * dc) / cnt
    tr, det = srr + scc, srr * scc - src * src
    lam1 = tr / 2 + np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    theta = 0.5 * np.arctan2(2 * src, scc - srr)               # major-axis angle from +col (east)
    ur, uc = np.sin(theta), np.cos(theta)                      # unit vector (row, col)
    ids = np.arange(1, n + 1)
    proj = dr * ur[k - 1] + dc * uc[k - 1]
    perp = -dr * uc[k - 1] + dc * ur[k - 1]
    ext = np.asarray(ndimage.maximum(proj, k, ids)) - np.asarray(ndimage.minimum(proj, k, ids)) + 1
    ext_w = np.asarray(ndimage.maximum(perp, k, ids)) - np.asarray(ndimage.minimum(perp, k, ids)) + 1
    return lam1, ur, uc, mr, mc, ext, ext_w


def measure(
    detections: np.ndarray,
    img: np.ndarray,
    background: np.ndarray,
    grid: Grid,
    threshold: np.ndarray | None = None,
    merge_radius_px: int = config.MERGE_RADIUS_PX,
    min_pixels: int = config.MIN_TARGET_PIXELS,
    max_length_m: float = config.MAX_TARGET_LENGTH_M,
    psf_sigma_m: float = config.S1_PSF_SIGMA_M,
    core_db: float = config.CORE_DB,
    confirm: np.ndarray | None = None,
    confirm_ratio: np.ndarray | None = None,
) -> pd.DataFrame:
    """One row per target, after the size filters.

    Shape (length, width, orientation) is measured on the target's *core*: its
    detected pixels within ``core_db`` of its peak. That keeps the impulse
    response's sidelobes (about -21 dB for Sentinel-1's Hamming weighting),
    which draw a bright cross through strong targets, out of the length.
    ``threshold`` is the CFAR's per-pixel intensity threshold, used for the
    blur correction. ``confirm`` (bool image, e.g. dilated VH detections) and
    ``confirm_ratio`` (e.g. VH / VH background) add per-target columns
    ``confirmed`` and ``confirm_peak_db``.
    """
    lab, n = label_targets(detections, merge_radius_px)
    r, c = np.nonzero(lab)
    cols = ["row", "col", "x", "y", "lon", "lat", "n_pixels", "peak", "peak_db_over_clutter",
            "length_m", "length_core_m", "length_moment_m", "length_psf_m", "width_m", "orientation_deg"]
    if n == 0 or r.size == 0:
        return pd.DataFrame(columns=cols)
    ids = np.arange(1, n + 1)
    k = lab[r, c]
    w = img[r, c]
    cnt = np.bincount(k, minlength=n + 1)[1:].astype(float)
    present = cnt > 0
    sw = np.bincount(k, weights=w, minlength=n + 1)[1:]
    row_c = np.bincount(k, weights=w * r, minlength=n + 1)[1:] / np.where(sw > 0, sw, 1)   # intensity-weighted
    col_c = np.bincount(k, weights=w * c, minlength=n + 1)[1:] / np.where(sw > 0, sw, 1)

    peak = np.asarray(ndimage.maximum(w, k, ids))
    peak_idx = np.array(ndimage.maximum_position(img, lab, ids))
    bg = background[peak_idx[:, 0], peak_idx[:, 1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        peak_db = 10 * np.log10(peak / bg)

    core_frac = 10 ** (-core_db / 10)
    core = w >= peak[k - 1] * core_frac
    lam1, ur, uc, mr, mc, ext_core, ext_w = _axes(r[core], c[core], k[core], n)
    # Feret length of *all* detected pixels along the core's axis (sidelobes included)
    proj_all = (r - mr[k - 1]) * ur[k - 1] + (c - mc[k - 1]) * uc[k - 1]
    ext_all = np.asarray(ndimage.maximum(proj_all, k, ids)) - np.asarray(ndimage.minimum(proj_all, k, ids)) + 1

    p = grid.pixel_m
    if threshold is not None:
        t_eff = np.maximum(threshold[peak_idx[:, 0], peak_idx[:, 1]] / peak, core_frac)
        overshoot = psf_sigma_m * stats.norm.isf(np.clip(t_eff, 1e-12, 0.5))
    else:
        overshoot = np.zeros(n)
    length_psf = np.maximum(ext_core * p - 2 * overshoot, p)
    x, y = grid.rowcol_to_xy(row_c, col_c)
    lon, lat = grid.rowcol_to_lonlat(row_c, col_c)
    # bearing of the major axis, clockwise from north (row axis points south)
    orient = np.degrees(np.arctan2(uc, -ur)) % 180.0
    cols = dict(
        row=row_c, col=col_c, x=x, y=y, lon=lon, lat=lat, n_pixels=cnt.astype(int), peak=peak,
        peak_db_over_clutter=peak_db, length_m=ext_all * p, length_core_m=ext_core * p,
        length_moment_m=np.sqrt(12 * lam1) * p, length_psf_m=length_psf, width_m=ext_w * p,
        orientation_deg=orient)
    if confirm is not None:
        cols["confirmed"] = np.asarray(ndimage.maximum(confirm[r, c].astype(float), k, ids)) > 0
    if confirm_ratio is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            cols["confirm_peak_db"] = 10 * np.log10(np.asarray(ndimage.maximum(confirm_ratio[r, c], k, ids)))
    df = pd.DataFrame(cols)[present]
    keep = (df.n_pixels >= min_pixels) & (df.length_core_m <= max_length_m)
    df = df[keep].reset_index(drop=True)
    df.attrs["n_clusters"] = int(present.sum())
    df.attrs["n_removed_small"] = int((present & (cnt < min_pixels)).sum())
    df.attrs["n_removed_long"] = int(present.sum() - keep.sum() - df.attrs["n_removed_small"])
    return df
