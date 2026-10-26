"""Synthetic SAR scenes with injected targets of known position, size and SCR.

Sea clutter is L-look gamma intensity, optionally K-distributed (gamma texture
with shape nu). Each target adds a deterministic scatterer to every look, so in
one pixel with clutter mean mu and target power P

    2L I / mu  ~  noncentral chi-square(2L, 2L P / mu),

the textbook model for a steady target in speckle (Swerling 0). Its detection
probability under CA-CFAR has a closed form (``clutter``-style integral in
``pd_point_theory``), which is what the tests check the detector against.

Two clutter modes:

* ``correlated=False``: independent pixels. Exact statistics, used for the
  false-alarm and detection-probability tests.
* ``correlated=True``: each look is complex Gaussian speckle filtered by a
  Gaussian impulse response (~ the 20 x 22 m IW GRDH resolution on 10 m pixels),
  so neighbouring pixels are correlated as in real GRD data.

Ships are rectangles of given length, width and heading with a random
along-hull brightness profile, blurred by the impulse response. Land (an
optional coast and breakwater) is much brighter than sea and full of point
scatterers, so an unmasked detector fires all over it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import integrate, ndimage, stats

from . import clutter, config, geo
from .scene import Scene, TruthTarget

DEFAULT_TIME = "2021-10-21T13:52:59Z"
PSF_SIGMA_PX = config.S1_PSF_SIGMA_M / config.S1_PIXEL_M   # ~0.9 px: 21 m FWHM on 10 m pixels
LOOK_AZIMUTH_DESC = 283.0                           # S1 descending at ~34 N: flight ~193, looks WNW
INCIDENCE_DEG = 38.0
LAND_CONTRAST_DB = 15.0


def default_grid(shape) -> geo.Grid:
    """A grid centred on the LA/LB AOI, so synthetic AIS has real coordinates."""
    aoi = config.AOIS[config.DEFAULT_AOI]
    epsg = geo.utm_epsg(aoi.lon_c, aoi.lat_c)
    xc, yc = geo.lonlat_to_utm(aoi.lon_c, aoi.lat_c, epsg)
    H, W = shape
    p = config.S1_PIXEL_M
    return geo.Grid(epsg, float(np.round(xc - W * p / 2, -1)), float(np.round(yc + H * p / 2, -1)), p, (H, W))


# --------------------------------------------------------------------------- #
# Target footprints
# --------------------------------------------------------------------------- #
def ship_pattern(length_m: float, width_m: float, heading_deg: float, row: float, col: float,
                 pixel_m: float = config.S1_PIXEL_M, psf_sigma_px: float = PSF_SIGMA_PX,
                 hull_sd: float = 0.5, supersample: int = 5, rng: np.random.Generator | None = None,
                 sidelobes: tuple[float, float, float] | None = None):
    """Footprint of one ship on the pixel grid, peak normalised to 1.

    Returns (patch, r0, c0): the patch's top-left pixel is (r0, c0). The hull is
    a rectangle sampled ``supersample`` x ``supersample`` per pixel, with a
    lognormal brightness that varies every ``pixel_m`` along the hull
    (superstructure, deck cargo), then blurred by the impulse response.

    ``sidelobes = (look_azimuth_deg, level_db, half_length_m)`` adds the
    impulse response's cross: arms along the range (look) and azimuth axes
    through the ship centre, ``level_db`` below the peak next to it and
    falling as 1/(1 + d / 20 m) in power. An approximation of a
    Hamming-weighted response, used to test that shape measurements and
    sidelobe flags cope with it.
    """
    rng = np.random.default_rng() if rng is None else rng
    half = int(np.ceil(0.5 * max(length_m, width_m) / pixel_m + 4 * psf_sigma_px + 2))
    if sidelobes is not None:
        half = max(half, int(np.ceil(sidelobes[2] / pixel_m + 4 * psf_sigma_px + 2)))
    r0, c0 = int(round(row)) - half, int(round(col)) - half
    n = 2 * half + 1
    sub = (np.arange(n * supersample) + 0.5) / supersample - 0.5
    rr = r0 + sub[:, None] - row                      # supersample offsets from the ship centre (px)
    cc = c0 + sub[None, :] - col
    h = np.radians(heading_deg)
    along = (cc * np.sin(h) - rr * np.cos(h)) * pixel_m      # metres along the bow direction
    across = (cc * np.cos(h) + rr * np.sin(h)) * pixel_m
    inside = (np.abs(along) <= length_m / 2) & (np.abs(across) <= width_m / 2)
    n_seg = max(1, int(np.ceil(length_m / pixel_m)))
    seg = np.clip(((along + length_m / 2) / pixel_m).astype(int), 0, n_seg - 1)
    bright = np.exp(hull_sd * rng.standard_normal(n_seg) - hull_sd ** 2 / 2)
    fine = np.where(inside, bright[seg], 0.0)
    patch = fine.reshape(n, supersample, n, supersample).mean(axis=(1, 3))
    if patch.max() == 0:                              # sub-pixel hull: keep its centre pixel
        patch[half, half] = 1.0
    patch = ndimage.gaussian_filter(patch, psf_sigma_px, mode="constant")
    patch /= patch.max()
    if sidelobes is not None:
        look, level_db, arm_m = sidelobes
        arms = np.zeros_like(patch)
        d = np.arange(-arm_m, arm_m + 0.1, pixel_m / 4)
        for az in (look, look - 90.0):                 # range and azimuth axes
            a = np.radians(az)
            rr_ = row - r0 - d * np.cos(a) / pixel_m
            cc_ = col - c0 + d * np.sin(a) / pixel_m
            ok = (rr_ >= 0) & (rr_ < n - 0.5) & (cc_ >= 0) & (cc_ < n - 0.5) & (np.abs(d) > 1.5 * pixel_m)
            np.add.at(arms, (np.round(rr_[ok]).astype(int), np.round(cc_[ok]).astype(int)),
                      0.25 * 10 ** (level_db / 10) / (1 + np.abs(d[ok]) / 20.0))
        patch += ndimage.gaussian_filter(arms, psf_sigma_px, mode="constant") / \
            ndimage.gaussian_filter(np.pad([[1.0]], 3), psf_sigma_px).max()
    return patch, r0, c0


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, scale: float) -> None:
    H, W = canvas.shape
    h, w = patch.shape
    ra, rb = max(r0, 0), min(r0 + h, H)
    ca, cb = max(c0, 0), min(c0 + w, W)
    if ra < rb and ca < cb:
        canvas[ra:rb, ca:cb] += scale * patch[ra - r0:rb - r0, ca - c0:cb - c0]


def target_power(shape, targets, sigma0: float, rng: np.random.Generator,
                 sidelobes: tuple[float, float, float] | None = None) -> np.ndarray:
    """Deterministic target power per pixel, P = sigma0 * 10^(SCR/10) * footprint."""
    P = np.zeros(shape)
    for t in targets:
        amp = sigma0 * 10.0 ** (t.scr_db / 10.0)
        if t.kind == "point":
            r, c = int(round(t.row)), int(round(t.col))
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                P[r, c] += amp
        else:
            patch, r0, c0 = ship_pattern(t.length_m, t.width_m, t.heading_deg, t.row, t.col, rng=rng,
                                         sidelobes=sidelobes)
            _paste(P, patch, r0, c0, amp)
    return P


# --------------------------------------------------------------------------- #
# Land
# --------------------------------------------------------------------------- #
def coast_mask(shape, rng: np.random.Generator) -> np.ndarray:
    """Land along the right edge with a wiggly coast, plus a thin breakwater."""
    H, W = shape
    rows = np.arange(H)
    coast = 0.8 * W + 0.04 * W * np.sin(2 * np.pi * rows / (0.7 * H)) \
        + ndimage.gaussian_filter1d(rng.standard_normal(H), 8) * 0.01 * W * 4
    land = np.arange(W)[None, :] > coast[:, None]
    r = int(0.3 * H)
    c_start = int(0.55 * W)
    land[r - 1:r + 2, c_start:] = True              # 3-px breakwater from the coast into the bay
    return land


# --------------------------------------------------------------------------- #
# Scenes
# --------------------------------------------------------------------------- #
def make_scene(
    shape=(1024, 1024),
    L: float = config.S1_ENL_IW_GRDH,
    nu: float = np.inf,
    sigma0: float = 0.01,
    targets: list[TruthTarget] | None = None,
    land: bool = False,
    correlated: bool = False,
    psf_sigma_px: float = PSF_SIGMA_PX,
    seed: int = config.SEED,
    grid: geo.Grid | None = None,
    time_utc: str = DEFAULT_TIME,
    look_azimuth_deg: float = LOOK_AZIMUTH_DESC,
    incidence_deg: float = INCIDENCE_DEG,
    with_vh: bool = False,
    sidelobe_db: float | None = None,
    sidelobe_arm_m: float = 600.0,
) -> Scene:
    """A synthetic VV (optionally VH) intensity scene with known targets.

    ``sigma0`` is the sea clutter mean (0.01 = -20 dB, typical C-band VV at
    moderate wind and ~38 deg incidence).
    """
    rng = np.random.default_rng(seed)
    targets = list(targets or [])
    landmask = coast_mask(shape, rng) if land else np.zeros(shape, dtype=bool)

    def channel(sig0: float, scr_offset_db: float) -> np.ndarray:
        tgts = [_with_scr(t, t.scr_db + scr_offset_db) for t in targets]
        sl = None if sidelobe_db is None else (look_azimuth_deg, sidelobe_db, sidelobe_arm_m)
        P = target_power(shape, tgts, sig0, rng, sl)
        mu = sig0 * (rng.gamma(nu, 1.0 / nu, size=shape) if np.isfinite(nu) else np.ones(shape))
        if correlated:
            I = _correlated_looks(mu, P, int(round(L)), psf_sigma_px, rng)
        else:
            I = _independent_looks(mu, P, L, rng)
        land_mu = sig0 * 10 ** (LAND_CONTRAST_DB / 10)
        I_land = land_mu * clutter.sample(shape, L, 0.7, rng)
        spikes = rng.random(shape) < 0.002
        I_land[spikes] *= 10 ** 2.5
        return np.where(landmask, I_land, I)

    vv = channel(sigma0, 0.0)
    # VH: cross-pol sea is ~8 dB darker relative to ships than VV at C band, i.e.
    # ships stand out more. ASSUMED offsets, only used to exercise the VH path.
    vh = channel(sigma0 * 10 ** (-0.8), +3.0) if with_vh else None
    return Scene(vv=vv, vh=vh, water=~landmask, grid=grid or default_grid(shape), time_utc=time_utc,
                 source="synthetic", scene_id=f"SYN/seed{seed}", incidence_deg=incidence_deg,
                 look_azimuth_deg=look_azimuth_deg, orbit_pass="DESCENDING", truth=targets,
                 meta=dict(L=L, nu=nu, sigma0=sigma0, correlated=correlated, land=land))


def _with_scr(t: TruthTarget, scr_db: float) -> TruthTarget:
    d = dict(t.__dict__)
    d["scr_db"] = scr_db
    return TruthTarget(**d)


def _independent_looks(mu, P, L, rng):
    """I = mu / (2L) * ncx2(2L, 2L P / mu); P = 0 reduces to L-look gamma."""
    I = mu / (2 * L) * rng.chisquare(2 * L, size=mu.shape)
    on = P > 0
    I[on] = mu[on] / (2 * L) * rng.noncentral_chisquare(2 * L, 2 * L * P[on] / mu[on])
    return I


def _correlated_looks(mu, P, n_looks, psf_sigma_px, rng):
    """Average of n_looks independent looks of filtered complex speckle plus target."""
    shape = mu.shape
    # unit-power normalisation of the filtered white noise
    k = np.zeros((41, 41))
    k[20, 20] = 1.0
    g2 = (ndimage.gaussian_filter(k, psf_sigma_px) ** 2).sum()
    I = np.zeros(shape)
    for _ in range(n_looks):
        c = (ndimage.gaussian_filter(rng.standard_normal(shape), psf_sigma_px, mode="wrap")
             + 1j * ndimage.gaussian_filter(rng.standard_normal(shape), psf_sigma_px, mode="wrap"))
        c /= np.sqrt(2 * g2)
        phase = np.exp(2j * np.pi * rng.random(shape))
        I += np.abs(np.sqrt(mu) * c + np.sqrt(P) * phase) ** 2
    return I / n_looks


# --------------------------------------------------------------------------- #
# Target populations
# --------------------------------------------------------------------------- #
def place_targets(n: int, shape, rng: np.random.Generator, water: np.ndarray | None = None,
                  margin_px: int = 50, min_sep_px: float = 60.0, land_clear_px: int = 30,
                  max_tries: int = 20000):
    """Random target centres at least ``min_sep_px`` apart and clear of land and edges."""
    H, W = shape
    ok = np.ones(shape, dtype=bool) if water is None else water.copy()
    if water is not None and land_clear_px > 0:
        ok = ~ndimage.binary_dilation(~water, iterations=land_clear_px)
    pts: list[tuple[float, float]] = []
    for _ in range(max_tries):
        if len(pts) == n:
            break
        r = rng.uniform(margin_px, H - margin_px)
        c = rng.uniform(margin_px, W - margin_px)
        if not ok[int(r), int(c)]:
            continue
        if pts and np.min(np.hypot(*(np.array(pts) - (r, c)).T)) < min_sep_px:
            continue
        pts.append((r, c))
    if len(pts) < n:
        raise RuntimeError(f"placed only {len(pts)} of {n} targets")
    return np.array(pts)


def point_targets(scr_db_values, shape, rng, **kw) -> list[TruthTarget]:
    pos = place_targets(len(scr_db_values), shape, rng, **kw)
    return [TruthTarget(i, float(r), float(c), 0.0, 0.0, 0.0, float(s), kind="point")
            for i, ((r, c), s) in enumerate(zip(pos, scr_db_values))]


def ship_scr_db(length_m, rng: np.random.Generator, at_100m: float = 18.0, slope: float = 20.0,
                scatter_db: float = 3.0):
    """ASSUMED peak SCR vs length: +20 dB per decade (RCS ~ length^2), 18 dB at 100 m.

    Only used to give the synthetic fleet a plausible spread of brightness so the
    validation code's length binning has something to bin. Real detection
    probability by length comes from the AIS comparison, not from this.
    """
    length_m = np.asarray(length_m, dtype=float)
    return at_100m + slope * np.log10(length_m / 100.0) + scatter_db * rng.standard_normal(length_m.shape)


def fleet(n: int, shape, rng: np.random.Generator, water=None, dark_frac: float = 0.1,
          moving_frac: float = 0.3, length_range=(10.0, 350.0), id0: int = 0, **kw) -> list[TruthTarget]:
    """A mixed fleet: log-uniform lengths, SCR from ``ship_scr_db``, some moving, some dark."""
    pos = place_targets(n, shape, rng, water=water, **kw)
    lengths = np.exp(rng.uniform(*np.log(length_range), size=n))
    scr = ship_scr_db(lengths, rng)
    out = []
    for i, ((r, c), Lm, s) in enumerate(zip(pos, lengths, scr)):
        moving = rng.random() < moving_frac
        out.append(TruthTarget(
            id=id0 + i, row=float(r), col=float(c), length_m=float(Lm), width_m=float(max(4.0, Lm / 6.5)),
            heading_deg=float(rng.uniform(0, 360)), scr_db=float(s), kind="ship",
            ais=bool(rng.random() >= dark_frac),
            sog_kn=float(rng.uniform(6, 16)) if moving else 0.0,
            cog_deg=0.0))
        out[-1].cog_deg = out[-1].heading_deg
    return out


def synthetic_ais(scene: Scene, report_dt_s: float = 180.0) -> pd.DataFrame:
    """MarineCadastre-style AIS rows for the scene's AIS-broadcasting targets.

    A target's true position is its image position minus the SAR azimuth shift
    its motion causes, so the validation's shift correction can be checked.
    Each ship reports ``report_dt_s`` before and after the acquisition.
    """
    t_sar = pd.Timestamp(scene.time_utc).tz_localize(None) if pd.Timestamp(scene.time_utc).tzinfo else pd.Timestamp(scene.time_utc)
    rows = []
    for t in scene.truth:
        if not t.ais or t.kind == "structure":
            continue
        sog = t.sog_kn * 0.514444
        x, y = scene.grid.rowcol_to_xy(t.row, t.col)
        de, dn = geo.azimuth_shift_m(sog, t.cog_deg, scene.look_azimuth_deg, scene.incidence_at(x, y))
        x, y = x - de, y - dn
        for dt in (-report_dt_s, report_dt_s):
            xe = x + sog * dt * np.sin(np.radians(t.cog_deg))
            yn = y + sog * dt * np.cos(np.radians(t.cog_deg))
            lon, lat = geo.utm_to_lonlat(xe, yn, scene.grid.epsg)
            rows.append(dict(MMSI=366000000 + t.id, BaseDateTime=t_sar + pd.Timedelta(seconds=dt),
                             LAT=float(lat), LON=float(lon), SOG=t.sog_kn, COG=t.cog_deg,
                             Heading=t.heading_deg, VesselName=f"SYN{t.id:04d}", VesselType=70,
                             Status=1 if t.sog_kn == 0 else 0, Length=round(t.length_m),
                             Width=round(t.width_m)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Detection theory for a single-pixel target
# --------------------------------------------------------------------------- #
def pd_point_theory(scr_db, pfa: float, L: float, n: float) -> np.ndarray:
    """P(detect) for a single-pixel steady target under gamma CA-CFAR.

    2L I / mu ~ ncx2(2L, 2L SCR) and Z_hat / mu ~ Gamma(nL, 1/(nL)), so
    Pd = E_z[ ncx2.sf(2L alpha z; 2L, 2L SCR) ].
    """
    alpha = clutter.ca_multiplier(pfa, L, n)
    k = n * L
    lo, hi = stats.gamma.ppf(1e-12, k, scale=1 / k), stats.gamma.isf(1e-12, k, scale=1 / k)
    out = []
    for s in np.atleast_1d(scr_db):
        scr = 10 ** (s / 10)
        f = lambda z: stats.ncx2.sf(2 * L * alpha * z, 2 * L, 2 * L * scr) * stats.gamma.pdf(z, k, scale=1 / k)
        out.append(integrate.quad(f, lo, hi, limit=200)[0])
    return np.clip(out, 0.0, 1.0)
