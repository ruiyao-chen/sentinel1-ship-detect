"""Validation: SAR detections against AIS (real) or injected truth (synthetic).

Matching is a one-to-one assignment (Hungarian) of detections to vessels on
distance, inside a gate of ``MATCH_GATE_M`` plus half the AIS length: the AIS
antenna can sit anywhere along the hull, and the SAR centroid is the hull's
middle. Vessel positions are the SAR-apparent ones (azimuth shift applied).

Reported:

* **Detection probability by vessel length**, among AIS vessels whose apparent
  position is on the buffered water mask, with Wilson 95 % intervals.
* **Unmatched detections per km^2** of tested water. This is an *upper bound*
  on the false-alarm rate: it also counts vessels without AIS, fixed
  structures, and azimuth ambiguities.
* **Explained and unexplained detections.** An unmatched detection near a
  matched vessel is a ``fragment`` of it; one on the range or azimuth axis of
  a much brighter target is a ``sidelobe``; one an azimuth-ambiguity offset
  along-track from a much brighter target is an ``ambiguity``. What is left is
  ``unexplained``: vessels without AIS, structures, or clutter.
* **Dark candidates**: unexplained detections at least ``DARK_MIN_LENGTH_M``
  long (and VH-confirmed when VH is available).
* **Length agreement** of matched pairs (SAR estimate vs AIS length).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

from . import config, geo
from .scene import Scene


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def match(dets: pd.DataFrame, vessels: pd.DataFrame, gate_m: float = config.MATCH_GATE_M,
          length_col: str | None = "length_m") -> pd.DataFrame:
    """One-to-one matches (det_idx, vessel_idx, dist_m) within the gate."""
    empty = pd.DataFrame(columns=["det_idx", "vessel_idx", "dist_m"])
    if len(dets) == 0 or len(vessels) == 0:
        return empty
    D = np.hypot(dets.x.to_numpy()[:, None] - vessels.x.to_numpy()[None, :],
                 dets.y.to_numpy()[:, None] - vessels.y.to_numpy()[None, :])
    half = 0.5 * np.nan_to_num(vessels[length_col].to_numpy(float), nan=0.0) if length_col else 0.0
    gate = gate_m + half
    big = 1e12
    cost = np.where(D <= gate[None, :] if np.ndim(gate) else D <= gate, D, big)
    r, c = optimize.linear_sum_assignment(cost)
    ok = cost[r, c] < big
    return pd.DataFrame(dict(det_idx=dets.index.to_numpy()[r[ok]], vessel_idx=vessels.index.to_numpy()[c[ok]],
                             dist_m=D[r[ok], c[ok]]))


def wilson(k, n, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pd_by_length(vessels: pd.DataFrame, detected: np.ndarray, bins=config.LENGTH_BINS_M,
                 length_col: str = "length_m") -> pd.DataFrame:
    L = vessels[length_col].to_numpy(float)
    edges = list(bins) + [np.inf]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (L >= lo) & (L < hi)
        rows.append((f"{lo:g}-{hi:g} m" if np.isfinite(hi) else f">= {lo:g} m", int(m.sum()), int(detected[m].sum())))
    m = ~np.isfinite(L)
    rows.append(("unknown", int(m.sum()), int(detected[m].sum())))
    rows.append(("all", len(L), int(detected.sum())))
    df = pd.DataFrame(rows, columns=["length_bin", "n_vessels", "n_detected"])
    df["pd"] = df.n_detected / df.n_vessels.where(df.n_vessels > 0)
    df["ci_low"], df["ci_high"] = wilson(df.n_detected, df.n_vessels.where(df.n_vessels > 0))
    return df


# --------------------------------------------------------------------------- #
# Explaining unmatched detections: ambiguities and sidelobes
# --------------------------------------------------------------------------- #
def flag_ambiguities(dets: pd.DataFrame, look_azimuth_deg: float, offset_m,
                     tol_m: float = 250.0, min_ratio_db: float = 15.0) -> np.ndarray:
    """True for detections one ambiguity offset along-track from a much brighter one.

    ``offset_m`` is a scalar or one value per detection (the offset of that
    detection's own ghosts, which depends on its sub-swath PRF).
    """
    offset_m = np.broadcast_to(np.asarray(offset_m, dtype=float), (len(dets),))
    if len(dets) < 2 or look_azimuth_deg is None or not np.isfinite(offset_m).any():
        return np.zeros(len(dets), dtype=bool)
    fl = np.radians(look_azimuth_deg - 90.0)
    u = np.array([np.sin(fl), np.cos(fl)])
    P = dets[["x", "y"]].to_numpy()
    pk = dets.peak.to_numpy()
    brighter = 10 * np.log10(pk[None, :] / pk[:, None]) >= min_ratio_db      # [i, j]: j >> i
    out = np.zeros(len(dets), dtype=bool)
    for s in (+1, -1):
        ghost = P + s * offset_m[:, None] * u                                 # where j's ghost would land
        d = np.hypot(P[:, None, 0] - ghost[None, :, 0], P[:, None, 1] - ghost[None, :, 1])
        out |= ((d < tol_m) & brighter).any(axis=1)
    return out


def flag_sidelobes(dets: pd.DataFrame, look_azimuth_deg: float, tol_m: float = 30.0,
                   max_dist_m: float = 2000.0, min_ratio_db: float = 12.0) -> np.ndarray:
    """True for detections on the range or azimuth axis of a much brighter one.

    The impulse response's sidelobes run along range (the look direction) and
    azimuth (the flight direction); fragments of that cross that break away
    from the bright target are detected on their own.
    """
    if len(dets) < 2 or look_azimuth_deg is None:
        return np.zeros(len(dets), dtype=bool)
    P = dets[["x", "y"]].to_numpy()
    pk = dets.peak.to_numpy()
    dx = P[:, None, 0] - P[None, :, 0]
    dy = P[:, None, 1] - P[None, :, 1]
    brighter = 10 * np.log10(pk[None, :] / pk[:, None]) >= min_ratio_db
    out = np.zeros(len(dets), dtype=bool)
    for az in (look_azimuth_deg, look_azimuth_deg - 90.0):
        a = np.radians(az)
        along = dx * np.sin(a) + dy * np.cos(a)
        across = -dx * np.cos(a) + dy * np.sin(a)
        out |= ((np.abs(across) < tol_m) & (np.abs(along) < max_dist_m) & brighter).any(axis=1)
    return out


def ambiguity_offsets(dets: pd.DataFrame, scene: Scene) -> np.ndarray:
    """Per-detection azimuth-ambiguity offset from its incidence angle and sub-swath PRF."""
    from .data import subswath_prf
    inc = scene.incidence_at(dets.x.to_numpy(), dets.y.to_numpy())
    prfs = scene.meta.get("prfs_hz") or list(config.IW_PRF_NOMINAL_HZ)
    prf = np.array([subswath_prf(prfs, float(i)) for i in inc])
    return geo.azimuth_ambiguity_m(inc, prf)


# --------------------------------------------------------------------------- #
# Full comparison
# --------------------------------------------------------------------------- #
@dataclass
class Validation:
    dets: pd.DataFrame                     # detections + matched flag, vessel id, flags
    vessels: pd.DataFrame                  # evaluated (on-water) vessels + detected flag
    pairs: pd.DataFrame
    pd_table: pd.DataFrame
    water_km2: float
    summary: dict = field(default_factory=dict)

    @property
    def dark(self) -> pd.DataFrame:
        return self.dets[self.dets.dark_candidate]


def validate(dets: pd.DataFrame, vessels: pd.DataFrame, water_km2: float, scene: Scene | None = None,
             gate_m: float = config.MATCH_GATE_M, length_est: str = "length_psf_m",
             id_col: str = "mmsi", structures: np.ndarray | None = None,
             structure_radius_m: float = config.NEAR_STRUCTURE_M) -> Validation:
    """Compare detections with vessels (AIS rows from ``ais.project_to_scene`` or truth).

    ``vessels`` needs x, y, length_m and on_water (and optionally in_scene).
    Detections are matched against *every* vessel in the scene, including
    ones berthed inside the land buffer, so a moored ship's hull poking out
    of the buffer is not counted as unexplained; detection probability is
    computed over on-water vessels only.

    Each unmatched detection is then explained, if it can be, as a
    ``fragment`` (within the gate of a vessel already matched: a second piece
    of the same ship), a ``sidelobe`` or an ``ambiguity`` of a brighter target,
    or ``near_structure`` (within ``structure_radius_m`` of a fixed structure in
    ``structures``, e.g. a platform's edge or its tender).
    The rest are ``unexplained``; those at least DARK_MIN_LENGTH_M long (and
    VH-confirmed, when a ``confirmed`` column exists) are dark candidates.
    """
    v_all = vessels[vessels.in_scene] if "in_scene" in vessels else vessels
    pairs = match(dets, v_all, gate_m)
    dets = dets.copy()
    dets["matched"] = dets.index.isin(pairs.det_idx)
    dets["vessel_id"] = pd.Series(v_all.loc[pairs.vessel_idx, id_col].to_numpy(), index=pairs.det_idx.to_numpy())
    v = v_all[v_all.on_water].copy()
    v["detected"] = v.index.isin(pairs.vessel_idx)

    un = ~dets.matched
    if len(v_all) and len(dets):
        D = np.hypot(dets.x.to_numpy()[:, None] - v_all.x.to_numpy()[None, :],
                     dets.y.to_numpy()[:, None] - v_all.y.to_numpy()[None, :])
        gate = gate_m + 0.5 * np.nan_to_num(v_all.length_m.to_numpy(float), nan=0.0)
        dets["fragment"] = un & (D <= gate[None, :]).any(axis=1)
    else:
        dets["fragment"] = False
    look = None if scene is None else scene.look_azimuth_deg
    if look is not None and len(dets):
        dets["sidelobe"] = un & ~dets.fragment & flag_sidelobes(dets, look)
        dets["ambiguity"] = un & ~dets.fragment & ~dets.sidelobe & flag_ambiguities(dets, look, ambiguity_offsets(dets, scene))
    else:
        dets["sidelobe"] = dets["ambiguity"] = False
    explained = dets.fragment | dets.sidelobe | dets.ambiguity
    dets["near_structure"] = False
    if structures is not None and scene is not None and structures.any() and len(dets):
        from scipy import ndimage
        dist = ndimage.distance_transform_edt(~structures) * scene.grid.pixel_m
        ri = np.clip(np.round(dets.row.to_numpy()).astype(int), 0, structures.shape[0] - 1)
        ci = np.clip(np.round(dets.col.to_numpy()).astype(int), 0, structures.shape[1] - 1)
        dets["near_structure"] = un & ~explained & (dist[ri, ci] <= structure_radius_m)
    dets["unexplained"] = un & ~explained & ~dets.near_structure
    confirmed = dets.confirmed if "confirmed" in dets else True
    dets["dark_candidate"] = dets.unexplained & confirmed & (dets[length_est] >= config.DARK_MIN_LENGTH_M)

    table = pd_by_length(v, v.detected.to_numpy())
    per_km2 = lambda n: n / water_km2 if water_km2 > 0 else np.nan
    summary = dict(
        n_detections=int(len(dets)), n_vessels_in_scene=int(len(v_all)), n_vessels_on_water=int(len(v)),
        n_matched=int(len(pairs)), n_matched_on_water=int(v.detected.sum()),
        pd_all=float(v.detected.mean()) if len(v) else np.nan,
        n_unmatched=int(un.sum()), unmatched_per_km2=per_km2(int(un.sum())),
        n_fragment=int(dets.fragment.sum()), n_sidelobe=int(dets.sidelobe.sum()),
        n_ambiguity=int(dets.ambiguity.sum()), n_near_structure=int(dets.near_structure.sum()),
        n_unexplained=int(dets.unexplained.sum()),
        unexplained_per_km2=per_km2(int(dets.unexplained.sum())),
        n_dark_candidates=int(dets.dark_candidate.sum()), water_km2=float(water_km2), gate_m=gate_m,
    )
    if len(pairs):
        est = dets.loc[pairs.det_idx, length_est].to_numpy(float)
        true = v_all.loc[pairs.vessel_idx, "length_m"].to_numpy(float)
        ok = np.isfinite(est) & np.isfinite(true)
        if ok.sum() >= 3:
            e = est[ok] - true[ok]
            summary.update(length_n=int(ok.sum()), length_bias_m=float(np.median(e)),
                           length_mae_m=float(np.median(np.abs(e))),
                           length_rel_err_median=float(np.median(np.abs(e) / true[ok])),
                           length_spearman=float(stats.spearmanr(est[ok], true[ok])[0]))
        summary["match_dist_median_m"] = float(pairs.dist_m.median())
    return Validation(dets, v, pairs, table, water_km2, summary)


def truth_vessels(scene: Scene, water: np.ndarray | None = None, ais_only: bool = False) -> pd.DataFrame:
    """Injected targets as a vessel table at their image positions (synthetic scenes)."""
    rows = []
    w = scene.water if water is None else water
    for t in scene.truth:
        if ais_only and not t.ais:
            continue
        x, y = scene.grid.rowcol_to_xy(t.row, t.col)
        ri, ci = int(round(t.row)), int(round(t.col))
        rows.append(dict(mmsi=t.id, x=float(x), y=float(y), row=t.row, col=t.col, length_m=t.length_m,
                         kind=t.kind, ais=t.ais, scr_db=t.scr_db, sog_kn=t.sog_kn,
                         in_scene=True, on_water=bool(w[ri, ci])))
    return pd.DataFrame(rows)
