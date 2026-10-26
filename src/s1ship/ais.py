"""AIS from MarineCadastre.gov: one day, one bounding box, positions at SAR time.

MarineCadastre (BOEM / NOAA) publishes U.S. coastal AIS as one zipped CSV per
day for the whole country (~350 MB zipped for 2021). ``fetch_day`` streams that
file to disk, keeps only the reports inside the AOI (plus a small margin) and
deletes the raw zip unless asked to keep it; the kept subset is a few MB.

Each vessel's position at the SAR acquisition time is interpolated between its
reports either side (at most 10 min apart) or, failing that, dead-reckoned from
a report within 30 min. ``project_to_scene`` then moves it to where a moving
ship *appears* in the SAR image (the azimuth shift from its line-of-sight
velocity) so that detections and AIS are compared like for like.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, geo
from .scene import Scene

KEEP = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading", "VesselName",
        "VesselType", "Status", "Length", "Width", "TransceiverClass"]
KN = 0.514444  # m/s per knot


def day_url(day: str) -> str:
    d = Date.fromisoformat(day)
    return config.AIS_URL.format(year=d.year, month=d.month, day=d.day)


# --------------------------------------------------------------------------- #
# Download and filter
# --------------------------------------------------------------------------- #
def download(url: str, dest: Path, chunk: int = 1 << 20) -> dict:  # pragma: no cover - network
    """Stream ``url`` to ``dest``; return its size and SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    n = 0
    with urllib.request.urlopen(url, timeout=300) as r, open(part, "wb") as f:
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)
            h.update(b)
            n += len(b)
    part.rename(dest)
    return dict(url=url, file=dest.name, bytes=n, sha256=h.hexdigest())


def filter_csv(stream, bbox: tuple[float, float, float, float], chunksize: int = 500_000) -> pd.DataFrame:
    """Rows of a MarineCadastre CSV stream inside ``bbox`` = (lon_min, lat_min, lon_max, lat_max)."""
    lon0, lat0, lon1, lat1 = bbox
    parts = []
    for ch in pd.read_csv(stream, chunksize=chunksize, usecols=lambda c: c in KEEP, low_memory=False):
        m = ch.LON.between(lon0, lon1) & ch.LAT.between(lat0, lat1)
        if m.any():
            parts.append(ch[m])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEEP)


def filter_zip(zip_path: Path, bbox) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as f:
            return filter_csv(io.TextIOWrapper(f, encoding="utf-8"), bbox)


def aoi_bbox(aoi: config.Aoi, margin_deg: float = config.AIS_BBOX_MARGIN_DEG):
    return (aoi.lon_min - margin_deg, aoi.lat_min - margin_deg, aoi.lon_max + margin_deg, aoi.lat_max + margin_deg)


def fetch_day(day: str, aoi: config.Aoi, out_csv: Path, raw_dir: Path = config.DATA_RAW,
              keep_raw: bool = False) -> dict:  # pragma: no cover - network
    """Download one AIS day, keep the AOI's reports in ``out_csv`` (gzip CSV). Returns a manifest."""
    url = day_url(day)
    raw = raw_dir / Path(url).name
    man = download(url, raw) if not raw.exists() else dict(url=url, file=raw.name, bytes=raw.stat().st_size,
                                                           sha256=_sha256(raw))
    df = filter_zip(raw, aoi_bbox(aoi))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, compression="gzip")
    man.update(bbox=list(aoi_bbox(aoi)), rows_kept=int(len(df)), vessels_kept=int(df.MMSI.nunique()),
               subset=out_csv.name, subset_sha256=_sha256(out_csv), raw_deleted=not keep_raw)
    if not keep_raw:
        raw.unlink()
    return man


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Cleaning and positions at a time
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Parse times, drop non-ship MMSIs, and turn AIS 'not available' codes into NaN."""
    df = df.copy()
    df["BaseDateTime"] = pd.to_datetime(df.BaseDateTime)
    if df.BaseDateTime.dt.tz is not None:
        df["BaseDateTime"] = df.BaseDateTime.dt.tz_convert(None)
    df = df[(df.MMSI >= 200_000_000) & (df.MMSI < 800_000_000)]     # ship stations (MID 2xx-7xx)
    df = df[df.LAT.between(-90, 90) & df.LON.between(-180, 180)]
    df.loc[df.SOG >= 102.2, "SOG"] = np.nan
    df.loc[df.COG >= 360.0, "COG"] = np.nan
    if "Heading" in df:
        df.loc[df.Heading >= 360, "Heading"] = np.nan
    for c in ("Length", "Width"):
        if c in df:
            df.loc[~(df[c] > 0), c] = np.nan
    return df.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)


def positions_at(df: pd.DataFrame, t, max_gap_s: float = config.AIS_INTERP_MAX_GAP_S,
                 nearest_max_s: float = config.AIS_NEAREST_MAX_S) -> pd.DataFrame:
    """Each vessel's position at time ``t`` (one row per MMSI).

    method = "interp" when reports bracket ``t`` at most ``max_gap_s`` apart,
    "nearest" when the closest report is within ``nearest_max_s`` (moved along
    its SOG/COG), otherwise the vessel is dropped.
    """
    t = pd.Timestamp(t)
    if t.tzinfo is not None:
        t = t.tz_convert(None)
    df = clean(df)
    df = df.assign(dt=(df.BaseDateTime - t).dt.total_seconds())
    static = df.groupby("MMSI").agg(length_m=("Length", "max"), width_m=("Width", "max"),
                                    vessel_type=("VesselType", "max"), name=("VesselName", "first"),
                                    n_reports=("dt", "size"))
    cols = ["dt", "LAT", "LON", "SOG", "COG", "Heading", "Status"]
    b = df[df.dt <= 0].groupby("MMSI")[cols].last()
    a = df[df.dt > 0].groupby("MMSI")[cols].first()
    j = b.merge(a, left_index=True, right_index=True, how="outer", suffixes=("_b", "_a"))
    j.index.name = "MMSI"

    gap = j.dt_a - j.dt_b
    interp = gap.notna() & (gap <= max_gap_s)
    nearer_b = j.dt_a.isna() | (j.dt_b.notna() & (-j.dt_b <= j.dt_a))
    pick = lambda c: np.where(nearer_b, j[f"{c}_b"], j[f"{c}_a"])
    dt_n = pick("dt")
    out = pd.DataFrame(index=j.index)
    out["sog_kn"], out["cog_deg"], out["heading_deg"], out["status"] = pick("SOG"), pick("COG"), pick("Heading"), pick("Status")
    out["dt_s"] = dt_n

    # interpolated positions
    f = np.where(interp, -j.dt_b / gap.where(gap > 0, 1.0), np.nan)
    lat_i = j.LAT_b + f * (j.LAT_a - j.LAT_b)
    lon_i = j.LON_b + f * (j.LON_a - j.LON_b)
    # dead-reckoned from the nearest report (stationary if SOG unknown or < 0.5 kn)
    lat_n, lon_n = pick("LAT"), pick("LON")
    sog = np.nan_to_num(out.sog_kn.to_numpy(float), nan=0.0)
    sog = np.where(sog < 0.5, 0.0, sog) * KN
    cog = np.radians(np.nan_to_num(out.cog_deg.to_numpy(float), nan=0.0))
    move = -dt_n.astype(float)
    lat_d = lat_n.astype(float) + sog * np.cos(cog) * move / 111_320.0
    lon_d = lon_n.astype(float) + sog * np.sin(cog) * move / (111_320.0 * np.cos(np.radians(lat_n.astype(float))))
    near_ok = np.abs(dt_n.astype(float)) <= nearest_max_s

    out["lat"] = np.where(interp, lat_i, lat_d)
    out["lon"] = np.where(interp, lon_i, lon_d)
    out["method"] = np.where(interp, "interp", np.where(near_ok, "nearest", "none"))
    out = out[out.method != "none"].join(static)
    return out.reset_index().rename(columns={"MMSI": "mmsi"})


def project_to_scene(pos: pd.DataFrame, scene: Scene, water: np.ndarray | None = None,
                     shift: bool = True) -> pd.DataFrame:
    """Add true and SAR-apparent positions (UTM, pixel) and scene membership flags.

    The apparent position is the true one plus the azimuth shift caused by the
    vessel's line-of-sight velocity (``geo.azimuth_shift_m``). ``in_scene`` means
    the apparent position has valid SAR data; ``on_water`` that it is on
    ``water`` (pass the buffered mask used for detection).
    """
    pos = pos.copy()
    x, y = geo.lonlat_to_utm(pos.lon.to_numpy(), pos.lat.to_numpy(), scene.grid.epsg)
    pos["x_true"], pos["y_true"] = x, y
    if shift and scene.look_azimuth_deg is not None:
        sog = np.nan_to_num(pos.sog_kn.to_numpy(float), nan=0.0) * KN
        cog = np.nan_to_num(pos.cog_deg.to_numpy(float), nan=0.0)
        de, dn = geo.azimuth_shift_m(sog, cog, scene.look_azimuth_deg, scene.incidence_at(x, y))
    else:
        de = dn = np.zeros(len(pos))
    pos["x"], pos["y"] = x + de, y + dn
    pos["shift_m"] = np.hypot(de, dn)
    r, c = scene.grid.xy_to_rowcol(pos.x.to_numpy(), pos.y.to_numpy())
    pos["row"], pos["col"] = r, c
    inside = scene.grid.contains(r, c)
    ri = np.clip(np.round(r).astype(int), 0, scene.grid.shape[0] - 1)
    ci = np.clip(np.round(c).astype(int), 0, scene.grid.shape[1] - 1)
    pos["in_scene"] = inside & np.isfinite(scene.vv[ri, ci])
    w = scene.water if water is None else water
    pos["on_water"] = pos.in_scene & w[ri, ci]
    return pos
