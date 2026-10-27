"""Data access: three backends behind one contract, ``load_scene(...) -> Scene``.

``backend="synthetic"`` builds a scene with injected ships, structures, a
coast and a breakwater, all of known position, size and SCR, so the whole
pipeline (CFAR, clustering, AIS matching, metrics) runs offline and is checked
against ground truth.

``backend="gee"`` pulls Sentinel-1 IW GRD from Google Earth Engine
(``COPERNICUS/S1_GRD_FLOAT``: the same ingest as ``COPERNICUS/S1_GRD``,
calibrated sigma0 in linear power), plus ESA WorldCover for the land mask.
Needs ``earthengine-api`` and a registered Earth Engine Cloud project.

``backend="pc"`` reads the same acquisition from Microsoft Planetary Computer:
the Sentinel-1 RTC product (gamma0, linear power, on the 10 m UTM grid) and
ESA WorldCover, both anonymously; the platform heading, per-line azimuth time,
incidence angle and PRFs come from the GRD product annotation XML. Needs
``rasterio``.

Both real backends return VV and VH on a north-up UTM grid aligned to 10 m.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from datetime import date as Date, datetime, timedelta, timezone

import numpy as np

from . import config, geo, synthetic
from .scene import Scene, TruthTarget

SCENE_CACHE = config.DATA / "scenes"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_scene(backend: str = "synthetic", aoi_key: str = config.DEFAULT_AOI,
               date: str = config.DEFAULT_DATE, orbit: int | None = config.DEFAULT_ORBIT,
               project: str | None = None, seed: int = config.SEED, refresh: bool = False) -> Scene:
    if backend == "synthetic":
        return demo_scene(seed=seed)
    if backend not in ("gee", "pc"):
        raise ValueError(f"unknown backend {backend!r}")
    cache = SCENE_CACHE / f"{backend}_{aoi_key}_{date}_o{orbit}.npz"
    if cache.exists() and not refresh:
        return Scene.load(cache)
    aoi = config.AOIS[aoi_key]
    scene = _gee_scene(aoi, date, orbit, project) if backend == "gee" else _pc_scene(aoi, date, orbit)
    cache.parent.mkdir(parents=True, exist_ok=True)
    scene.save(cache)
    return scene


# --------------------------------------------------------------------------- #
# Synthetic backend
# --------------------------------------------------------------------------- #
def demo_scene(seed: int = config.SEED, shape=(1536, 1536), n_ships: int = 70, nu: float = 6.0) -> Scene:
    """End-to-end synthetic test scene.

    Correlated K-distributed sea (nu = 6), a coast and breakwater, a fleet of
    70 ships 10-350 m long (about 10 % dark, 30 % moving 6-16 kn) and three
    bright fixed structures with no AIS. Every target is known.
    """
    rng = np.random.default_rng(seed)
    land = synthetic.coast_mask(shape, np.random.default_rng(seed))
    ships = synthetic.fleet(n_ships, shape, rng, water=~land, min_sep_px=45)
    used = np.array([[t.row, t.col] for t in ships])
    structures = []
    for k in range(3):
        for _ in range(1000):
            r, c = rng.uniform(80, shape[0] - 80), rng.uniform(80, 0.7 * shape[1])
            if np.min(np.hypot(used[:, 0] - r, used[:, 1] - c)) > 60:
                break
        structures.append(TruthTarget(id=900 + k, row=float(r), col=float(c), length_m=40.0, width_m=40.0,
                                      heading_deg=0.0, scr_db=28.0, kind="structure", ais=False))
        used = np.vstack([used, [r, c]])
    return synthetic.make_scene(shape, nu=nu, targets=ships + structures, land=True, correlated=True,
                                seed=seed, with_vh=True)


# --------------------------------------------------------------------------- #
# Sentinel-1 annotation (platform heading, azimuth time, incidence, PRF)
# --------------------------------------------------------------------------- #
def parse_annotation(xml: str) -> dict:
    """The fields of a Sentinel-1 product annotation XML that the detector needs."""
    one = lambda tag: re.search(f"<{tag}>([^<]*)</{tag}>", xml).group(1)
    pts = re.findall(r"<geolocationGridPoint>(.*?)</geolocationGridPoint>", xml, re.S)
    grab = lambda tag, s: re.search(f"<{tag}>([^<]*)</{tag}>", s).group(1)
    return dict(
        platform_heading_deg=float(one("platformHeading")),
        pass_=one("pass"),
        prf_hz=[float(v) for v in re.findall(r"<prf>([^<]*)</prf>", xml)],
        grid_lat=np.array([float(grab("latitude", p)) for p in pts]),
        grid_lon=np.array([float(grab("longitude", p)) for p in pts]),
        grid_incidence=np.array([float(grab("incidenceAngle", p)) for p in pts]),
        grid_time=np.array([np.datetime64(grab("azimuthTime", p)) for p in pts]),
    )


def annotation_at(ann: dict, lon: float, lat: float) -> dict:
    """Incidence angle and zero-Doppler time at a point (inverse-distance, 4 nearest grid points)."""
    d = np.hypot((ann["grid_lon"] - lon) * np.cos(np.radians(lat)), ann["grid_lat"] - lat)
    k = np.argsort(d)[:4]
    w = 1.0 / np.maximum(d[k], 1e-9)
    w /= w.sum()
    t0 = ann["grid_time"][k[0]]
    dt = (ann["grid_time"][k] - t0) / np.timedelta64(1, "us")
    t = t0 + np.timedelta64(int(round(float((w * dt).sum()))), "us")
    return dict(incidence_deg=float((w * ann["grid_incidence"][k]).sum()), time=t)


def subswath_prf(prfs: list[float], incidence_deg: float) -> float:
    """PRF of the IW sub-swath containing this incidence angle (nominal swath limits)."""
    edges = config.IW_SUBSWATH_INCIDENCE_EDGES
    i = int(np.searchsorted(edges, incidence_deg))
    i = min(max(i, 0), len(prfs) - 1)
    return float(prfs[i])


# --------------------------------------------------------------------------- #
# Planetary Computer backend
# --------------------------------------------------------------------------- #
def _http_get(url: str, body: dict | None = None, tries: int = 5) -> bytes:  # pragma: no cover - network
    """GET (or POST a JSON body), retrying transient network errors with backoff."""
    data = None if body is None else json.dumps(body).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except (OSError, urllib.error.URLError):
            if k == tries - 1:
                raise
            time.sleep(2.0 * 2 ** k)
    raise AssertionError("unreachable")


def _http_json(url: str, body: dict | None = None) -> dict:  # pragma: no cover - network
    return json.loads(_http_get(url, body))


def _pc_token(collection: str) -> str:
    return _http_json(config.PC_TOKEN.format(collection=collection))["token"]


def _pc_search(collection: str, aoi: config.Aoi, day: str | None = None, ids: list[str] | None = None) -> list[dict]:
    body: dict = {"collections": [collection], "limit": 100}
    if ids:
        body["ids"] = ids
    else:
        body["bbox"] = list(aoi.bounds())
    if day:
        d0 = Date.fromisoformat(day)
        body["datetime"] = f"{d0.isoformat()}T00:00:00Z/{(d0 + timedelta(days=1)).isoformat()}T00:00:00Z"
    return _http_json(f"{config.PC_STAC}/search", body)["features"]


def list_acquisitions(aoi_key: str = config.DEFAULT_AOI, start: str = "2021-10-01", end: str = "2021-10-31") -> list[dict]:
    """Sentinel-1 RTC acquisitions over the AOI between two dates (Planetary Computer STAC)."""
    aoi = config.AOIS[aoi_key]
    feats = _http_json(f"{config.PC_STAC}/search", {
        "collections": [config.PC_S1], "bbox": list(aoi.bounds()), "limit": 200,
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z"})["features"]
    return [dict(id=f["id"], datetime=f["properties"]["datetime"],
                 orbit=f["properties"].get("sat:relative_orbit"),
                 pass_=f["properties"].get("sat:orbit_state")) for f in feats]


def _read_aligned(href: str, grid: geo.Grid, resampling: str = "nearest") -> np.ndarray:
    """Read a raster onto ``grid``: a direct window if the grids align, else a warped VRT."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window

    H, W = grid.shape
    with rasterio.Env(GDAL_HTTP_MAX_RETRY=5, GDAL_HTTP_RETRY_DELAY=2), rasterio.open(href) as src:
        t = src.transform
        aligned = (src.crs is not None and src.crs.to_epsg() == grid.epsg and abs(t.a - grid.pixel_m) < 1e-6
                   and abs(t.e + grid.pixel_m) < 1e-6
                   and abs(((grid.x0 - t.c) / grid.pixel_m) % 1) < 1e-6
                   and abs(((t.f - grid.y0) / grid.pixel_m) % 1) < 1e-6)
        if aligned:
            col0 = int(round((grid.x0 - t.c) / grid.pixel_m))
            row0 = int(round((t.f - grid.y0) / grid.pixel_m))
            a = src.read(1, window=Window(col0, row0, W, H), boundless=True, fill_value=src.nodata or 0)
            nodata = src.nodata
        else:
            dst = from_origin(grid.x0, grid.y0, grid.pixel_m, grid.pixel_m)
            with WarpedVRT(src, crs=f"EPSG:{grid.epsg}", transform=dst, width=W, height=H,
                           resampling=getattr(Resampling, resampling)) as vrt:
                a = vrt.read(1)
                nodata = vrt.nodata
    a = a.astype(float)
    if nodata is not None:
        a[a == nodata] = np.nan
    return a


def _pc_bands(aoi: config.Aoi, day: str, orbit: int | None, grid: geo.Grid,
              bands=("vv", "vh")):  # pragma: no cover - network
    """Read Sentinel-1 RTC bands of one pass onto ``grid``; returns (arrays, STAC items)."""
    items = _pc_search(config.PC_S1, aoi, day)
    if orbit is not None:
        items = [f for f in items if f["properties"].get("sat:relative_orbit") == orbit]
    if not items:
        raise RuntimeError(f"no {config.PC_S1} items over {aoi.name} on {day} (orbit {orbit})")
    items.sort(key=lambda f: f["properties"]["datetime"])
    token = _pc_token(config.PC_S1)
    out = {b: np.full(grid.shape, np.nan) for b in bands}
    for f in items:                                  # consecutive slices of one pass: fill gaps
        for band, arr in out.items():
            if band in f["assets"]:
                a = _read_aligned(f["assets"][band]["href"] + "?" + token, grid)
                fill = np.isnan(arr) & np.isfinite(a)
                arr[fill] = a[fill]
    for arr in out.values():
        arr[arr <= 0] = np.nan
    return out, items


def _pc_scene(aoi: config.Aoi, day: str, orbit: int | None) -> Scene:  # pragma: no cover - network
    grid = geo.grid_for_aoi(aoi)
    b, items = _pc_bands(aoi, day, orbit, grid)
    vv, vh = b["vv"], b["vh"]
    water = _pc_water(aoi, grid)
    main = items[len(items) // 2]
    grd_id = main["id"].removesuffix("_rtc")
    grd = _pc_search("sentinel-1-grd", aoi, ids=[grd_id])[0]
    xml = _http_get(grd["assets"]["schema-product-vv"]["href"] + "?" + _pc_token("sentinel-1-grd")).decode()
    ann = parse_annotation(xml)
    at = annotation_at(ann, aoi.lon_c, aoi.lat_c)
    heading = ann["platform_heading_deg"] % 360.0
    return Scene(
        vv=vv, vh=vh, water=water, grid=grid, time_utc=str(at["time"]) + "Z", source="pc",
        scene_id=",".join(f["id"] for f in items), incidence_deg=at["incidence_deg"],
        look_azimuth_deg=(heading + 90.0) % 360.0, orbit_pass=ann["pass_"].upper(),
        meta=dict(platform_heading_deg=heading, prf_hz=subswath_prf(ann["prf_hz"], at["incidence_deg"]),
                  prfs_hz=ann["prf_hz"], incidence_plane=incidence_plane(ann, grid),
                  relative_orbit=main["properties"].get("sat:relative_orbit"),
                  product=config.PC_S1, quantity="gamma0 (RTC), linear power",
                  stac_ids=[f["id"] for f in items], annotation=grd_id, landcover=config.PC_LANDCOVER))


def incidence_plane(ann: dict, grid: geo.Grid, pad_deg: float = 0.5) -> list[float]:
    """Least-squares plane inc = a + b x_km + c y_km through the annotation grid near the scene.

    Coordinates are km from the grid's top-left corner; over a 25 km AOI the
    incidence angle is linear to well under 0.1 deg.
    """
    lon_c, lat_c = grid.rowcol_to_lonlat(grid.shape[0] / 2, grid.shape[1] / 2)
    near = (np.abs(ann["grid_lon"] - lon_c) < 1.5 + pad_deg) & (np.abs(ann["grid_lat"] - lat_c) < 1.0 + pad_deg)
    x, y = geo.lonlat_to_utm(ann["grid_lon"][near], ann["grid_lat"][near], grid.epsg)
    A = np.c_[np.ones(near.sum()), (x - grid.x0) / 1e3, (y - grid.y0) / 1e3]
    coef, *_ = np.linalg.lstsq(A, ann["grid_incidence"][near], rcond=None)
    return [float(c) for c in coef]


def structure_mask(aoi_key: str, orbit: int, dates: list[str], ratio: float = 3.0,
                   land_buffer_m: float = config.LAND_BUFFER_M) -> tuple[np.ndarray, dict]:  # pragma: no cover
    """Fixed bright structures (breakwaters, platforms, buoys, pilings) from other passes.

    For each reference date, VV is divided by its local sea mean (the CFAR
    ring); a pixel is a structure if that ratio exceeds ``ratio`` on at least
    half of the dates, i.e. its median over dates does. Ships come and go, a
    breakwater does not. For gamma speckle with L = 4.4 a single date exceeds
    3x its mean with probability 1.5e-3, so 5 of 9 dates by chance is ~1e-12.
    """
    from . import cfar
    aoi = config.AOIS[aoi_key]
    grid = geo.grid_for_aoi(aoi)
    water = _pc_water(aoi, grid)
    sea = _buffer_water(water, land_buffer_m, grid.pixel_m)
    count = np.zeros(grid.shape, dtype=np.int16)
    used = []
    for d in dates:
        try:
            vv = _pc_bands(aoi, d, orbit, grid, bands=("vv",))[0]["vv"]
        except RuntimeError:
            continue
        ok = sea & np.isfinite(vv)
        mean, _, _ = cfar.ring_stats(vv, ok, config.CFAR.guard_px, config.CFAR.background_px)
        with np.errstate(invalid="ignore", divide="ignore"):
            count += (np.isfinite(vv) & (vv > ratio * mean) & (mean > 0)).astype(np.int16)
        used.append(d)
    mask = count >= int(np.ceil(len(used) / 2))
    return mask, dict(dates=used, orbit=orbit, ratio=ratio, n_pixels=int(mask.sum()), grid=grid.as_dict())


def _buffer_water(water: np.ndarray, buffer_m: float, pixel_m: float) -> np.ndarray:
    from scipy import ndimage
    n = int(round(buffer_m / pixel_m))
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    return ~ndimage.binary_dilation(~water, structure=x * x + y * y <= n * n)


def _pc_water(aoi: config.Aoi, grid: geo.Grid) -> np.ndarray:  # pragma: no cover - network
    """Water from ESA WorldCover (class 80); outside any tile (open ocean) is water."""
    items = _pc_search(config.PC_LANDCOVER, aoi)
    items = [f for f in items if "v200" in f["id"] or f["properties"].get("esa_worldcover:product_version", "").startswith("2")] or items
    token = _pc_token(config.PC_LANDCOVER)
    cls = np.zeros(grid.shape)
    for f in items:
        a = _read_aligned(f["assets"]["map"]["href"] + "?" + token, grid)
        a = np.nan_to_num(a, nan=0.0)
        cls = np.where(cls == 0, a, cls)
    return (cls == config.WORLDCOVER_WATER) | (cls == 0)


# --------------------------------------------------------------------------- #
# Earth Engine backend
# --------------------------------------------------------------------------- #
def _require_ee():
    try:
        import ee  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "The 'gee' backend needs the Earth Engine API.\n"
            "    pip install earthengine-api\n"
            "    earthengine authenticate\n"
            "Then pass your registered Cloud project: --project your-ee-project."
        ) from exc
    return __import__("ee")


def _gee_scene(aoi: config.Aoi, day: str, orbit: int | None, project: str | None) -> Scene:  # pragma: no cover
    if not project:
        raise ValueError("the gee backend needs an Earth Engine Cloud project id (--project your-ee-project)")
    ee = _require_ee()
    ee.Initialize(project=project)
    region = ee.Geometry.Rectangle(list(aoi.bounds()), "EPSG:4326", False)
    d0 = Date.fromisoformat(day)
    coll = (ee.ImageCollection(config.EE_S1).filterBounds(region)
            .filterDate(d0.isoformat(), (d0 + timedelta(days=1)).isoformat())
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")))
    if orbit is not None:
        coll = coll.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
    props = coll.aggregate_array("system:index").getInfo()
    if not props:
        raise RuntimeError(f"no {config.EE_S1} images over {aoi.name} on {day} (orbit {orbit})")
    first = ee.Image(coll.first())
    meta = ee.Dictionary({
        "anx": first.get("orbitProperties_ascendingNodeTime"), "start_anx": first.get("startTimeANX"),
        "stop_anx": first.get("stopTimeANX"), "footprint": first.get("system:footprint"),
        "heading": first.get("platformHeading"), "pass": first.get("orbitProperties_pass"),
        "orbit": first.get("relativeOrbitNumber_start"),
        "pols": first.get("transmitterReceiverPolarisation")}).getInfo()
    pols = [p for p in ("VV", "VH") if p in meta["pols"]]

    water = (ee.ImageCollection(config.EE_LANDCOVER).first().select("Map")
             .eq(config.WORLDCOVER_WATER).unmask(1).rename("water"))
    stack = coll.mosaic().select(pols + ["angle"]).unmask(-1).addBands(water).toFloat()
    grid = geo.grid_for_aoi(aoi)
    bands = _ee_compute_pixels(ee, stack, grid, pols + ["angle", "water"])

    vv = np.where(bands["VV"] > 0, bands["VV"], np.nan)
    vh = np.where(bands["VH"] > 0, bands["VH"], np.nan) if "VH" in bands else None
    angle = np.where(bands["angle"] > 0, bands["angle"], np.nan)
    inc = float(np.nanmean(angle))
    heading = float(meta["heading"]) % 360.0
    look = (heading + 90.0) % 360.0          # Sentinel-1 looks right of track
    # The angle band is stored on a coarse grid, so fit a plane rather than differentiate it.
    rr, cc = np.mgrid[0:grid.shape[0]:8, 0:grid.shape[1]:8]
    ok = np.isfinite(angle[rr, cc])
    A = np.c_[np.ones(ok.sum()), cc[ok] * grid.pixel_m / 1e3, -rr[ok] * grid.pixel_m / 1e3]
    plane, *_ = np.linalg.lstsq(A, angle[rr, cc][ok], rcond=None)
    t = _gee_aoi_time(meta, grid, heading)
    return Scene(
        vv=vv, vh=vh, water=bands["water"] > 0.5, grid=grid, time_utc=t.isoformat() + "Z", source="gee",
        scene_id=",".join(props), incidence_deg=inc, look_azimuth_deg=look, orbit_pass=meta["pass"],
        meta=dict(platform_heading_deg=heading, relative_orbit=meta["orbit"],
                  incidence_plane=[float(c) for c in plane],
                  prf_hz=subswath_prf(config.IW_PRF_NOMINAL_HZ, inc), product=config.EE_S1,
                  quantity="sigma0, linear power",
                  time_note="slice start/stop times (ANX) interpolated along track to the AOI centre",
                  landcover=config.EE_LANDCOVER))


def _gee_aoi_time(meta: dict, grid: geo.Grid, heading_deg: float) -> datetime:
    """Azimuth time at the AOI centre, interpolated between the slice start and stop.

    Earth Engine keeps no annotation, only the slice's start and stop times
    after the ascending node and its footprint. The footprint's extent along
    the platform heading spans that interval; the AOI centre's position along
    the same axis gives its time to about a second.
    """
    fp = np.asarray(meta["footprint"]["coordinates"], dtype=float)
    fx, fy = geo.lonlat_to_utm(fp[:, 0], fp[:, 1], grid.epsg)
    cx, cy = geo.lonlat_to_utm(*grid.rowcol_to_lonlat(grid.shape[0] / 2, grid.shape[1] / 2), grid.epsg)
    h = np.radians(heading_deg)
    s = (fx - cx) * np.sin(h) + (fy - cy) * np.cos(h)      # along-track metres, AOI centre at 0
    frac = float(np.clip(-s.min() / (s.max() - s.min()), 0.0, 1.0))
    t0 = meta["anx"] + meta["start_anx"]
    ms = t0 + frac * (meta["stop_anx"] - meta["start_anx"])
    return datetime.fromtimestamp(ms / 1000.0, timezone.utc).replace(tzinfo=None)


def _ee_compute_pixels(ee, image, grid: geo.Grid, names: list[str], tile: int = 1024) -> dict:  # pragma: no cover
    """Fetch ``image`` on ``grid`` in tiles with ee.data.computePixels (NumPy output)."""
    H, W = grid.shape
    out = {n: np.zeros((H, W), dtype=np.float32) for n in names}
    for r0 in range(0, H, tile):
        for c0 in range(0, W, tile):
            h, w = min(tile, H - r0), min(tile, W - c0)
            req = {
                "expression": image,
                "fileFormat": "NUMPY_NDARRAY",
                "grid": {
                    "dimensions": {"width": w, "height": h},
                    "affineTransform": {"scaleX": grid.pixel_m, "shearX": 0,
                                        "translateX": grid.x0 + c0 * grid.pixel_m,
                                        "shearY": 0, "scaleY": -grid.pixel_m,
                                        "translateY": grid.y0 - r0 * grid.pixel_m},
                    "crsCode": f"EPSG:{grid.epsg}",
                },
            }
            arr = ee.data.computePixels(req)
            for n in names:
                out[n][r0:r0 + h, c0:c0 + w] = arr[n]
    return out


def load_structure_mask(aoi_key: str = config.DEFAULT_AOI, orbit: int = config.DEFAULT_ORBIT,
                        grid: geo.Grid | None = None) -> np.ndarray | None:
    """The cached fixed-structure mask (see ``structure_mask``), or None if not fetched."""
    path = config.DATA / f"structures_{aoi_key}_o{orbit}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    shape = tuple(int(v) for v in z["shape"])
    mask = np.unpackbits(z["mask"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
    if grid is not None and grid.shape != shape:
        raise ValueError(f"structure mask shape {shape} does not match the scene grid {grid.shape}")
    return mask
