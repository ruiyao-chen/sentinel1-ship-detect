"""Configuration: areas of interest, sensor constants, detector defaults, sources.

Every number that changes a detection is defined here with where it comes from.
Three kinds appear:

* **Published.** Sentinel-1 product and orbit constants from ESA documentation
  or the product metadata (e.g. the IW GRDH equivalent number of looks, 4.4).
* **Estimated from the scene.** Clutter parameters (K shape, measured ENL) are
  fitted to each scene's water pixels by ``clutter``; nothing here fixes them.
* **Assumed.** Detector design choices (false-alarm rate, window sizes, land
  buffer, AIS matching gate). These are labelled ``ASSUMED`` and are what the
  experiments vary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SEED = 0


# --------------------------------------------------------------------------- #
# Areas of interest
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Aoi:
    name: str
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float
    note: str = ""

    @property
    def lon_c(self) -> float:
        return 0.5 * (self.lon_min + self.lon_max)

    @property
    def lat_c(self) -> float:
        return 0.5 * (self.lat_min + self.lat_max)

    def bounds(self) -> tuple[float, float, float, float]:
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)


AOIS: dict[str, Aoi] = {
    # San Pedro Bay: the Los Angeles / Long Beach breakwater, the anchorages
    # inside and south of it, and open water out to ~20 km. In October 2021 the
    # port congestion put dozens of container ships at anchor here.
    "lalb": Aoi("Los Angeles / Long Beach, San Pedro Bay",
                -118.30, 33.56, -118.02, 33.77,
                "LA/LB anchorages; breakwater, port terminals and oil islands are masked as land"),
    "nynj": Aoi("New York / New Jersey Harbor approaches",
                -74.10, 40.42, -73.85, 40.60,
                "Lower Bay and Ambrose / Sandy Hook channels"),
}
DEFAULT_AOI = "lalb"

# Default real-data case: Sentinel-1A, relative orbit 71 (descending), the
# 2021-10-21 13:52 UTC pass over San Pedro Bay. Listed by ASF and Planetary
# Computer; the AIS day file for 2021-10-21 is on MarineCadastre.
DEFAULT_DATE = "2021-10-21"
DEFAULT_ORBIT = 71

# Reference passes (same relative orbit) for the fixed-structure mask, chosen
# outside the 2021-22 port congestion so that anchored ships are rarely in the
# same place on most of them; breakwaters, platforms and pilings always are.
STRUCTURE_REF_DATES = ("2020-01-12", "2020-03-12", "2020-05-11", "2020-07-10", "2020-09-08",
                       "2023-01-20", "2023-04-14", "2023-07-07", "2023-10-11")
STRUCTURE_RATIO = 3.0             # ASSUMED: > 3x the local sea mean on most dates
STRUCTURE_DILATE_PX = 2


# --------------------------------------------------------------------------- #
# Sentinel-1 IW GRDH (published)
# --------------------------------------------------------------------------- #
S1_WAVELENGTH_M = 0.05546        # C band, 5.405 GHz
S1_ENL_IW_GRDH = 4.4             # ESA product definition; also STAC sar:looks_equivalent_number
S1_PIXEL_M = 10.0                # GRDH pixel spacing (range x azimuth)
S1_RESOLUTION_M = (20.0, 22.0)   # IW GRDH resolution, range x azimuth
S1_PSF_SIGMA_M = 21.0 / 2.355    # Gaussian with the mean 21 m resolution as FWHM
S1_ALTITUDE_M = 693e3            # nominal orbit altitude
S1_VELOCITY_MS = 7590.0          # orbital speed; azimuth shift uses R / V
EARTH_RADIUS_M = 6371e3
# IW sub-swath PRFs (IW1, IW2, IW3) as read from the 2021-10-21 product
# annotation; fixed by the instrument configuration. Used to place azimuth
# ambiguities when a backend gives no annotation (Earth Engine).
IW_PRF_NOMINAL_HZ = (1717.13, 1451.63, 1685.82)
# Approximate incidence angles (deg) at the IW1/IW2 and IW2/IW3 boundaries.
IW_SUBSWATH_INCIDENCE_EDGES = (36.5, 41.8)

# Earth Engine: S1_GRD_FLOAT is the same ingest as COPERNICUS/S1_GRD (thermal
# noise removed, calibrated, terrain corrected) but in linear power, not dB.
EE_S1 = "COPERNICUS/S1_GRD_FLOAT"
EE_S1_DB = "COPERNICUS/S1_GRD"
EE_LANDCOVER = "ESA/WorldCover/v200"
# Planetary Computer: Sentinel-1 RTC (gamma0, linear power, 10 m UTM COGs) and
# ESA WorldCover; both read anonymously with a short-lived SAS token.
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"
PC_S1 = "sentinel-1-rtc"
PC_LANDCOVER = "esa-worldcover"
WORLDCOVER_WATER = 80            # ESA WorldCover class "permanent water bodies"


# --------------------------------------------------------------------------- #
# Detector design (ASSUMED; varied in the experiments)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CfarParams:
    pfa: float = 1e-6               # design per-pixel false-alarm probability
    guard_px: int = 12              # guard window half-width (25 x 25 px = 250 m)
    background_px: int = 40         # background window half-width (81 x 81 px = 810 m)
    min_background_frac: float = 0.25  # need this share of the ring valid, else no decision
    clutter: str = "k"              # "gamma" or "k"
    censor: bool = True             # second pass with detections removed from the ring
    censor_dilate_px: int = 3


CFAR = CfarParams()

LAND_BUFFER_M = 100.0            # ASSUMED: grow land by this before detecting
MERGE_RADIUS_PX = 2              # ASSUMED: detections closer than this join one target
MIN_TARGET_PIXELS = 2            # ASSUMED: S1 resolution spans >= 2 pixels
MAX_TARGET_LENGTH_M = 500.0      # ASSUMED: longer clusters are not ships (land leaks, wakes)
CORE_DB = 15.0                   # ASSUMED: shape from pixels within 15 dB of the peak (sidelobes ~ -21 dB)
VH_CONFIRM_PFA = 1e-4            # ASSUMED: VH CFAR rate used to confirm VV targets
VH_CONFIRM_DILATE_PX = 2


# --------------------------------------------------------------------------- #
# AIS validation (ASSUMED unless noted)
# --------------------------------------------------------------------------- #
AIS_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.zip"
AIS_BBOX_MARGIN_DEG = 0.03        # keep AIS a little outside the AOI (moving ships)
AIS_INTERP_MAX_GAP_S = 600        # interpolate between reports at most 10 min apart
AIS_NEAREST_MAX_S = 1800          # else use a report within 30 min, dead-reckoned
MATCH_GATE_M = 200.0              # plus half the AIS length (antenna offset)
LENGTH_BINS_M = (0, 25, 50, 100, 200, 400)
DARK_MIN_LENGTH_M = 20.0          # unmatched detections shorter than this are not reported
NEAR_STRUCTURE_M = 300.0          # unmatched detections this close to a fixed structure are not "dark"
