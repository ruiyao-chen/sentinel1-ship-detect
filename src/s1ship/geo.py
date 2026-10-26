"""Geometry: UTM projection, the scene pixel grid, and SAR viewing geometry.

Scenes are held on a north-up UTM grid with square pixels in metres (the
native grid of the Planetary Computer RTC product, and what the Earth Engine
backend requests), so distances and lengths are plain Euclidean. AIS positions
come as WGS84 lon/lat; ``lonlat_to_utm`` converts them with the Krueger series
(accurate to well under a metre within a zone), so the core package needs no
projection library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config

# WGS84 and UTM constants
_A = 6378137.0
_F = 1.0 / 298.257223563
_N = _F / (2.0 - _F)
_K0 = 0.9996
_E0 = 500000.0
_RECT = _A / (1.0 + _N) * (1.0 + _N ** 2 / 4.0 + _N ** 4 / 64.0)
_ALPHA = (_N / 2 - 2 * _N ** 2 / 3 + 5 * _N ** 3 / 16 + 41 * _N ** 4 / 180,
          13 * _N ** 2 / 48 - 3 * _N ** 3 / 5 + 557 * _N ** 4 / 1440,
          61 * _N ** 3 / 240 - 103 * _N ** 4 / 140,
          49561 * _N ** 4 / 161280)
_BETA = (_N / 2 - 2 * _N ** 2 / 3 + 37 * _N ** 3 / 96 - _N ** 4 / 360,
         _N ** 2 / 48 + _N ** 3 / 15 - 437 * _N ** 4 / 1440,
         17 * _N ** 3 / 480 - 37 * _N ** 4 / 840,
         4397 * _N ** 4 / 161280)
_DELTA = (2 * _N - 2 * _N ** 2 / 3 - 2 * _N ** 3 + 116 * _N ** 4 / 45,
          7 * _N ** 2 / 3 - 8 * _N ** 3 / 5 - 227 * _N ** 4 / 45,
          56 * _N ** 3 / 15 - 136 * _N ** 4 / 35,
          4279 * _N ** 4 / 630)


def utm_zone(lon: float) -> int:
    return int(np.floor((lon + 180.0) / 6.0)) % 60 + 1


def utm_epsg(lon: float, lat: float) -> int:
    return (32600 if lat >= 0 else 32700) + utm_zone(lon)


def _zone_of(epsg: int) -> tuple[int, bool]:
    if 32601 <= epsg <= 32660:
        return epsg - 32600, True
    if 32701 <= epsg <= 32760:
        return epsg - 32700, False
    raise ValueError(f"not a WGS84 UTM EPSG code: {epsg}")


def lonlat_to_utm(lon, lat, epsg: int):
    """WGS84 lon/lat (deg) -> UTM easting/northing (m) in the zone of ``epsg``."""
    zone, north = _zone_of(epsg)
    lam = np.radians(np.asarray(lon, dtype=float) - (6.0 * zone - 183.0))
    phi = np.radians(np.asarray(lat, dtype=float))
    c = 2.0 * np.sqrt(_N) / (1.0 + _N)
    t = np.sinh(np.arctanh(np.sin(phi)) - c * np.arctanh(c * np.sin(phi)))
    xi_p = np.arctan2(t, np.cos(lam))
    eta_p = np.arctanh(np.sin(lam) / np.sqrt(1.0 + t * t))
    xi, eta = xi_p.copy(), eta_p.copy()
    for j, a in enumerate(_ALPHA, start=1):
        xi += a * np.sin(2 * j * xi_p) * np.cosh(2 * j * eta_p)
        eta += a * np.cos(2 * j * xi_p) * np.sinh(2 * j * eta_p)
    x = _E0 + _K0 * _RECT * eta
    y = (0.0 if north else 10_000_000.0) + _K0 * _RECT * xi
    return x, y


def utm_to_lonlat(x, y, epsg: int):
    """UTM easting/northing (m) -> WGS84 lon/lat (deg)."""
    zone, north = _zone_of(epsg)
    xi = (np.asarray(y, dtype=float) - (0.0 if north else 10_000_000.0)) / (_K0 * _RECT)
    eta = (np.asarray(x, dtype=float) - _E0) / (_K0 * _RECT)
    xi_p, eta_p = xi.copy(), eta.copy()
    for j, b in enumerate(_BETA, start=1):
        xi_p -= b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        eta_p -= b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    chi = np.arcsin(np.sin(xi_p) / np.cosh(eta_p))
    phi = chi.copy()
    for j, d in enumerate(_DELTA, start=1):
        phi += d * np.sin(2 * j * chi)
    lam = np.arctan2(np.sinh(eta_p), np.cos(xi_p))
    return np.degrees(lam) + (6.0 * zone - 183.0), np.degrees(phi)


# --------------------------------------------------------------------------- #
# Scene grid
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Grid:
    """North-up UTM pixel grid. (x0, y0) is the outer top-left corner."""
    epsg: int
    x0: float
    y0: float
    pixel_m: float
    shape: tuple[int, int]

    def rowcol_to_xy(self, row, col):
        """Pixel indices (centre = integer index) -> UTM x, y."""
        return (self.x0 + (np.asarray(col, dtype=float) + 0.5) * self.pixel_m,
                self.y0 - (np.asarray(row, dtype=float) + 0.5) * self.pixel_m)

    def xy_to_rowcol(self, x, y):
        return ((self.y0 - np.asarray(y, dtype=float)) / self.pixel_m - 0.5,
                (np.asarray(x, dtype=float) - self.x0) / self.pixel_m - 0.5)

    def lonlat_to_rowcol(self, lon, lat):
        return self.xy_to_rowcol(*lonlat_to_utm(lon, lat, self.epsg))

    def rowcol_to_lonlat(self, row, col):
        return utm_to_lonlat(*self.rowcol_to_xy(row, col), self.epsg)

    def contains(self, row, col, margin: float = 0.0):
        row, col = np.asarray(row), np.asarray(col)
        H, W = self.shape
        return (row >= -0.5 + margin) & (row < H - 0.5 - margin) & (col >= -0.5 + margin) & (col < W - 0.5 - margin)

    @property
    def area_km2(self) -> float:
        return self.shape[0] * self.shape[1] * self.pixel_m ** 2 / 1e6

    def as_dict(self) -> dict:
        return dict(epsg=self.epsg, x0=self.x0, y0=self.y0, pixel_m=self.pixel_m, shape=list(self.shape))


def grid_for_aoi(aoi: config.Aoi, pixel_m: float = config.S1_PIXEL_M) -> Grid:
    """Smallest pixel-aligned UTM grid covering the AOI's lon/lat box."""
    epsg = utm_epsg(aoi.lon_c, aoi.lat_c)
    lons = np.array([aoi.lon_min, aoi.lon_max, aoi.lon_min, aoi.lon_max, aoi.lon_c, aoi.lon_c])
    lats = np.array([aoi.lat_min, aoi.lat_min, aoi.lat_max, aoi.lat_max, aoi.lat_min, aoi.lat_max])
    x, y = lonlat_to_utm(lons, lats, epsg)
    x0 = np.floor(x.min() / pixel_m) * pixel_m
    y0 = np.ceil(y.max() / pixel_m) * pixel_m
    W = int(np.ceil((x.max() - x0) / pixel_m))
    H = int(np.ceil((y0 - y.min()) / pixel_m))
    return Grid(epsg, float(x0), float(y0), float(pixel_m), (H, W))


# --------------------------------------------------------------------------- #
# SAR viewing geometry
# --------------------------------------------------------------------------- #
def slant_range_m(incidence_deg, altitude_m: float = config.S1_ALTITUDE_M,
                  earth_radius_m: float = config.EARTH_RADIUS_M):
    """Slant range for a spherical Earth, from the local incidence angle."""
    th = np.radians(np.asarray(incidence_deg, dtype=float))
    rs = earth_radius_m + altitude_m
    look = np.arcsin(earth_radius_m / rs * np.sin(th))
    gamma = th - look
    return np.sqrt(earth_radius_m ** 2 + rs ** 2 - 2 * earth_radius_m * rs * np.cos(gamma))


def azimuth_shift_m(sog_ms, cog_deg, look_azimuth_deg: float, incidence_deg,
                    velocity_ms: float = config.S1_VELOCITY_MS):
    """Along-track displacement of a moving target in a SAR image.

    A target with line-of-sight velocity v_r appears displaced along the
    platform track by -(R / V) v_r (v_r positive away from the radar): ships
    heading toward the radar shift forward along the satellite's direction of
    flight. ``look_azimuth_deg`` is the ground-range (look) direction,
    clockwise from north; for a right-looking SAR the flight direction is
    90 deg counter-clockwise from it. Returns (d_east, d_north) in metres.
    """
    cog = np.radians(np.asarray(cog_deg, dtype=float))
    look = np.radians(look_azimuth_deg)
    v_e, v_n = np.asarray(sog_ms) * np.sin(cog), np.asarray(sog_ms) * np.cos(cog)
    v_ground_range = v_e * np.sin(look) + v_n * np.cos(look)
    v_r = v_ground_range * np.sin(np.radians(incidence_deg))
    shift = -slant_range_m(incidence_deg) / velocity_ms * v_r
    flight = look - np.pi / 2.0
    return shift * np.sin(flight), shift * np.cos(flight)


def azimuth_ambiguity_m(incidence_deg, prf_hz: float, wavelength_m: float = config.S1_WAVELENGTH_M,
                        velocity_ms: float = config.S1_VELOCITY_MS):
    """Along-track offset of the first azimuth ambiguity, lambda R PRF / (2 V).

    A very bright ship leaks a faint copy of itself this far ahead of and
    behind it along the flight direction; an unmatched detection there is
    probably that ghost, not a dark vessel.
    """
    return wavelength_m * slant_range_m(incidence_deg) * prf_hz / (2.0 * velocity_ms)
