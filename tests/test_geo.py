"""UTM projection, scene grid, and SAR viewing geometry."""

from __future__ import annotations

import numpy as np
import pytest

from s1ship import config, geo

# Reference values from pyproj 3.8 (PROJ), EPSG:4326 -> UTM, always_xy.
PYPROJ = [
    (-118.18, 33.70, 32611, 390644.9820, 3729518.2345),
    (-120.90, 33.00, 32611, 135562.9635, 3658048.5676),   # 3 deg off the central meridian
    (-74.00, 40.50, 32618, 584734.5389, 4483734.9916),
    (151.20, -33.90, 32756, 333568.9410, 6247473.3368),   # southern hemisphere
]


@pytest.mark.parametrize("lon,lat,epsg,x,y", PYPROJ)
def test_utm_matches_proj(lon, lat, epsg, x, y):
    gx, gy = geo.lonlat_to_utm(lon, lat, epsg)
    assert gx == pytest.approx(x, abs=1e-3) and gy == pytest.approx(y, abs=1e-3)
    lo, la = geo.utm_to_lonlat(gx, gy, epsg)
    assert lo == pytest.approx(lon, abs=1e-9) and la == pytest.approx(lat, abs=1e-9)


def test_utm_zone_and_epsg():
    assert geo.utm_zone(-118.2) == 11 and geo.utm_epsg(-118.2, 33.7) == 32611
    assert geo.utm_epsg(-74.0, 40.5) == 32618 and geo.utm_epsg(151.2, -33.9) == 32756


def test_grid_round_trips_and_covers_aoi():
    aoi = config.AOIS["lalb"]
    g = geo.grid_for_aoi(aoi)
    assert g.x0 % 10 == 0 and g.y0 % 10 == 0
    r, c = g.lonlat_to_rowcol(np.array([aoi.lon_min, aoi.lon_max]), np.array([aoi.lat_max, aoi.lat_min]))
    assert g.contains(r, c).all()
    lon, lat = g.rowcol_to_lonlat(r, c)
    assert lon == pytest.approx([aoi.lon_min, aoi.lon_max], abs=1e-9)
    x, y = g.rowcol_to_xy(12.0, 34.0)
    assert g.xy_to_rowcol(x, y) == pytest.approx((12.0, 34.0))


def test_slant_range_is_geometric():
    r = geo.slant_range_m(np.array([30.0, 39.0, 46.0]))
    assert np.all(np.diff(r) > 0)
    flat = config.S1_ALTITUDE_M / np.cos(np.radians([30.0, 39.0, 46.0]))
    assert np.all(r < flat) and np.all(r > 0.9 * flat)     # Earth curvature shortens it a little


def test_azimuth_shift_direction_and_size():
    look, inc = 283.0, 39.0
    flight = look - 90.0
    # moving along the flight direction: no line-of-sight velocity, no shift
    de, dn = geo.azimuth_shift_m(5.0, flight, look, inc)
    assert np.hypot(de, dn) == pytest.approx(0.0, abs=1e-9)
    # moving toward the radar: shifted forward along the flight direction by R/V * v sin(inc)
    de, dn = geo.azimuth_shift_m(5.0, look - 180.0, look, inc)
    expected = geo.slant_range_m(inc) / config.S1_VELOCITY_MS * 5.0 * np.sin(np.radians(inc))
    assert np.hypot(de, dn) == pytest.approx(expected)
    bearing = np.degrees(np.arctan2(de, dn)) % 360
    assert bearing == pytest.approx(flight % 360, abs=1e-6)


def test_azimuth_ambiguity_offset_is_a_few_km():
    off = geo.azimuth_ambiguity_m(39.0, 1451.63)
    assert 4000 < off < 5200
