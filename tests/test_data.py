"""Scene contract, persistence, annotation parsing, and backend selection (offline)."""

from __future__ import annotations

import numpy as np
import pytest

from s1ship import data, synthetic as S
from s1ship.scene import Scene, TruthTarget

ANNOTATION = """<product><adsHeader/><generalAnnotation><productInformation>
<pass>Descending</pass><platformHeading>-1.670977994095180e+02</platformHeading>
</productInformation><downlinkInformationList>
<downlinkInformation><prf>1.717128973878037e+03</prf></downlinkInformation>
<downlinkInformation><prf>1.451627112193990e+03</prf></downlinkInformation>
<downlinkInformation><prf>1.685817302492702e+03</prf></downlinkInformation>
</downlinkInformationList></generalAnnotation><geolocationGrid><geolocationGridPointList>
<geolocationGridPoint><azimuthTime>2021-10-21T13:52:50.000000</azimuthTime><line>0</line><pixel>0</pixel>
<latitude>33.6</latitude><longitude>-118.3</longitude><height>0</height><incidenceAngle>38.0</incidenceAngle></geolocationGridPoint>
<geolocationGridPoint><azimuthTime>2021-10-21T13:52:50.000000</azimuthTime><line>0</line><pixel>1</pixel>
<latitude>33.6</latitude><longitude>-118.0</longitude><height>0</height><incidenceAngle>40.0</incidenceAngle></geolocationGridPoint>
<geolocationGridPoint><azimuthTime>2021-10-21T13:52:54.000000</azimuthTime><line>1</line><pixel>0</pixel>
<latitude>33.8</latitude><longitude>-118.3</longitude><height>0</height><incidenceAngle>38.0</incidenceAngle></geolocationGridPoint>
<geolocationGridPoint><azimuthTime>2021-10-21T13:52:54.000000</azimuthTime><line>1</line><pixel>1</pixel>
<latitude>33.8</latitude><longitude>-118.0</longitude><height>0</height><incidenceAngle>40.0</incidenceAngle></geolocationGridPoint>
</geolocationGridPointList></geolocationGrid></product>"""


def test_parse_annotation_and_interpolate():
    ann = data.parse_annotation(ANNOTATION)
    assert ann["platform_heading_deg"] == pytest.approx(-167.0977994)
    assert ann["pass_"] == "Descending" and len(ann["prf_hz"]) == 3
    at = data.annotation_at(ann, -118.15, 33.7)          # the centre of the four points
    assert at["incidence_deg"] == pytest.approx(39.0, abs=1e-6)
    assert at["time"] == np.datetime64("2021-10-21T13:52:52.000000")


def test_subswath_prf():
    prfs = [1717.1, 1451.6, 1685.8]
    assert data.subswath_prf(prfs, 33.0) == 1717.1
    assert data.subswath_prf(prfs, 39.0) == 1451.6
    assert data.subswath_prf(prfs, 44.0) == 1685.8


def test_scene_save_load_round_trip(tmp_path):
    tg = [TruthTarget(0, 20.0, 30.0, 100.0, 15.0, 45.0, 18.0, sog_kn=8.0, cog_deg=45.0)]
    sc = S.make_scene((64, 80), targets=tg, land=True, seed=2, with_vh=True)
    p = tmp_path / "s.npz"
    sc.save(p)
    back = Scene.load(p)
    assert np.allclose(back.vv, sc.vv.astype(np.float32)) and np.array_equal(back.water, sc.water)
    assert back.vh is not None and back.grid == sc.grid
    assert back.truth == sc.truth and back.time_utc == sc.time_utc
    assert back.look_azimuth_deg == sc.look_azimuth_deg


def test_land_buffer_grows_land():
    sc = S.make_scene((200, 200), land=True, seed=4)
    w0, w1 = sc.water_buffered(0.0), sc.water_buffered(100.0)
    assert np.array_equal(w0, sc.water)
    assert w1.sum() < w0.sum() and not (w1 & ~w0).any()


def test_synthetic_backend_and_backend_errors():
    sc = data.load_scene("synthetic")
    assert sc.vh is not None and len(sc.truth) == 73 and sc.source == "synthetic"
    with pytest.raises(ValueError):
        data.load_scene("sentinelhub")
    with pytest.raises(ValueError, match="project"):
        data._gee_scene(None, "2021-10-21", 71, project=None)


def test_incidence_plane_recovers_a_linear_field():
    from s1ship import geo
    grid = S.default_grid((300, 300))
    lon, lat = grid.rowcol_to_lonlat(np.array([0, 0, 299, 299, 150]), np.array([0, 299, 0, 299, 150]))
    x, y = geo.lonlat_to_utm(lon, lat, grid.epsg)
    inc = 40.0 + 0.2 * (x - grid.x0) / 1e3 - 0.05 * (y - grid.y0) / 1e3
    ann = dict(grid_lon=lon, grid_lat=lat, grid_incidence=inc)
    coef = data.incidence_plane(ann, grid)
    assert coef == pytest.approx([40.0, 0.2, -0.05], abs=1e-6)
    sc = S.make_scene((300, 300), seed=0)
    sc.meta["incidence_plane"] = coef
    assert sc.incidence_at(x[4], y[4]) == pytest.approx(inc[4], abs=1e-6)
