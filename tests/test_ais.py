"""AIS: bounding-box filtering, cleaning, positions at SAR time, apparent positions."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from s1ship import ais, synthetic as S

HEADER = "MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,VesselName,IMO,CallSign,VesselType,Status,Length,Width,Draft,Cargo,TransceiverClass\n"
ROWS = [
    "367000001,2021-10-21T13:50:00,33.70,-118.20,0.0,360.0,511,ANCHORED,,,70,1,300,48,,,A",
    "367000001,2021-10-21T13:56:00,33.70,-118.20,0.1,12.0,511,ANCHORED,,,70,1,300,48,,,A",
    "367000002,2021-10-21T13:52:00,33.65,-118.10,10.0,0.0,0,MOVER,,,60,0,,,,,A",
    "367000003,2021-10-21T13:52:00,40.00,-74.00,5.0,90.0,90,ELSEWHERE,,,70,0,100,20,,,A",
    "003660001,2021-10-21T13:52:00,33.71,-118.21,0.0,0.0,0,BASESTATION,,,,,,,,,",
    "367000004,2021-10-21T12:00:00,33.66,-118.15,0.0,0.0,0,STALE,,,70,1,80,12,,,A",
]
CSV = HEADER + "\n".join(ROWS) + "\n"
BBOX = (-118.30, 33.56, -118.02, 33.77)


def test_day_url():
    assert ais.day_url("2021-10-21").endswith("/2021/AIS_2021_10_21.zip")


def test_filter_csv_keeps_only_the_box():
    df = ais.filter_csv(io.StringIO(CSV), BBOX, chunksize=2)
    assert set(df.MMSI) == {367000001, 367000002, 3660001, 367000004}


def test_filter_zip(tmp_path):
    z = tmp_path / "AIS_2021_10_21.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        f.writestr("AIS_2021_10_21.csv", CSV)
    assert len(ais.filter_zip(z, BBOX)) == 5


def test_clean_drops_non_ships_and_not_available_codes():
    df = ais.clean(pd.read_csv(io.StringIO(CSV)))
    assert 3660001 not in set(df.MMSI)                       # base station (MID-less MMSI)
    a = df[df.MMSI == 367000001].iloc[0]
    assert np.isnan(a.COG) and np.isnan(a.Heading)            # 360 and 511 = not available
    assert np.isnan(df[df.MMSI == 367000002].Length.iloc[0])


def test_positions_at_interpolates_dead_reckons_and_drops_stale():
    t = pd.Timestamp("2021-10-21T13:53:00")
    pos = ais.positions_at(pd.read_csv(io.StringIO(CSV)), t).set_index("mmsi")
    assert pos.loc[367000001, "method"] == "interp"
    assert pos.loc[367000001, "lat"] == pytest.approx(33.70)
    assert pos.loc[367000001, "length_m"] == 300
    mv = pos.loc[367000002]                                    # 10 kn due north, report 60 s earlier
    assert mv.method == "nearest"
    assert (mv.lat - 33.65) * 111_320.0 == pytest.approx(10 * ais.KN * 60, rel=1e-6)
    assert mv.lon == pytest.approx(-118.10)
    assert 367000004 not in pos.index                          # last report 53 min earlier


def test_apparent_positions_reproduce_injected_image_positions():
    """synthetic_ais places each ship at its image position minus its azimuth
    shift; project_to_scene must put it back."""
    rng = np.random.default_rng(0)
    tg = S.fleet(20, (512, 512), rng, moving_frac=0.6, dark_frac=0.0, min_sep_px=40)
    sc = S.make_scene((512, 512), targets=tg, seed=1)
    pos = ais.positions_at(S.synthetic_ais(sc), sc.time_utc)
    pj = ais.project_to_scene(pos, sc).set_index("mmsi")
    moving = 0
    for t in tg:
        p = pj.loc[366000000 + t.id]
        assert p.row == pytest.approx(t.row, abs=0.05) and p.col == pytest.approx(t.col, abs=0.05)
        moving += t.sog_kn > 0
        if t.sog_kn > 0:
            assert p.shift_m >= 0
    assert moving >= 5
    assert pj.in_scene.all() and pj.on_water.all()
