"""Matching and validation metrics, and the whole synthetic pipeline against truth."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from s1ship import ais, data, pipeline, synthetic as S, validation as V


def _df(**cols):
    return pd.DataFrame(cols)


def test_match_is_one_to_one_and_gated_by_length():
    dets = _df(x=[0.0, 30.0, 1000.0], y=[0.0, 0.0, 0.0])
    ships = _df(x=[10.0, 1250.0], y=[0.0, 0.0], length_m=[50.0, 300.0])
    m = V.match(dets, ships, gate_m=200.0)
    assert len(m) == 2
    assert set(m.det_idx) == {0, 2}                 # the closer of the two near ship 0
    m2 = V.match(dets, ships.assign(length_m=[50.0, 0.0]), gate_m=200.0)
    assert len(m2) == 1                              # 250 m is outside 200 m + 0


def test_wilson_interval():
    lo, hi = V.wilson(np.array([0, 5, 10]), np.array([10, 10, 10]))
    assert lo[0] == pytest.approx(0.0, abs=1e-12) and 0.25 < hi[0] < 0.32
    assert lo[1] < 0.5 < hi[1]
    assert hi[2] == pytest.approx(1.0) and 0.68 < lo[2] < 0.75


def test_pd_by_length_bins():
    v = _df(length_m=[10.0, 30.0, 30.0, 150.0, np.nan])
    t = V.pd_by_length(v, np.array([False, True, False, True, True])).set_index("length_bin")
    assert t.loc["25-50 m", "n_vessels"] == 2 and t.loc["25-50 m", "pd"] == 0.5
    assert t.loc["unknown", "n_detected"] == 1 and t.loc["all", "n_vessels"] == 5


def test_azimuth_ambiguity_flags():
    look, off = 283.0, 4600.0
    fl = np.radians(look - 90.0)
    u = np.array([np.sin(fl), np.cos(fl)])
    pts = np.array([[0.0, 0.0], off * u, -off * u + [10, 10], [off, 0.0], [0.0, 9000.0]])
    dets = _df(x=pts[:, 0], y=pts[:, 1], peak=[100.0, 0.5, 0.5, 0.5, 100.0])
    flags = V.flag_ambiguities(dets, look, off)
    assert flags.tolist() == [False, True, True, False, False]


def test_synthetic_scene_end_to_end():
    """The demo scene: K sea, coast, breakwater, 70 ships (some dark, some moving)
    and 3 fixed structures, all known."""
    sc = data.demo_scene()
    run = pipeline.detect_scene(sc, clutter_model="k")
    truth = V.truth_vessels(sc, run.water)
    v = V.validate(run.targets, truth, run.water_km2, sc)
    s = v.summary
    # every detection is a real target: nothing from land, the breakwater or sea clutter
    assert s["n_unmatched"] == 0
    # big ships are always found; Pd falls toward small ones (the injected SCR-length relation)
    t = v.pd_table.set_index("length_bin")
    assert t.loc["100-200 m", "pd"] == 1.0 and t.loc["200-400 m", "pd"] == 1.0
    assert t.loc["0-25 m", "pd"] < t.loc["50-100 m", "pd"]
    assert abs(s["length_bias_m"]) < 10 and s["length_rel_err_median"] < 0.1
    # the three structures (no AIS) are detected
    det_ids = set(v.dets.vessel_id.dropna().astype(int))
    assert {900, 901, 902} <= det_ids

    # Against synthetic AIS: the unmatched detections are exactly the detected
    # targets without AIS, and correcting the azimuth shift matters.
    pos = ais.positions_at(S.synthetic_ais(sc), sc.time_utc)
    va = V.validate(run.targets, ais.project_to_scene(pos, sc, run.water), run.water_km2, sc)
    no_ais = {t.id for t in sc.truth if not t.ais}
    assert va.summary["n_unmatched"] == len(det_ids & no_ais)
    unshifted = V.validate(run.targets, ais.project_to_scene(pos, sc, run.water, shift=False), run.water_km2, sc)
    assert va.summary["n_matched"] > unshifted.summary["n_matched"]

    # The gamma model on the same textured sea: false alarms appear.
    g = pipeline.detect_scene(sc, clutter_model="gamma")
    vg = V.validate(g.targets, truth, g.water_km2, sc)
    assert vg.summary["unmatched_per_km2"] > 0.1


def test_sidelobe_flags_only_on_range_and_azimuth_axes():
    look = 283.0
    rg = np.array([np.sin(np.radians(look)), np.cos(np.radians(look))])
    az = np.array([np.sin(np.radians(look - 90)), np.cos(np.radians(look - 90))])
    diag = (rg + az) / np.sqrt(2)
    # 0 bright; 1, 2 on its range / azimuth axes; 3 off-axis; 4 on-axis but too far;
    # 5 on-axis but nearly as bright
    pts = np.array([[0.0, 0.0], 400 * rg, -900 * az, 400 * diag, 3500 * rg, 1000 * rg])
    dets = _df(x=pts[:, 0], y=pts[:, 1], peak=[1000.0, 5.0, 5.0, 5.0, 5.0, 900.0])
    assert V.flag_sidelobes(dets, look).tolist() == [False, True, True, False, False, False]


def test_vh_confirmation_separates_ships_from_clutter_alarms():
    """Gamma-model CFAR on the K-textured demo sea makes ~100 clutter alarms;
    VH (ships ~3 dB brighter relative to sea than in VV, independent speckle)
    confirms the ships and almost none of the alarms."""
    sc = data.demo_scene()
    run = pipeline.detect_scene(sc, clutter_model="gamma")
    v = V.validate(run.targets, V.truth_vessels(sc, run.water), run.water_km2, sc)
    d = v.dets
    assert (~d.matched).sum() > 50
    assert d[d.matched].confirmed.mean() > 0.9
    assert d[~d.matched].confirmed.mean() < 0.1
    assert v.summary["n_dark_candidates"] <= 0.1 * (~d.matched).sum()


def test_near_structure_category():
    sc = S.make_scene((200, 200), seed=0)
    st = np.zeros((200, 200), dtype=bool)
    st[100, 100] = True                                  # a platform
    x, y = sc.grid.rowcol_to_xy(np.array([100.0, 110.0, 180.0]), np.array([120.0, 100.0, 20.0]))
    dets = _df(x=x, y=y, row=[100.0, 110.0, 180.0], col=[120.0, 100.0, 20.0], peak=[1.0, 1.0, 1.0],
               length_psf_m=[50.0, 50.0, 50.0])
    none = _df(x=[1e9], y=[1e9], length_m=[10.0], on_water=[True], mmsi=[1])
    v = V.validate(dets, none, 4.0, sc, structures=st)
    assert v.dets.near_structure.tolist() == [True, True, False]
    assert v.dets.dark_candidate.tolist() == [False, False, True]
