"""Clustering and target measurements against injected ships of known size."""

from __future__ import annotations

import numpy as np
import pytest

from s1ship import cfar, synthetic as S, targets as T
from s1ship.scene import TruthTarget


def test_fragments_within_merge_radius_become_one_target():
    det = np.zeros((40, 40), dtype=bool)
    det[10, 10:13] = True
    det[10, 15:18] = True                  # 2-px gap
    assert T.label_targets(det, merge_radius_px=2)[1] == 1
    assert T.label_targets(det, merge_radius_px=0)[1] == 2


def test_measure_on_a_noise_free_ship_recovers_length_and_heading():
    grid = S.default_grid((200, 200))
    patch, r0, c0 = S.ship_pattern(200.0, 30.0, 60.0, 100.0, 100.0, psf_sigma_px=0.0001, hull_sd=0.0,
                                   rng=np.random.default_rng(0))
    img = np.full((200, 200), 1e-6)
    S._paste(img, patch, r0, c0, 1.0)
    det = img > 0.5
    df = T.measure(det, img, np.full(img.shape, 1e-6), grid, min_pixels=1)
    assert len(df) == 1
    assert df.length_m[0] == pytest.approx(200.0, abs=15.0)
    assert df.orientation_deg[0] == pytest.approx(60.0, abs=2.0)
    assert df.row[0] == pytest.approx(100.0, abs=0.5) and df.col[0] == pytest.approx(100.0, abs=0.5)


def test_length_estimates_on_ships_in_clutter():
    """At SCR 20 dB the raw extent reads ~30 m long (impulse-response blur);
    the blur correction removes most of that."""
    rng = np.random.default_rng(3)
    shape = (1024, 1024)
    pos = S.place_targets(50, shape, rng, min_sep_px=100)
    lengths = np.exp(rng.uniform(np.log(40), np.log(350), len(pos)))
    tg = [TruthTarget(i, r, c, float(L), float(max(5, L / 6.5)), float(rng.uniform(0, 360)), 20.0)
          for i, ((r, c), L) in enumerate(zip(pos, lengths))]
    sc = S.make_scene(shape, targets=tg, seed=5, correlated=True)
    res = cfar.detect(sc.vv, sc.water, "ca", pfa=1e-6)
    df = T.measure(res.detections, sc.vv, res.background, sc.grid, res.threshold)
    truth = np.array([[t.row, t.col] for t in tg])
    d = np.hypot(df.row.values[:, None] - truth[:, 0], df.col.values[:, None] - truth[:, 1])
    j, ok = d.argmin(1), d.min(1) < 3
    assert ok.sum() == len(tg) == len(df)                       # every ship, nothing else
    true_len = lengths[j[ok]]
    raw = df.length_m.values[ok] - true_len
    psf = df.length_psf_m.values[ok] - true_len
    assert np.median(raw) > 20.0
    assert abs(np.median(psf)) < 8.0
    assert np.median(np.abs(psf) / true_len) < 0.08
    assert np.median(d.min(1)[ok]) < 1.0                        # centroid within a pixel


def test_size_filters():
    grid = S.default_grid((100, 100))
    det = np.zeros((100, 100), dtype=bool)
    det[10, 10] = True                       # single pixel: removed
    det[50, 5:95] = True                     # 900 m line: removed
    det[80:83, 40:46] = True                 # small boat: kept
    img = det * 1.0 + 0.01
    df = T.measure(det, img, np.full(det.shape, 0.01), grid, min_pixels=2, max_length_m=500.0)
    assert len(df) == 1
    assert df.attrs["n_removed_small"] == 1 and df.attrs["n_removed_long"] == 1


def test_core_length_ignores_the_sidelobe_cross():
    """Bright ships (38 dB) with a -21 dB sidelobe cross: the raw extent runs
    out along the arms, the core length does not, and the arm fragments that
    break away are flagged as sidelobes."""
    from s1ship import validation as V
    rng = np.random.default_rng(1)
    pos = S.place_targets(12, (1024, 1024), rng, min_sep_px=220, margin_px=120)
    lengths = np.exp(rng.uniform(np.log(80), np.log(330), len(pos)))
    tg = [TruthTarget(i, r, c, float(L), float(L / 6.5), float(rng.uniform(0, 360)), 38.0)
          for i, ((r, c), L) in enumerate(zip(pos, lengths))]
    sc = S.make_scene((1024, 1024), targets=tg, seed=2, correlated=True, sidelobe_db=-21.0)
    res = cfar.detect(sc.vv, None, "ca", pfa=1e-6)
    df = T.measure(res.detections, sc.vv, res.background, sc.grid, res.threshold)
    truth = V.truth_vessels(sc)
    v = V.validate(df, truth, 100.0, sc)
    m = v.dets[v.dets.matched]
    true_len = truth.set_index("mmsi").loc[m.vessel_id.astype(int), "length_m"].to_numpy()
    assert v.summary["n_matched"] == len(tg)
    assert np.median(m.length_m - true_len) > 200                   # arms included
    assert abs(np.median(m.length_psf_m - true_len)) < 10
    un = v.dets[~v.dets.matched]
    # all but design-rate clutter (~1 alarm at 1e-6 over 1M pixels) is explained
    assert len(un) > 5 and (un.sidelobe | un.fragment).sum() >= len(un) - 2
