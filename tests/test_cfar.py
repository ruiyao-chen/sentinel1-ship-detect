"""CFAR detector: window statistics, false-alarm rate against the design rate,
detection probability against theory, land masking, and censoring."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import L_IW, far_tolerance, gamma_image
from s1ship import cfar, synthetic as S
from s1ship.scene import TruthTarget


# --------------------------------------------------------------------------- #
# Window statistics
# --------------------------------------------------------------------------- #
def test_box_sum_matches_brute_force(rng):
    a = rng.random((37, 51))
    for h in (0, 1, 4):
        got = cfar.box_sum(a, h)
        padded = np.pad(a, h)
        for r, c in [(0, 0), (5, 7), (36, 50), (18, 0), (0, 25)]:
            assert got[r, c] == pytest.approx(padded[r:r + 2 * h + 1, c:c + 2 * h + 1].sum())


def test_ring_stats_match_masked_brute_force(rng):
    img = rng.gamma(4.4, 1 / 4.4, size=(60, 60))
    w = rng.random((60, 60)) > 0.3
    g, b = 2, 6
    mean, sd, n = cfar.ring_stats(img, w, g, b)
    for r, c in [(30, 30), (3, 3), (59, 10)]:
        ring = np.zeros_like(w)
        ring[max(r - b, 0):r + b + 1, max(c - b, 0):c + b + 1] = True
        ring[max(r - g, 0):r + g + 1, max(c - g, 0):c + g + 1] = False
        vals = img[ring & w]
        assert n[r, c] == vals.size
        assert mean[r, c] == pytest.approx(vals.mean())
        assert sd[r, c] == pytest.approx(vals.std(), rel=1e-6)


# --------------------------------------------------------------------------- #
# False-alarm rate: measured vs design
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pfa", [1e-3, 1e-4])
def test_gamma_ca_cfar_holds_its_design_false_alarm_rate(rng, pfa):
    img = gamma_image((1000, 1000), rng)
    res = cfar.detect(img, None, "ca", pfa=pfa, L=L_IW, censor=False)
    expected = pfa * res.n_tested
    assert abs(res.n_detected - expected) < far_tolerance(res.n_tested, pfa)


def test_k_ca_cfar_holds_rate_on_k_clutter_where_gamma_model_fails(rng):
    nu, pfa = 2.0, 1e-3
    img = gamma_image((1000, 1000), rng, nu=nu)
    k = cfar.detect(img, None, "ca", pfa=pfa, L=L_IW, nu=nu, censor=False)
    g = cfar.detect(img, None, "ca", pfa=pfa, L=L_IW, censor=False)
    assert abs(k.n_detected - pfa * k.n_tested) < far_tolerance(k.n_tested, pfa)
    assert g.n_detected > 10 * pfa * g.n_tested      # heavy tail -> gamma CFAR floods


def test_censoring_does_not_change_the_false_alarm_rate(rng):
    pfa = 1e-3
    img = gamma_image((800, 800), rng)
    res = cfar.detect(img, None, "ca", pfa=pfa, L=L_IW, censor=True)
    assert abs(res.n_detected - pfa * res.n_tested) < far_tolerance(res.n_tested, pfa)


def test_two_parameter_gaussian_threshold_overdetects_speckle(rng):
    pfa = 1e-4
    img = gamma_image((800, 800), rng)
    gauss = cfar.detect(img, None, "2p", pfa=pfa, L=L_IW, censor=False, t_model="gaussian")
    model = cfar.detect(img, None, "2p", pfa=pfa, L=L_IW, censor=False, t_model="gamma")
    assert gauss.n_detected > 5 * pfa * gauss.n_tested
    assert abs(model.n_detected - pfa * model.n_tested) < far_tolerance(model.n_tested, pfa)


def test_fit_clutter_recovers_texture_of_k_sea(rng):
    img = gamma_image((800, 800), rng, nu=3.0)
    fit = cfar.fit_clutter(img, np.ones(img.shape, dtype=bool), L_nominal=L_IW)
    assert fit.L == pytest.approx(L_IW)
    assert fit.nu == pytest.approx(3.0, rel=0.1)


# --------------------------------------------------------------------------- #
# Detection probability: rises with SCR and matches theory
# --------------------------------------------------------------------------- #
SCR_LEVELS = [0, 3, 5, 7, 9, 11, 13, 16]


def test_detection_probability_rises_with_scr_and_matches_theory():
    pfa, per_level = 1e-6, 14
    hits = {s: [] for s in SCR_LEVELS}
    for k in range(6):
        rng = np.random.default_rng(100 + k)
        # 60 px apart: no target sits in another's 81 x 81 background window
        pos = np.round(S.place_targets(len(SCR_LEVELS) * per_level, (1024, 1024), rng, min_sep_px=60))
        scr = np.repeat(SCR_LEVELS, per_level)
        tg = [TruthTarget(i, float(r), float(c), 0, 0, 0, float(s), kind="point")
              for i, ((r, c), s) in enumerate(zip(pos, scr))]
        sc = S.make_scene((1024, 1024), targets=tg, seed=200 + k)
        res = cfar.detect(sc.vv, None, "ca", pfa=pfa, L=L_IW, censor=False)
        for t in tg:
            hits[t.scr_db].append(res.detections[int(t.row), int(t.col)])
    measured = np.array([np.mean(hits[s]) for s in SCR_LEVELS])
    theory = S.pd_point_theory(SCR_LEVELS, pfa, L_IW, n=float(cfar.ring_size(12, 40)))
    n = 6 * per_level
    assert np.all(np.abs(measured - theory) < 4 * np.sqrt(theory * (1 - theory) / n) + 0.03)
    assert np.all(np.diff(measured) >= -0.05)        # non-decreasing up to sampling noise
    assert measured[0] < 0.05 and measured[-1] == 1.0


# --------------------------------------------------------------------------- #
# Land masking
# --------------------------------------------------------------------------- #
def test_land_mask_removes_land_false_alarms_and_keeps_coastal_rate():
    pfa = 1e-4
    sc = S.make_scene((1024, 1024), land=True, seed=3)
    water = sc.water_buffered(30.0)
    masked = cfar.detect(sc.vv, water, "ca", pfa=pfa, L=L_IW, censor=False)
    unmasked = cfar.detect(sc.vv, None, "ca", pfa=pfa, L=L_IW, censor=False)
    land = ~sc.water
    assert not (masked.detections & land).any()
    assert abs(masked.n_detected - pfa * masked.n_tested) < far_tolerance(masked.n_tested, pfa)
    # water pixels within 20 px of the coast: their rings straddle land, masked stats keep the rate
    near = masked.valid & ~sc.water_buffered(30.0 + 200.0)
    assert near.sum() > 20_000
    assert masked.detections[near].sum() < pfa * near.sum() + far_tolerance(int(near.sum()), pfa)
    # without the mask, land scatterers flood the output
    assert (unmasked.detections & land).sum() > 30 * pfa * unmasked.n_tested


# --------------------------------------------------------------------------- #
# Censoring in a crowded anchorage
# --------------------------------------------------------------------------- #
def test_censoring_recovers_small_targets_between_large_bright_ships():
    tg, i = [], 0
    for r in range(150, 880, 40):
        for c in range(150, 880, 40):
            tg.append(TruthTarget(i, float(r), float(c), 250.0, 40.0, 35.0, 30.0))
            tg.append(TruthTarget(i + 1, r + 20.0, c + 20.0, 0, 0, 0, 10.0, kind="point"))
            i += 2
    sc = S.make_scene((1024, 1024), targets=tg, seed=9)
    small = [t for t in tg if t.kind == "point"]
    pd = {}
    for censor in (False, True):
        res = cfar.detect(sc.vv, None, "ca", pfa=1e-6, L=L_IW, censor=censor)
        pd[censor] = np.mean([res.detections[int(t.row), int(t.col)] for t in small])
    assert pd[False] < 0.2 and pd[True] > 0.9


def test_unknown_method_rejected(rng):
    with pytest.raises(ValueError):
        cfar.detect(gamma_image((50, 50), rng), method="os")
