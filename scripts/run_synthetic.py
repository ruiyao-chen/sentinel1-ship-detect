#!/usr/bin/env python3
"""Synthetic validation: every number checked against an injected known answer.

  1. False-alarm rate vs design rate (gamma and K sea; CA-CFAR and two-parameter).
  2. Detection probability vs signal-to-clutter ratio, measured vs closed-form theory.
  3. Length estimates for ships of known length (with and without a sidelobe cross).
  4. Censoring in a crowded anchorage.
  5. The demo scene end to end (coast, breakwater, structures, dark and moving ships),
     against truth and against synthetic AIS.

Writes results/synthetic/*.csv, summary.json, run_info.json and figures/synthetic_*.png.
Offline; ~3 minutes.

    python scripts/run_synthetic.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from s1ship import ais, cfar, clutter, config, data, pipeline, plotting, synthetic as S, targets as T  # noqa: E402
from s1ship import validation as V  # noqa: E402
from s1ship.scene import TruthTarget  # noqa: E402

OUT = config.RESULTS / "synthetic"
L = config.S1_ENL_IW_GRDH


def far_experiment(n_images: int, size: int) -> pd.DataFrame:
    cases = [
        ("gamma sea, gamma CA-CFAR", np.inf, dict(method="ca", nu=np.inf)),
        ("K sea (nu=2), K CA-CFAR", 2.0, dict(method="ca", nu=2.0)),
        ("K sea (nu=2), gamma CA-CFAR", 2.0, dict(method="ca", nu=np.inf)),
        ("gamma sea, 2-parameter (Gaussian t)", np.inf, dict(method="2p", nu=np.inf, t_model="gaussian")),
    ]
    rows = []
    images = {}
    for name, sea_nu, kw in cases:
        if sea_nu not in images:
            rng = np.random.default_rng(1000 + (0 if np.isinf(sea_nu) else 1))
            images[sea_nu] = [0.01 * clutter.sample((size, size), L, sea_nu, rng) for _ in range(n_images)]
        for pfa in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            n_t = n_d = 0
            for img in images[sea_nu]:
                r = cfar.detect(img, None, pfa=pfa, L=L, censor=False, **kw)
                n_t += r.n_tested
                n_d += r.n_detected
            rows.append(dict(case=name, design_pfa=pfa, n_tested=n_t, n_detected=n_d,
                             expected=pfa * n_t, measured_pfa=n_d / n_t, ratio=n_d / (pfa * n_t)))
    return pd.DataFrame(rows)


def pd_experiment(levels, per_level: int, n_images: int, pfa: float = 1e-6) -> pd.DataFrame:
    hits = {s: [] for s in levels}
    for k in range(n_images):
        rng = np.random.default_rng(2000 + k)
        pos = np.round(S.place_targets(len(levels) * per_level, (1024, 1024), rng, min_sep_px=60))
        scr = np.repeat(levels, per_level)
        tg = [TruthTarget(i, float(r), float(c), 0, 0, 0, float(s), kind="point")
              for i, ((r, c), s) in enumerate(zip(pos, scr))]
        sc = S.make_scene((1024, 1024), targets=tg, seed=3000 + k)
        res = cfar.detect(sc.vv, None, "ca", pfa=pfa, L=L, censor=False)
        for t in tg:
            hits[t.scr_db].append(bool(res.detections[int(t.row), int(t.col)]))
    n = float(cfar.ring_size(config.CFAR.guard_px, config.CFAR.background_px))
    theory = S.pd_point_theory(levels, pfa, L, n)
    return pd.DataFrame(dict(scr_db=levels, n_targets=[len(hits[s]) for s in levels],
                             pd_measured=[np.mean(hits[s]) for s in levels], pd_theory=theory))


def length_experiment() -> pd.DataFrame:
    rows = []
    cases = [("SCR 12 dB", 12.0, None, 50, 100, 1024), ("SCR 20 dB", 20.0, None, 50, 100, 1024),
             ("SCR 30 dB", 30.0, None, 50, 100, 1024), ("38 dB + sidelobe cross", 38.0, -21.0, 20, 220, 1536)]
    for k, (name, scr, sl, n, sep, size) in enumerate(cases):
        rng = np.random.default_rng(4000 + k)
        pos = S.place_targets(n, (size, size), rng, min_sep_px=sep, margin_px=120 if sl else 50)
        lengths = np.exp(rng.uniform(np.log(30 if sl is None else 80), np.log(350), n))
        tg = [TruthTarget(i, r, c, float(Lm), float(max(5, Lm / 6.5)), float(rng.uniform(0, 360)), scr)
              for i, ((r, c), Lm) in enumerate(zip(pos, lengths))]
        sc = S.make_scene((size, size), targets=tg, seed=5000 + k, correlated=True, sidelobe_db=sl)
        res = cfar.detect(sc.vv, None, "ca", pfa=1e-6)
        df = T.measure(res.detections, sc.vv, res.background, sc.grid, res.threshold)
        v = V.validate(df, V.truth_vessels(sc), sc.grid.area_km2, sc)
        m = v.dets[v.dets.matched]
        truth = V.truth_vessels(sc).set_index("mmsi")
        for _, t in m.iterrows():
            rows.append(dict(case=name, true_m=truth.loc[int(t.vessel_id), "length_m"], length_m=t.length_m,
                             length_core_m=t.length_core_m, length_moment_m=t.length_moment_m,
                             length_psf_m=t.length_psf_m, n_ships=n, n_matched=len(m),
                             n_unmatched=int((~v.dets.matched).sum()),
                             n_unexplained=int(v.dets.unexplained.sum())))
    return pd.DataFrame(rows)


def censoring_experiment() -> pd.DataFrame:
    tg, i = [], 0
    for r in range(150, 880, 40):
        for c in range(150, 880, 40):
            tg.append(TruthTarget(i, float(r), float(c), 250.0, 40.0, 35.0, 30.0))
            tg.append(TruthTarget(i + 1, r + 20.0, c + 20.0, 0, 0, 0, 10.0, kind="point"))
            i += 2
    sc = S.make_scene((1024, 1024), targets=tg, seed=9)
    small = [t for t in tg if t.kind == "point"]
    rows = []
    for censor in (False, True):
        res = cfar.detect(sc.vv, None, "ca", pfa=1e-6, L=L, censor=censor)
        rows.append(dict(censor=censor, n_small=len(small), n_large=len(small),
                         pd_small_10db=float(np.mean([res.detections[int(t.row), int(t.col)] for t in small]))))
    return pd.DataFrame(rows)


def demo_experiment():
    sc = data.demo_scene()
    out, figs = {}, {}
    for model in ("k", "gamma"):
        run = pipeline.detect_scene(sc, clutter_model=model)
        truth = V.truth_vessels(sc, run.water)
        for stage, dets in (("VV", run.targets), ("VV+VH", run.targets[run.targets.confirmed])):
            v = V.validate(dets.reset_index(drop=True), truth, run.water_km2, sc)
            out[f"{model}/{stage}/truth"] = v
        pos = ais.positions_at(S.synthetic_ais(sc), sc.time_utc)
        for shift in (True, False):
            pj = ais.project_to_scene(pos, sc, run.water, shift=shift)
            out[f"{model}/VV/ais{'' if shift else '_noshift'}"] = V.validate(run.targets, pj, run.water_km2, sc)
        if model == "k":
            figs["scene"] = plotting.plot_scene(sc, out["k/VV/truth"], "Synthetic demo scene: K-CFAR (Pfa 1e-6) vs injected truth",
                                                water=run.water, vmin=-30, vmax=0, ref="injected target")
            out["_fit"] = run.fit
    return sc, out, figs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fast", action="store_true", help="fewer images (smoke test)")
    args = p.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    config.FIGURES.mkdir(exist_ok=True)

    print("1. false-alarm rate vs design")
    far = far_experiment(n_images=2 if args.fast else 4, size=1000 if args.fast else 1500)
    far.to_csv(OUT / "far.csv", index=False)
    print(far[["case", "design_pfa", "n_detected", "expected", "ratio"]].to_string(index=False))
    plotting.plot_far(far).savefig(config.FIGURES / "synthetic_far.png", dpi=150)

    print("\n2. detection probability vs SCR")
    levels = list(range(0, 15))
    pdf = pd_experiment(levels, per_level=8, n_images=4 if args.fast else 12)
    pdf.to_csv(OUT / "pd_scr.csv", index=False)
    print(pdf.to_string(index=False))
    fine = np.linspace(0, 14, 57)
    plotting.plot_pd_scr(pdf.scr_db, pdf.pd_measured.to_numpy(), int(pdf.n_targets[0]), fine,
                         S.pd_point_theory(fine, 1e-6, L, float(cfar.ring_size(12, 40))), 1e-6
                         ).savefig(config.FIGURES / "synthetic_pd_scr.png", dpi=150)

    print("\n3. length estimates")
    ln = length_experiment()
    ln.to_csv(OUT / "length.csv", index=False)
    agg = []
    for case, g in ln.groupby("case", sort=False):
        row = dict(case=case, n_ships=int(g.n_ships.iloc[0]), n_matched=int(g.n_matched.iloc[0]),
                   n_unexplained=int(g.n_unexplained.iloc[0]))
        for k in ("length_m", "length_core_m", "length_psf_m"):
            e = g[k] - g.true_m
            row[f"{k}_bias"] = float(np.median(e))
            row[f"{k}_mae"] = float(np.median(np.abs(e)))
        row["length_psf_m_rel_mae"] = float(np.median(np.abs(g.length_psf_m - g.true_m) / g.true_m))
        agg.append(row)
    agg = pd.DataFrame(agg)
    agg.to_csv(OUT / "length_summary.csv", index=False)
    print(agg.round(1).to_string(index=False))
    plotting.plot_length_synthetic(ln).savefig(config.FIGURES / "synthetic_length.png", dpi=150)

    print("\n4. censoring")
    cen = censoring_experiment()
    cen.to_csv(OUT / "censoring.csv", index=False)
    print(cen.to_string(index=False))

    print("\n5. demo scene end to end")
    sc, demo, figs = demo_experiment()
    figs["scene"].savefig(config.FIGURES / "synthetic_scene.png", dpi=110)
    rows = []
    for key, v in demo.items():
        if key.startswith("_"):
            continue
        model, stage, ref = key.split("/")
        rows.append(dict(model=model, stage=stage, reference=ref, **{k: v.summary.get(k) for k in (
            "n_detections", "n_matched", "pd_all", "n_unmatched", "unmatched_per_km2", "n_unexplained",
            "unexplained_per_km2", "n_dark_candidates", "length_bias_m", "length_mae_m")}))
    dsum = pd.DataFrame(rows)
    dsum.to_csv(OUT / "demo_summary.csv", index=False)
    demo["k/VV/truth"].pd_table.to_csv(OUT / "demo_pd_by_length.csv", index=False)
    print(dsum.round(3).to_string(index=False))
    fit = demo["_fit"]
    truth_dark = sorted(t.id for t in sc.truth if not t.ais)
    found_dark = sorted(int(i) for i in demo["k/VV/truth"].dets.vessel_id.dropna() if int(i) in truth_dark)
    ais_unmatched = demo["k/VV/ais"].dets[~demo["k/VV/ais"].dets.matched]

    summary = dict(
        far=far.to_dict("records"),
        pd_scr_max_abs_diff=float(np.max(np.abs(pdf.pd_measured - pdf.pd_theory))),
        length=agg.to_dict("records"),
        censoring=cen.to_dict("records"),
        demo=dict(rows=dsum.to_dict("records"), k_fit=dict(L=fit.L, nu=fit.nu, enl_measured=fit.enl_measured),
                  injected_nu=float(sc.meta["nu"]), n_targets=len(sc.truth),
                  n_no_ais_targets=len(truth_dark), no_ais_targets_detected=len(found_dark),
                  ais_unmatched_are_no_ais_targets=bool(len(ais_unmatched) == len(found_dark))),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    info = dict(script="run_synthetic.py", fast=args.fast, seed_base=config.SEED,
                utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                python=platform.python_version(), runtime_s=round(time.time() - t0, 1),
                versions={m.__name__: m.__version__ for m in (np, pd)})
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"\ndone in {time.time() - t0:.0f} s -> {OUT.relative_to(config.ROOT)}, figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
