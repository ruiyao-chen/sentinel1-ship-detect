#!/usr/bin/env python3
"""Detect ships in a real Sentinel-1 scene and validate against AIS.

Needs the inputs from scripts/fetch_data.py (the scene cache, the AIS subset,
and optionally the fixed-structure mask). Writes results/real/ and
figures/real_*.png.

    python scripts/run_detect.py                       # Planetary Computer scene, LA/LB, 2021-10-21
    python scripts/run_detect.py --backend gee --project YOUR_EE_PROJECT

The main configuration is fixed in config.py (CA-CFAR, K clutter fitted to the
scene, Pfa 1e-6, 100 m land buffer, VH confirmation at Pfa 1e-4, AIS gate
200 m + half the length). --sensitivity re-runs it across Pfa, clutter model
and masks.
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
from scipy import ndimage  # noqa: E402

from s1ship import ais, cfar, clutter, config, data, pipeline, plotting, validation as V  # noqa: E402

OUT = config.RESULTS / "real"
KEYS = ("n_detections", "n_vessels_in_scene", "n_vessels_on_water", "n_matched_on_water", "pd_all",
        "n_unmatched", "unmatched_per_km2", "n_fragment", "n_sidelobe", "n_ambiguity", "n_near_structure",
        "n_unexplained",
        "unexplained_per_km2", "n_dark_candidates", "length_n", "length_bias_m", "length_mae_m",
        "length_rel_err_median", "length_spearman", "match_dist_median_m", "water_km2")


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Round for CSV without truncating identifiers: lon/lat to 1e-6 deg, metres to 0.1 m."""
    df = df.copy()
    for c in df.columns:
        if c in ("mmsi", "vessel_id"):
            df[c] = df[c].astype("Int64")
        elif c in ("lon", "lat"):
            df[c] = df[c].round(6)
        elif c == "peak":
            df[c] = df[c].map(lambda v: float(f"{v:.6g}"))
        elif df[c].dtype.kind == "f":
            df[c] = df[c].round(4 if c.endswith(("_db", "pfa", "per_km2", "pd")) else 2)
    return df


def run_once(scene, vessels, structures, **kw):
    run = pipeline.detect_scene(scene, exclude=structures, **kw)
    pj = ais.project_to_scene(vessels, scene, run.water)
    st = None if structures is None else structures & scene.water
    vv = V.validate(run.targets, pj, run.water_km2, scene, structures=st)
    conf = run.targets[run.targets.confirmed].reset_index(drop=True) if "confirmed" in run.targets else run.targets
    vh = V.validate(conf, pj, run.water_km2, scene, structures=st)
    return run, pj, vv, vh


def clutter_tail_sample(scene, water, vessels_px, guard, background, bright_ratio=30.0, max_n=3_000_000, seed=0):
    """I / local mean on sea pixels away from bright targets (600 m) and AIS vessels (300 m)."""
    img = scene.vv
    mean0, _, _ = cfar.ring_stats(img, water, guard, background)
    with np.errstate(invalid="ignore", divide="ignore"):
        bright = water & (img > bright_ratio * mean0)
    y, x = np.mgrid[-60:61, -60:61]
    excl = ndimage.binary_dilation(bright, structure=x * x + y * y <= 3600)
    pts = np.zeros(img.shape, dtype=bool)
    r, c = np.round(vessels_px[:, 0]).astype(int), np.round(vessels_px[:, 1]).astype(int)
    ok = (r >= 0) & (r < img.shape[0]) & (c >= 0) & (c < img.shape[1])
    pts[r[ok], c[ok]] = True
    y, x = np.mgrid[-30:31, -30:31]
    excl |= ndimage.binary_dilation(pts, structure=x * x + y * y <= 900)
    clean = water & ~excl
    mean, _, n = cfar.ring_stats(img, clean, guard, background)
    sel = clean & (n >= 0.25 * cfar.ring_size(guard, background)) & (mean > 0)
    ratio = img[sel] / mean[sel]
    if ratio.size > max_n:
        ratio = np.random.default_rng(seed).choice(ratio, max_n, replace=False)
    return ratio, int(sel.sum())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["pc", "gee"], default="pc")
    p.add_argument("--project", default=None)
    p.add_argument("--aoi", default=config.DEFAULT_AOI, choices=list(config.AOIS))
    p.add_argument("--date", default=config.DEFAULT_DATE)
    p.add_argument("--orbit", type=int, default=config.DEFAULT_ORBIT)
    p.add_argument("--no-sensitivity", action="store_true")
    args = p.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    scene = data.load_scene(args.backend, args.aoi, args.date, args.orbit, project=args.project)
    structures = data.load_structure_mask(args.aoi, args.orbit, scene.grid)
    ais_csv = config.DATA / "ais" / f"ais_{args.date}_{args.aoi}.csv.gz"
    reports = pd.read_csv(ais_csv)
    vessels = ais.positions_at(reports, scene.time_utc)
    print(f"scene {scene.scene_id}\n  {scene.time_utc}, incidence {scene.incidence_deg:.1f} deg, "
          f"look {scene.look_azimuth_deg:.1f} deg, grid {scene.grid.shape}")
    print(f"AIS: {len(reports):,} reports, {reports.MMSI.nunique()} MMSI; {len(vessels)} with a position at SAR time "
          f"({(vessels.method == 'interp').sum()} interpolated)")
    print(f"structure mask: {'none' if structures is None else int(structures.sum())} px")

    # --- main configuration ------------------------------------------------ #
    run, pj, vv, vh = run_once(scene, vessels, structures)
    st = None if structures is None else structures & scene.water
    fit, vfit = run.fit, run.vh_fit
    print(f"clutter fit VV: L {fit.L:.2f}, nu {fit.nu:.1f}, ENL {fit.enl_measured:.2f}; "
          f"VH: L {vfit.L:.2f}, nu {vfit.nu:.1f}, ENL {vfit.enl_measured:.2f}")
    print(f"tested water {run.water_km2:.1f} km2; targets {len(run.targets)} "
          f"({run.targets.attrs})")
    for name, v in (("VV", vv), ("VV+VH", vh)):
        s = v.summary
        print(f"\n[{name}] detections {s['n_detections']}, AIS on water {s['n_vessels_on_water']}, "
              f"detected {s['n_matched_on_water']} (Pd {s['pd_all']:.3f})")
        print(f"  unmatched {s['n_unmatched']} ({s['unmatched_per_km2']:.3f}/km2): fragment {s['n_fragment']}, "
              f"sidelobe {s['n_sidelobe']}, ambiguity {s['n_ambiguity']}, near structure {s['n_near_structure']}, "
              f"unexplained {s['n_unexplained']} "
              f"({s['unexplained_per_km2']:.3f}/km2), dark candidates {s['n_dark_candidates']}")
        if "length_n" in s:
            print(f"  length vs AIS (n={s['length_n']}): bias {s['length_bias_m']:+.1f} m, MAE {s['length_mae_m']:.1f} m, "
                  f"median rel err {s['length_rel_err_median']:.2f}, Spearman {s['length_spearman']:.2f}")
        print(v.pd_table.to_string(index=False))

    # moving vs anchored, and the azimuth-shift correction
    ev = vv.vessels
    moving = ev.sog_kn.fillna(0) >= 1.0
    noshift = V.validate(run.targets, ais.project_to_scene(vessels, scene, run.water, shift=False),
                         run.water_km2, scene, structures=st)
    extra = dict(
        n_moving=int(moving.sum()), pd_moving=float(ev.detected[moving].mean()) if moving.any() else None,
        n_stationary=int((~moving).sum()), pd_stationary=float(ev.detected[~moving].mean()),
        shift_m_median_moving=float(ev.shift_m[moving].median()) if moving.any() else None,
        shift_m_max=float(ev.shift_m.max()),
        matched_without_shift_correction=int(noshift.summary["n_matched_on_water"]),
        matched_with_shift_correction=int(vv.summary["n_matched_on_water"]),
    )
    print(f"\nmoving (>= 1 kn): {extra['n_moving']} vessels, Pd {extra['pd_moving']}; stationary {extra['n_stationary']}, "
          f"Pd {extra['pd_stationary']:.3f}; matched with/without azimuth-shift correction "
          f"{extra['matched_with_shift_correction']}/{extra['matched_without_shift_correction']}")

    # --- clutter tail ------------------------------------------------------ #
    in_scene = pj[pj.in_scene]
    ratio, n_sea = clutter_tail_sample(scene, run.water, in_scene[["row", "col"]].to_numpy(),
                                       config.CFAR.guard_px, config.CFAR.background_px)
    enl = clutter.enl_logcumulant(ratio)
    tail = []
    for t in (3, 4, 5, 6, 8, 10, 15, 20):
        tail.append(dict(t=t, empirical=float((ratio > t).mean()), n_above=int((ratio > t).sum()),
                         gamma_nominal=float(clutter.gamma_sf(t, config.S1_ENL_IW_GRDH)),
                         gamma_measured_enl=float(clutter.gamma_sf(t, enl)),
                         k_fit=float(clutter.k_sf(t, fit.L, fit.nu)) if np.isfinite(fit.nu) else None))
    tail = pd.DataFrame(tail)
    tail.to_csv(OUT / "clutter_tail.csv", index=False)
    print("\nclutter tail (sea away from bright targets and AIS vessels):")
    print(tail.to_string(index=False))

    # --- outputs ------------------------------------------------------------ #
    det = vv.dets.copy()
    det["vh_confirmed"] = det.confirmed
    det["dark_candidate"] = det.unexplained & det.vh_confirmed & (det.length_psf_m >= config.DARK_MIN_LENGTH_M)
    tidy(det).to_csv(OUT / "detections.csv", index=False)
    dark = det[det.dark_candidate].sort_values("peak_db_over_clutter", ascending=False)
    tidy(dark[["lon", "lat", "length_psf_m", "peak_db_over_clutter", "confirm_peak_db", "n_pixels"]]).to_csv(
        OUT / "dark_candidates.csv", index=False)
    tidy(ev.drop(columns=["name"], errors="ignore")).to_csv(OUT / "ais_vessels_evaluated.csv", index=False)
    pd_tab = pd.concat([vv.pd_table.assign(stage="VV"), vh.pd_table.assign(stage="VV+VH")], ignore_index=True)
    pd_tab.to_csv(OUT / "pd_by_length.csv", index=False)

    figs = {
        "real_scene": plotting.plot_scene(scene, vh, f"Sentinel-1A VV, San Pedro Bay, {scene.time_utc[:16]} UTC: "
                                          "VV CFAR + VH confirmation vs AIS", water=run.water),
        "real_pd_length": plotting.plot_pd_length({"VV K-CFAR": vv.pd_table, "VV + VH confirmation": vh.pd_table}),
        "real_clutter_tail": plotting.plot_clutter_tail(ratio, fit.L, fit.nu, enl),
        "real_dark_candidates": plotting.plot_chips(scene, dark, "Unexplained, VH-confirmed detections (dark candidates)"),
    }
    m = vh.dets[vh.dets.matched]
    if len(m):
        ais_len = pj.set_index("mmsi").reindex(m.vessel_id.astype("int64")).length_m.to_numpy()
        figs["real_length"] = plotting.plot_length_vs_ais(m.length_psf_m.to_numpy(), ais_len, m.length_core_m.to_numpy())
        lstats = {}
        for k in ("length_m", "length_core_m", "length_moment_m", "length_psf_m"):
            e, ok = m[k].to_numpy() - ais_len, np.isfinite(ais_len)
            lstats[k] = dict(n=int(ok.sum()), bias_m=float(np.median(e[ok])), mae_m=float(np.median(np.abs(e[ok]))),
                             rel_err_median=float(np.median(np.abs(e[ok]) / ais_len[ok])),
                             within_25pct=float(np.mean(np.abs(e[ok]) / ais_len[ok] <= 0.25)))
        est, ok = m.length_psf_m.to_numpy(), np.isfinite(ais_len)
        lstats["n_large_read_short"] = int(((ais_len > 150) & (est < 0.5 * ais_len)).sum())
        lstats["n_large"] = int((ais_len > 150).sum())
        lstats["n_small_read_long"] = int(((ais_len < 60) & (est > 2 * ais_len)).sum())
        lstats["n_small"] = int((ais_len < 60).sum())
        print("\nlength vs AIS (VV+VH matched):", json.dumps(lstats, indent=1, default=float))
    for name, f in figs.items():
        f.savefig(config.FIGURES / f"{name}.png", dpi=150 if name != "real_scene" else 110)

    # --- sensitivity ------------------------------------------------------- #
    sens = []
    if not args.no_sensitivity:
        variants = [dict(label="main: K, Pfa 1e-6", kw={})]
        variants += [dict(label=f"K, Pfa {p:g}", kw=dict(pfa=p)) for p in (1e-5, 1e-7, 1e-8)]
        variants += [dict(label="gamma (L=4.4), Pfa 1e-6", kw=dict(clutter_model="gamma")),
                     dict(label="no censoring", kw=dict(censor=False)),
                     dict(label="land buffer 200 m", kw=dict(land_buffer_m=200.0)),
                     dict(label="no structure mask", kw={}, no_struct=True)]
        for var in variants:
            r, _, a, b = run_once(scene, vessels, None if var.get("no_struct") else structures, **var["kw"])
            for stage, v in (("VV", a), ("VV+VH", b)):
                sens.append(dict(variant=var["label"], stage=stage, **{k: v.summary.get(k) for k in KEYS}))
            print(f"  {var['label']:<26} VV Pd {a.summary['pd_all']:.3f} unexpl/km2 {a.summary['unexplained_per_km2']:.3f} | "
                  f"VV+VH Pd {b.summary['pd_all']:.3f} unexpl/km2 {b.summary['unexplained_per_km2']:.3f}")
        tidy(pd.DataFrame(sens)).to_csv(OUT / "sensitivity.csv", index=False)

    summary = dict(
        scene=dict(id=scene.scene_id, time_utc=scene.time_utc, source=scene.source, incidence_deg=scene.incidence_deg,
                   look_azimuth_deg=scene.look_azimuth_deg, orbit_pass=scene.orbit_pass, grid=scene.grid.as_dict(),
                   meta={k: v for k, v in scene.meta.items() if k != "stac_ids"}),
        ais=dict(file=ais_csv.name, reports=int(len(reports)), mmsi=int(reports.MMSI.nunique()),
                 positioned=int(len(vessels)), interpolated=int((vessels.method == "interp").sum()),
                 in_scene=int(pj.in_scene.sum()), on_water=int(pj.on_water.sum())),
        clutter_fit=dict(vv=dict(L=fit.L, nu=fit.nu, enl=fit.enl_measured),
                         vh=dict(L=vfit.L, nu=vfit.nu, enl=vfit.enl_measured),
                         tail_sample_pixels=n_sea, tail_enl=enl),
        structures_px_on_water=int((structures & scene.water_buffered(config.LAND_BUFFER_M)).sum())
        if structures is not None else None,
        targets_attrs=dict(run.targets.attrs),
        vv=vv.summary, vv_vh=vh.summary, motion=extra, length_vs_ais=lstats if len(m) else None,
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    man = json.loads((config.DATA / "manifest.json").read_text()) if (config.DATA / "manifest.json").exists() else {}
    info = dict(script="run_detect.py", args=vars(args), utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                runtime_s=round(time.time() - t0, 1), python=platform.python_version(),
                versions={m_.__name__: m_.__version__ for m_ in (np, pd)},
                config=dict(cfar=config.CFAR.__dict__, land_buffer_m=config.LAND_BUFFER_M,
                            merge_radius_px=config.MERGE_RADIUS_PX, min_target_pixels=config.MIN_TARGET_PIXELS,
                            core_db=config.CORE_DB, vh_confirm_pfa=config.VH_CONFIRM_PFA,
                            match_gate_m=config.MATCH_GATE_M, dark_min_length_m=config.DARK_MIN_LENGTH_M),
                inputs=man)
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    print(f"\ndone in {time.time() - t0:.0f} s -> {OUT.relative_to(config.ROOT)}, figures/real_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
