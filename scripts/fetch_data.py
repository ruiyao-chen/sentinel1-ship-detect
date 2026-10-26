#!/usr/bin/env python3
"""Fetch the real-data inputs for one AOI and day: Sentinel-1 and AIS.

Examples
--------
# Sentinel-1 from Planetary Computer (no account) + the AIS day file:
python scripts/fetch_data.py --backend pc --aoi lalb --date 2021-10-21 --orbit 71

# Sentinel-1 from Google Earth Engine instead (registered EE Cloud project):
python scripts/fetch_data.py --backend gee --project YOUR_EE_PROJECT --aoi lalb --date 2021-10-21 --orbit 71

# Fixed-structure mask from nine other passes of the same orbit (Planetary Computer):
python scripts/fetch_data.py --what structures

# Only AIS, or only Sentinel-1:
python scripts/fetch_data.py --what ais
python scripts/fetch_data.py --what s1 --backend pc

# List Sentinel-1 passes over the AOI in a month (Planetary Computer STAC):
python scripts/fetch_data.py --list 2021-10-01 2021-10-31

The AIS day file (~350 MB zipped, all U.S. waters) is streamed to data/raw,
filtered to the AOI box, and deleted; the subset goes to
data/ais/ais_<date>_<aoi>.csv.gz. data/manifest.json records URLs, sizes and
SHA-256 of what was downloaded and kept.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from s1ship import ais, config, data  # noqa: E402


def _update_manifest(key: str, entry: dict) -> None:
    path = config.DATA / "manifest.json"
    man = json.loads(path.read_text()) if path.exists() else {}
    entry["fetched_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    man[key] = entry
    path.write_text(json.dumps(man, indent=2) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aoi", default=config.DEFAULT_AOI, choices=list(config.AOIS))
    p.add_argument("--date", default=config.DEFAULT_DATE)
    p.add_argument("--orbit", type=int, default=config.DEFAULT_ORBIT, help="relative orbit (0 = any)")
    p.add_argument("--backend", choices=["pc", "gee"], default="pc")
    p.add_argument("--project", default=None, help="Earth Engine Cloud project (gee backend)")
    p.add_argument("--what", default="ais,s1,structures", help="comma list: ais, s1, structures")
    p.add_argument("--keep-raw", action="store_true", help="keep the full AIS day zip in data/raw")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--list", nargs=2, metavar=("START", "END"), help="list S1 passes over the AOI and exit")
    args = p.parse_args()
    orbit = args.orbit or None

    if args.list:
        for a in data.list_acquisitions(args.aoi, *args.list):
            print(f"{a['datetime']}  orbit {a['orbit']:>3}  {a['pass_']:<10}  {a['id']}")
        return 0

    aoi = config.AOIS[args.aoi]
    what = {w.strip() for w in args.what.split(",")}

    if "ais" in what:
        out = config.DATA / "ais" / f"ais_{args.date}_{args.aoi}.csv.gz"
        if out.exists() and not args.refresh:
            print(f"AIS subset exists: {out.relative_to(config.ROOT)}")
        else:
            t0 = time.time()
            print(f"AIS: {ais.day_url(args.date)} -> {out.relative_to(config.ROOT)}")
            man = ais.fetch_day(args.date, aoi, out, keep_raw=args.keep_raw)
            print(f"  {man['bytes'] / 1e6:.0f} MB downloaded, kept {man['rows_kept']:,} reports from "
                  f"{man['vessels_kept']} vessels ({time.time() - t0:.0f} s)")
            _update_manifest(f"ais_{args.date}_{args.aoi}", man)

    if "s1" in what:
        t0 = time.time()
        sc = data.load_scene(args.backend, args.aoi, args.date, orbit, project=args.project, refresh=args.refresh)
        print(f"Sentinel-1 ({args.backend}): {sc.scene_id}")
        print(f"  {sc.time_utc}  {sc.orbit_pass}  incidence {sc.incidence_deg:.1f} deg  "
              f"look azimuth {sc.look_azimuth_deg:.1f} deg  grid {sc.grid.shape}  ({time.time() - t0:.0f} s)")
        _update_manifest(f"s1_{args.backend}_{args.aoi}_{args.date}", dict(
            scene_id=sc.scene_id, time_utc=sc.time_utc, grid=sc.grid.as_dict(), source=sc.source,
            meta={k: v for k, v in sc.meta.items() if k != "stac_ids"}))
    if "structures" in what:
        out = config.DATA / f"structures_{args.aoi}_o{orbit}.npz"
        if out.exists() and not args.refresh:
            print(f"structure mask exists: {out.relative_to(config.ROOT)}")
        else:
            t0 = time.time()
            mask, info = data.structure_mask(args.aoi, orbit, list(config.STRUCTURE_REF_DATES),
                                             ratio=config.STRUCTURE_RATIO)
            np.savez_compressed(out, mask=np.packbits(mask), shape=np.array(mask.shape), info=json.dumps(info))
            print(f"structure mask: {info['n_pixels']} px from {len(info['dates'])} passes ({time.time() - t0:.0f} s)")
            _update_manifest(f"structures_{args.aoi}_o{orbit}", info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
