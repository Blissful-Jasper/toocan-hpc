#!/usr/bin/env python3
import argparse, logging, os, sys, json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toocan_hpc.catalog_run import run_catalog
ap = argparse.ArgumentParser(description="TOOCAN-HPC catalog (FileTracking) stage, parallel over day files")
ap.add_argument("toocan_cfg"); ap.add_argument("geo_cfg"); ap.add_argument("start"); ap.add_argument("end")
ap.add_argument("lonmin", type=float); ap.add_argument("lonmax", type=float); ap.add_argument("latmin", type=float); ap.add_argument("latmax", type=float)
ap.add_argument("--masks", required=True, help="root of the mask tree (pathout_TOOCAN of the segmentation stage)")
ap.add_argument("--out", required=True); ap.add_argument("--nproc", type=int, default=16)
ap.add_argument("--ellipse", action="store_true", help="also compute equivalent ellipses (not part of the original output)")
ap.add_argument("--no-legacy", action="store_true")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
f = "%Y-%m-%dT%H:%M"
r = run_catalog(a.toocan_cfg, a.geo_cfg, datetime.strptime(a.start, f), datetime.strptime(a.end, f),
                (a.lonmin, a.lonmax, a.latmin, a.latmax), a.masks, a.out, nproc=a.nproc, ellipse=a.ellipse, legacy=not a.no_legacy)
print(json.dumps(r, indent=1, default=str))
