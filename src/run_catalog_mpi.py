#!/usr/bin/env python3
"""Multi-node catalog stage with mpi4py.  No communication until the final gather:
rank r processes the 6-slot blocks r, r+P, r+2P, ...  (slots are independent), rank 0 merges and writes the catalog.

    srun --mpi=pmix_v5 -n 128 python run_catalog_mpi.py <toocan.dat> <geo.dat> START END lon0 lon1 lat0 lat1 --masks M --out O
(Launch with the MPI that mpi4py was built against, e.g. `srun --mpi=pmix`; a `mpirun` from a different MPI would start
 independent single-rank copies.  Validated with 8 ranks: all 51 catalog variables bit-identical to the original.)
"""
import argparse, logging, os, sys, time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpi4py import MPI                                    # noqa: E402
import numpy as np                                        # noqa: E402

from toocan_hpc import legacy_format as lf                # noqa: E402
from toocan_hpc.catalog import catalog_block, mask_blocks                # noqa: E402
from toocan_hpc.catalog_run import _mask_day_files, build_obs   # noqa: E402
from toocan_hpc.navigation import crop_navigation         # noqa: E402
from toocan_hpc.params import parse_param_file            # noqa: E402
from toocan_hpc.readers import scan_inputs                # noqa: E402


def main():
    comm = MPI.COMM_WORLD; rank, size = comm.Get_rank(), comm.Get_size()
    ap = argparse.ArgumentParser()
    ap.add_argument("toocan_cfg"); ap.add_argument("geo_cfg"); ap.add_argument("start"); ap.add_argument("end")
    for k in ("lonmin", "lonmax", "latmin", "latmax"): ap.add_argument(k, type=float)
    ap.add_argument("--masks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ellipse", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING, format="%(asctime)s %(message)s")
    f = "%Y-%m-%dT%H:%M"; start, end = datetime.strptime(a.start, f), datetime.strptime(a.end, f)
    pt, pg = parse_param_file(a.toocan_cfg), parse_param_file(a.geo_cfg)
    box = (a.lonmin, a.lonmax, a.latmin, a.latmax)
    nav = crop_navigation(pg["file_navigation"], *box)
    t0 = time.perf_counter()
    recs = scan_inputs(pg["path_ir"], start, end) if rank == 0 else None
    recs = comm.bcast(recs, root=0)
    EP = datetime(1970, 1, 1); ir_by_day = {}
    for fn, t, ti in recs:
        ir_by_day.setdefault(t.date(), {})[float((t - EP).total_seconds())] = (fn, ti)
    masks = _mask_day_files(a.masks, str(pt.get("version", "2.08")), pg.get("REGION", "UNKNOWN"), start, end)
    days = sorted(set(masks) & set(ir_by_day))
    tasks = [(masks[d], ir_by_day[d], nav, a.ellipse, i0, i1) for d in days for (i0, i1) in mask_blocks(masks[d], 6)]
    ep, sl = [], []
    for t in tasks[rank::size]:            # static round-robin over 6-slot blocks (no communication before the gather)
        e, s = catalog_block(t); ep += e; sl += s
    allp = comm.gather((ep, sl), root=0)
    if rank == 0:
        epochs = [e for p in allp for e in p[0]]; slots = [s for p in allp for s in p[1]]
        order = np.argsort(epochs)                         # gather is rank-major; restore time order
        obs = build_obs([epochs[i] for i in order], [slots[i] for i in order])
        tracks = lf.tracks_from_obs(obs)
        INT = lf.compute_INT_variables(tracks, dt_minutes=int(pg["temporalresolution"]))
        gt = sorted({t for tr in tracks.values() for t in tr["times"]})
        os.makedirs(a.out, exist_ok=True)
        LC = lf.build_LC_variables(tracks, gt)
        outf = os.path.join(a.out, f"TOOCAN-{pg['REGION']}-{start:%Y%m%d}-{end:%Y%m%d}.nc")
        lf.write_tracking_nc(outf, INT, LC, gt, region=pg["REGION"], platform=pg.get("GEOplatform", "UNKNOWN"), version=pt["version"])
        logging.info("catalog: %d ranks, %d days, %d DCS, %d obs in %.1fs -> %s", size, len(days), len(tracks),
                     obs["label"].size, time.perf_counter() - t0, outf)


if __name__ == "__main__":
    main()
