import argparse
import logging
import sys
from datetime import datetime

from .driver import Runner


def main(argv=None):
    ap = argparse.ArgumentParser(description="TOOCAN-HPC: exact high-performance TOOCAN segmentation")
    ap.add_argument("toocan_cfg"); ap.add_argument("geo_cfg")
    ap.add_argument("start", help="YYYY-MM-DDTHH:MM"); ap.add_argument("end", help="YYYY-MM-DDTHH:MM")
    ap.add_argument("lonmin", type=float); ap.add_argument("lonmax", type=float)
    ap.add_argument("latmin", type=float); ap.add_argument("latmax", type=float)
    ap.add_argument("--out", required=True, help="output root (pathout_TOOCAN of the original)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--readers", type=int, default=8, help="reader processes (OLR->BT conversion + NetCDF decode)")
    ap.add_argument("--writers", type=int, default=3, help="writer processes (NetCDF compression of finished days)")
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--quant-mode", type=int, default=1, help="1 = float32 multiply (what numba does; verified by validation/test_quantization.py)")
    ap.add_argument("--var", default=None)
    ap.add_argument("--fast-level", type=int, default=2, choices=(0, 1, 2),
                    help="0 literal port (every cold voxel), 1 restricted scan + frontier list, 2 = 1 + deterministic parallel BFS")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    fmt = "%Y-%m-%dT%H:%M"
    Runner(a.toocan_cfg, a.geo_cfg, datetime.strptime(a.start, fmt), datetime.strptime(a.end, fmt),
           (a.lonmin, a.lonmax, a.latmin, a.latmax), a.out, nthreads=a.threads, nread=a.readers, nwrite=a.writers,
           state_dir=a.state_dir, resume=not a.no_resume, quant_mode=a.quant_mode, var=a.var, fast=a.fast_level).run()


if __name__ == "__main__":
    main()
