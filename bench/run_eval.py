import argparse
import json
import os
import sys
import time

from aios.config import PROJECT_ROOT

from .experiments import EXPERIMENTS, Context


def main(argv=None):
    ap = argparse.ArgumentParser(prog="run_eval", description="AIOS evaluation: application-managed vs AIOS-managed inference")
    ap.add_argument("--only", default=",".join(EXPERIMENTS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv)
    out = args.out or os.path.join(PROJECT_ROOT, "results", time.strftime("%Y%m%d-%H%M%S"))
    ctx = Context(out, quick=args.quick, seed=args.seed)
    chosen = [e.strip() for e in args.only.split(",") if e.strip()]
    meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "quick": args.quick, "seed": args.seed, "experiments": chosen,
            "config": ctx.cfg, "cpus": os.cpu_count(), "python": sys.version.split()[0],
            "loadavg_before": list(os.getloadavg())}
    t0 = time.time()
    for name in chosen:
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name}")
        EXPERIMENTS[name](ctx)
    meta["duration_s"] = round(time.time() - t0, 1)
    meta["loadavg_after"] = list(os.getloadavg())
    ctx.save("meta.json", meta)
    if not args.no_report:
        from .report import build_report
        path = build_report(out)
        print(f"report: {path}")
    print(f"results in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
