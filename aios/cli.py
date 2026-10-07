import argparse
import json
import sys

import numpy as np

from . import kdev
from .api import AIOSError, Client


def parse_kv(items):
    out = {}
    for item in items:
        k, _, v = item.partition("=")
        if not _:
            raise SystemExit(f"expected key=value, got {item!r}")
        low = v.lower()
        if low in ("true", "on", "yes"):
            out[k] = True
        elif low in ("false", "off", "no"):
            out[k] = False
        else:
            try:
                out[k] = float(v) if any(c in v for c in ".e") else int(v)
            except ValueError:
                out[k] = v
    return out


def onoff(v):
    return None if v is None else v == "on"


def fmt(v, unit="", nd=1):
    return "-" if v is None else f"{v:.{nd}f}{unit}"


def print_status(st):
    dev = st["device"]
    host = st.get("host") or {}
    print(f"aiosd pid {st['pid']}  uptime {st['uptime_s']}s  objective {st['objective']}  "
          f"migration {'on' if st['migration_enabled'] else 'off'}  transports {st.get('transports')}")
    print(f"device battery {dev['battery_level'] * 100:.2f}%{' (charging)' if dev['charging'] else ''}"
          f"{'  LOW' if dev['low_battery'] else ''}  energy used {dev['energy_used_j']:.3f} J")
    if host:
        th = host.get("thermal", {}).get("max_c")
        bat = host.get("battery", {})
        print(f"host cpu {host.get('cpu_utilization', 0) * 100:.1f}%  mem avail {host.get('memory', {}).get('available_mb')} MB"
              f"  max temp {fmt(th, 'C')}  battery {fmt((bat.get('level') or 0) * 100 if bat.get('level') is not None else None, '%')}"
              f" {bat.get('status') or ''}")
    print()
    print(f"{'target':<10} {'tier':<7} {'online':<6} {'eGFLOPS':>8} {'load':>5} {'temp':>6} {'thr':>4} {'act/q':>6} "
          f"{'util':>5} {'rtt':>7} {'bw':>8} {'cache':>14}  models")
    for name, t in st["targets"].items():
        tel = t.get("telemetry") or {}
        cache = tel.get("cache", {})
        prof = t.get("profile", {})
        print(f"{name:<10} {prof.get('tier', ''):<7} {str(t['online']):<6} {fmt(tel.get('effective_gflops')):>8} "
              f"{fmt(tel.get('load'), '', 2):>5} {fmt(tel.get('temperature_c'), 'C'):>6} {('Y' if tel.get('throttled') else 'n'):>4} "
              f"{tel.get('active', 0)}/{tel.get('queued', 0):<4} {fmt(tel.get('utilization'), '', 2):>5} "
              f"{fmt(t.get('rtt_ms'), 'ms'):>7} {fmt(t.get('bandwidth_mbps'), 'M'):>8} "
              f"{fmt(cache.get('used_mb'))}/{fmt(cache.get('capacity_mb'), '', 0):>6}  {','.join(cache.get('models', []))}")
    s = st["stats"]
    print()
    print(f"requests {s['requests']}  completed {s['completed']}  failed {s['failed']}  migrations {s['migrations']}  "
          f"failovers {s['failovers']}  session switches {s['session_switches']}  model pushes {s['model_pushes']}")
    print(f"placements {s['placements']}")
    if "latency_ms" in s:
        print(f"latency ms {s['latency_ms']}")
    if st.get("active"):
        print(f"active {st['active']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aiosctl", description="control and query the AIOS inference daemon")
    ap.add_argument("--socket", default=None)
    ap.add_argument("--transport", choices=["auto", "unix", "kernel"], default="auto")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("models")
    for name in ("infer", "explain"):
        p = sub.add_parser(name)
        p.add_argument("model")
        p.add_argument("--input", default=None)
        p.add_argument("--objective", choices=["latency", "energy", "balanced"])
        p.add_argument("--deadline", type=float, dest="deadline_ms")
        p.add_argument("--privacy", choices=["any", "trusted", "device"])
        p.add_argument("--session")
        if name == "infer":
            p.add_argument("--pin")
            p.add_argument("--no-migrate", action="store_true")
            p.add_argument("--repeat", type=int, default=1)
    p = sub.add_parser("set")
    p.add_argument("target")
    p.add_argument("conditions", nargs="+")
    p = sub.add_parser("link")
    p.add_argument("target")
    p.add_argument("settings", nargs="+")
    p = sub.add_parser("device")
    p.add_argument("settings", nargs="+")
    p = sub.add_parser("policy")
    p.add_argument("--objective", choices=["latency", "energy", "balanced"])
    p.add_argument("--migration", choices=["on", "off"])
    p.add_argument("--prefetch", choices=["on", "off"])
    p.add_argument("--cache-aware", choices=["on", "off"])
    p.add_argument("--feedback", choices=["on", "off"])
    p = sub.add_parser("evict")
    p.add_argument("target")
    p.add_argument("model")
    p = sub.add_parser("records")
    p.add_argument("-n", type=int, default=20)
    sub.add_parser("reset")
    sub.add_parser("shutdown")
    p = sub.add_parser("kstat")
    p.add_argument("--device", default=kdev.DEFAULT_DEVICE)
    args = ap.parse_args(argv)

    if args.cmd == "kstat":
        if not kdev.available(args.device):
            print(f"{args.device} not present; load kernel/aios.ko first", file=sys.stderr)
            return 1
        with kdev.KernelChannel(args.device) as ch:
            print(json.dumps(ch.stats(), indent=2))
        return 0

    try:
        c = Client(args.socket, transport=args.transport)
    except AIOSError as exc:
        print(f"aiosctl: {exc}", file=sys.stderr)
        return 1
    try:
        return run(c, args)
    except AIOSError as exc:
        print(f"aiosctl: {exc}", file=sys.stderr)
        return 1
    finally:
        c.close()


def run(c, args):
    out = None
    if args.cmd == "status":
        st = c.status()
        if args.json:
            out = st
        else:
            print_status(st)
    elif args.cmd == "models":
        out = c.models()
        if not args.json:
            for m in out:
                print(f"{m['key']:<18} in {str(m['input_shape']):<16} layers {m['layers']:>3}  {m['gflop']:>7.4f} GFLOP  "
                      f"{m['size_mb']:>8.3f} MB  {m['description']}")
            out = None
    elif args.cmd in ("infer", "explain"):
        req = {k: getattr(args, k) for k in ("objective", "deadline_ms", "privacy", "session") if getattr(args, k) is not None}
        if args.cmd == "explain":
            out = c.explain(args.model, **req)
            if not args.json:
                print(f"chosen {out.get('target')} ({out.get('reason')}), objective {out.get('objective')}")
                for r in out.get("candidates", []):
                    if r.get("feasible"):
                        print(f"  {r['target']:<10} score {r.get('score', float('nan')):>8.3f}  latency {r['latency_ms']:>9.2f} ms "
                              f"(q {r['queue_ms']:.1f} c {r['compute_ms']:.1f} n {r['network_ms']:.1f} m {r['model_ms']:.1f})"
                              f"  energy {r['device_energy_j'] * 1000:>8.2f} mJ")
                    else:
                        print(f"  {r['target']:<10} infeasible: {r.get('reason')}")
                out = None
        else:
            shape = next(m["input_shape"] for m in c.models() if m["key"].split(":")[0] == args.model.split(":")[0])
            x = np.load(args.input) if args.input else np.random.default_rng(0).random(shape, dtype=np.float32)
            results = []
            for _ in range(args.repeat):
                r = c.infer(args.model, x, pin=args.pin, migrate=not args.no_migrate, **req)
                results.append(r)
                if not args.json:
                    m = r.meta
                    mig = f"  migrations {[(g['from'], g['to'], g['layer']) for g in m['migrations']]}" if m["migrations"] else ""
                    print(f"{m['model']} -> {m['target']}  {m['latency_ms']:.2f} ms (predicted {m['predicted_ms']}) "
                          f"q {m['queue_ms']:.1f} c {m['compute_ms']:.1f} n {m['network_ms']:.1f} m {m['model_ms']:.1f}  "
                          f"energy {m['device_energy_j'] * 1000:.2f} mJ  top1 {int(r.argmax().ravel()[0])}{mig}")
            if args.json:
                out = [r.meta for r in results]
    elif args.cmd == "set":
        out = c.control(args.target, **parse_kv(args.conditions))
    elif args.cmd == "link":
        out = c.control(args.target, link=parse_kv(args.settings))
    elif args.cmd == "device":
        out = c.control(device=parse_kv(args.settings))
    elif args.cmd == "policy":
        out = c.set_policy(**{k: v for k, v in {"objective": args.objective, "migration": onoff(args.migration),
                                                 "prefetch": onoff(args.prefetch), "cache_aware": onoff(args.cache_aware),
                                                 "feedback": onoff(args.feedback)}.items() if v is not None})
    elif args.cmd == "evict":
        out = {"evicted": c.evict(args.target, args.model)}
    elif args.cmd == "records":
        out = c.records(args.n)
    elif args.cmd == "reset":
        c.reset_stats()
        out = {"ok": True}
    elif args.cmd == "shutdown":
        c.shutdown()
        out = {"ok": True}
    if out is not None:
        print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
