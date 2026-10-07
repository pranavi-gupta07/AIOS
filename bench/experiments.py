import contextlib
import json
import os
import threading
import time

import numpy as np

from aios.api import Client
from aios.config import load_config
from aios.engine import Engine
from aios.models import ModelStore
from aios.node import ExecutionNode
from aios.testbed import Testbed

from .workload import Workload, inputs_for, periodic_arrivals, poisson_arrivals, summarize

DEADLINES = {"sensor-mlp": 15.0, "kws-cnn": 30.0, "vision-cnn": 80.0, "batch-encoder": 300.0, "doc-encoder": 1500.0}
SMALL = {"sensor-mlp", "kws-cnn"}


def heuristic(model, stream):
    return {"pin": "local-npu" if model in SMALL else "edge"}


def pinned(target):
    return lambda model, stream: {"pin": target}


def aios(objective):
    return lambda model, stream: {"objective": objective}


class Context:
    def __init__(self, out_dir, quick=False, seed=7):
        self.out_dir = out_dir
        self.quick = quick
        self.seed = seed
        os.makedirs(out_dir, exist_ok=True)
        runtime = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
        self.socket = os.path.join(runtime, f"aios-bench-{os.getpid()}.sock")
        self.overrides = {"daemon": {"socket": self.socket},
                          "targets": {"edge": {"port": 7201}, "cloud": {"port": 7202}}}
        self.cfg = load_config(None, self.overrides)
        self.store = ModelStore()
        self.store.ensure_zoo()
        self.shapes = {k.split(":")[0]: self.store.get(k).input_shape for k in self.store.keys()}
        self.inputs = inputs_for(self.shapes, seed)

    def scale(self, full, quick):
        return quick if self.quick else full

    def save(self, name, obj):
        with open(os.path.join(self.out_dir, name), "w") as fh:
            json.dump(obj, fh, indent=1, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


@contextlib.contextmanager
def system(ctx, policy=None, log_name=None):
    tb = Testbed(ctx.cfg, overrides=ctx.overrides, log_dir=os.path.join(ctx.out_dir, "logs"))
    try:
        tb.start_nodes()
        tb.start_daemon(["--no-kernel"])
        admin = Client(ctx.socket, transport="unix")
        if policy:
            admin.set_policy(**policy)
        time.sleep(0.6)
        yield admin
        admin.close()
    finally:
        tb.stop()
        with contextlib.suppress(OSError):
            os.unlink(ctx.socket)


def busy_snapshot(admin):
    st = admin.status()
    return {name: {"compute_s": (t.get("telemetry") or {}).get("compute_s", 0.0),
                   "slots": (t.get("profile") or {}).get("slots", 1)} for name, t in st["targets"].items()}, st


def utilization(before, after, window):
    out = {}
    for name in after:
        d = after[name]["compute_s"] - before.get(name, {}).get("compute_s", 0.0)
        out[name] = max(0.0, d) / (window * after[name]["slots"])
    return out


def host_cpu():
    with open("/proc/stat") as fh:
        vals = [int(v) for v in fh.readline().split()[1:]]
    return sum(vals), vals[3] + vals[4]


def host_util(a, b):
    dt = b[0] - a[0]
    return 0.0 if dt <= 0 else 1.0 - (b[1] - a[1]) / dt


def e1_streams(rng, duration):
    return [
        {"name": "sensor", "model": "sensor-mlp", "arrivals": poisson_arrivals(12, duration, rng), "deadline_ms": DEADLINES["sensor-mlp"]},
        {"name": "audio", "model": "kws-cnn", "arrivals": poisson_arrivals(6, duration, rng), "deadline_ms": DEADLINES["kws-cnn"]},
        {"name": "camera", "model": "vision-cnn", "arrivals": poisson_arrivals(12, duration, rng), "deadline_ms": DEADLINES["vision-cnn"]},
        {"name": "batch", "model": "batch-encoder", "arrivals": poisson_arrivals(2, duration, rng), "deadline_ms": DEADLINES["batch-encoder"]},
    ]


def e1_steady(ctx):
    policies = {
        "app:local-cpu": pinned("local-cpu"),
        "app:local-npu": pinned("local-npu"),
        "app:edge": pinned("edge"),
        "app:cloud": pinned("cloud"),
        "app:heuristic": heuristic,
        "aios:balanced": aios("balanced"),
        "aios:latency": aios("latency"),
        "aios:energy": aios("energy"),
    }
    warm, dur = ctx.scale(4.0, 2.0), ctx.scale(15.0, 6.0)
    results = {}
    for name, fn in policies.items():
        print(f"[E1] {name}")
        rng = np.random.default_rng(ctx.seed)
        with system(ctx) as admin:
            wl = Workload(ctx.socket, ctx.inputs)
            wl.run(e1_streams(np.random.default_rng(ctx.seed + 100), warm), fn, warm)
            admin.reset_stats()
            b, _ = busy_snapshot(admin)
            h0 = host_cpu()
            records, wall = wl.run(e1_streams(rng, dur), fn, dur)
            h1 = host_cpu()
            a, st = busy_snapshot(admin)
            wl.close()
        summary = summarize(records, wall)
        summary["utilization"] = utilization(b, a, wall)
        summary["host_cpu_util"] = host_util(h0, h1)
        summary["per_model"] = {m: summarize([r for r in records if r["model"] == m], wall)
                                for m in sorted({r["model"] for r in records})}
        summary["daemon_stats"] = st["stats"]
        results[name] = summary
        print(f"   mean {summary['mean_ms']:.1f} ms  p95 {summary['p95_ms']:.1f}  miss {summary['deadline_miss_rate']:.3f}  "
              f"thr {summary['throughput_rps']:.1f}/s  dev {summary['device_mj_per_req']:.1f} mJ/req  fail {summary['failed']}")
    ctx.save("e1_steady.json", results)
    return results


def e2_dynamic(ctx):
    phase = ctx.scale(6.0, 3.0)
    dur = phase * 5
    phases = ["normal", "npu contention", "edge link degraded", "edge offline", "recovered + low battery"]

    def streams(rng, duration):
        return [
            {"name": "camera", "model": "vision-cnn", "session": "cam", "arrivals": periodic_arrivals(15, duration),
             "deadline_ms": DEADLINES["vision-cnn"]},
            {"name": "sensor", "model": "sensor-mlp", "arrivals": poisson_arrivals(10, duration, rng),
             "deadline_ms": DEADLINES["sensor-mlp"]},
        ]

    policies = {"app:local-npu": pinned("local-npu"), "app:edge": pinned("edge"), "app:heuristic": heuristic,
                "aios:balanced": aios("balanced")}
    results = {}
    for name, fn in policies.items():
        print(f"[E2] {name}")
        with system(ctx) as admin:
            wl = Workload(ctx.socket, ctx.inputs)
            wl.run(streams(np.random.default_rng(ctx.seed + 1), 3.0), fn, 3.0)
            admin.reset_stats()
            ctl = Client(ctx.socket, transport="unix")
            lock = threading.Lock()

            def act(**kw):
                with lock:
                    ctl.control(**kw)

            events = [
                (phase * 1, "npu contention", lambda: act(target="local-npu", load=0.7)),
                (phase * 2, "edge link degraded", lambda: act(target="edge", link={"rtt_ms": 60, "bandwidth_mbps": 8})),
                (phase * 3, "edge offline", lambda: act(target="edge", online=False)),
                (phase * 4, "edge online", lambda: act(target="edge", online=True)),
                (phase * 4 + 0.01, "edge link restored", lambda: act(target="edge", link={"rtt_ms": 6, "bandwidth_mbps": 200})),
                (phase * 4 + 0.02, "npu load cleared", lambda: act(target="local-npu", load=0.0)),
                (phase * 4 + 0.03, "low battery", lambda: act(device={"battery_level": 0.12})),
            ]
            records, wall = wl.run(streams(np.random.default_rng(ctx.seed + 2), dur), fn, dur, events=events)
            ctl.close()
            wl.close()
        for r in records:
            r["phase"] = min(4, int(r["t"] // phase))
        per_phase = {phases[i]: summarize([r for r in records if r["phase"] == i], phase) for i in range(5)}
        results[name] = {"overall": summarize(records, wall), "phases": per_phase,
                         "timeline": [{k: r.get(k) for k in ("t", "stream", "latency_ms", "target", "ok", "phase",
                                                             "device_energy_j", "deadline_ms")} for r in records]}
        print("   " + "  ".join(f"{p[:14]}: {per_phase[p]['mean_ms']:.0f}ms/{per_phase[p]['deadline_miss_rate']:.2f}" for p in phases))
    ctx.save("e2_dynamic.json", {"phase_s": phase, "phases": phases, "results": results})
    return results


def e3_migration(ctx):
    trials = ctx.scale(6, 3)
    trigger_s = 0.06
    out = {"trials": [], "trigger_ms": trigger_s * 1000}
    with system(ctx) as admin:
        c = Client(ctx.socket, transport="unix")
        x = ctx.inputs["doc-encoder"]
        for t in ("edge", "cloud", "local-npu", "local-cpu"):
            c.infer("doc-encoder", x, pin=t)
        for mode in ("migration", "no-migration"):
            for disturb in (True, False):
                admin.set_policy(migration=(mode == "migration"))
                for i in range(trials):
                    admin.control("edge", load=0.0)
                    time.sleep(1.2)
                    res = {}

                    def run():
                        t0 = time.perf_counter()
                        r = c.infer("doc-encoder", x, objective="latency")
                        res["ms"] = (time.perf_counter() - t0) * 1000
                        res["meta"] = r.meta

                    th = threading.Thread(target=run)
                    th.start()
                    if disturb:
                        time.sleep(trigger_s)
                        admin.control("edge", load=0.9)
                    th.join()
                    m = res["meta"]
                    rec = {"mode": mode, "disturbed": disturb, "trial": i, "latency_ms": res["ms"],
                           "initial": m["initial_target"], "final": m["target"],
                           "path": [s["target"] for s in m["segments"] if "error" not in s],
                           "migrations": m["migrations"], "segments": m["segments"],
                           "device_energy_j": m["device_energy_j"], "system_energy_j": m["system_energy_j"]}
                    out["trials"].append(rec)
                    mig = m["migrations"][0] if m["migrations"] else None
                    print(f"[E3] {mode:<12} disturbed={disturb!s:<5} #{i}: {res['ms']:.0f} ms path {rec['path']}"
                          + (f" ckpt {mig['checkpoint_ms']:.0f} ms state {mig['state_bytes'] / 1024:.0f} KiB"
                             f" resume-net {mig.get('resume_network_ms', 0):.0f} ms" if mig else ""))
        admin.control("edge", load=0.0)
        c.close()
    ctx.save("e3_migration.json", out)
    return out


def e4_cache(ctx):
    dur = ctx.scale(15.0, 7.0)

    def streams(rng):
        return [
            {"name": "camera", "model": "vision-cnn", "arrivals": poisson_arrivals(6, dur, rng), "deadline_ms": DEADLINES["vision-cnn"]},
            {"name": "batch", "model": "batch-encoder", "arrivals": poisson_arrivals(2, dur, rng), "deadline_ms": DEADLINES["batch-encoder"]},
            {"name": "docs", "model": "doc-encoder", "arrivals": poisson_arrivals(0.6, dur, rng), "deadline_ms": DEADLINES["doc-encoder"]},
            {"name": "audio", "model": "kws-cnn", "arrivals": poisson_arrivals(5, dur, rng), "deadline_ms": DEADLINES["kws-cnn"]},
            {"name": "sensor", "model": "sensor-mlp", "arrivals": poisson_arrivals(5, dur, rng), "deadline_ms": DEADLINES["sensor-mlp"]},
        ]

    modes = {
        "cache-aware + prefetch": {"cache_aware": True, "prefetch": True},
        "cache-aware": {"cache_aware": True, "prefetch": False},
        "cache-oblivious": {"cache_aware": False, "prefetch": False},
    }
    results = {}
    for name, pol in modes.items():
        print(f"[E4] {name}")
        with system(ctx, policy=pol) as admin:
            wl = Workload(ctx.socket, ctx.inputs)
            records, wall = wl.run(streams(np.random.default_rng(ctx.seed + 4)), aios("balanced"), dur)
            st = admin.status()
            wl.close()
        s = summarize(records, wall)
        s["model_pushes"] = st["stats"]["model_pushes"]
        s["model_push_mb"] = st["stats"]["model_push_bytes"] / 2 ** 20
        s["caches"] = {n: (t.get("telemetry") or {}).get("cache") for n, t in st["targets"].items()}
        s["prefetches"] = st.get("model_manager", {}).get("recent_prefetches", [])
        s["model_ms_total"] = sum(r.get("model_ms", 0) for r in records if r.get("ok"))
        s["cold_requests"] = sum(1 for r in records if r.get("ok") and r.get("model_ms", 0) > 0)
        results[name] = s
        print(f"   mean {s['mean_ms']:.1f} ms  p95 {s['p95_ms']:.1f}  miss {s['deadline_miss_rate']:.3f}  pushes {s['model_pushes']}"
              f"  pushed {s['model_push_mb']:.1f} MB  cold requests {s['cold_requests']}")
    ctx.save("e4_cache.json", results)
    return results


def e5_overhead(ctx):
    n = ctx.scale(400, 150)
    out = {}
    prof = next(t for t in ctx.cfg["targets"] if t["name"] == "local-npu")
    node = ExecutionNode(dict(prof), store=ctx.store)
    x = ctx.inputs["sensor-mlp"]
    node.execute("w", "sensor-mlp:1", x)
    direct = []
    for i in range(n):
        t0 = time.perf_counter()
        node.execute(f"d{i}", "sensor-mlp:1", x)
        direct.append((time.perf_counter() - t0) * 1000)
    node.close()
    with system(ctx) as admin:
        c = Client(ctx.socket, transport="unix")
        for mode, kw in (("aios pinned", {"pin": "local-npu"}), ("aios scheduled", {"objective": "latency"})):
            lat = []
            c.infer("sensor-mlp", x, **kw)
            for _ in range(n):
                t0 = time.perf_counter()
                c.infer("sensor-mlp", x, **kw)
                lat.append((time.perf_counter() - t0) * 1000)
            out[mode] = lat
        c.close()
        cfg = load_config(None, ctx.overrides)
        eng = Engine(cfg, store=ctx.store)
        eng.start()
        time.sleep(0.5)
        from aios.scheduler import Requirements
        decide = {}
        for key in ctx.store.keys():
            model = ctx.store.get(key)
            req = Requirements(objective="balanced")
            eng.scheduler.place(model, req)
            ts = []
            for _ in range(n):
                t0 = time.perf_counter()
                eng.scheduler.place(model, req)
                ts.append((time.perf_counter() - t0) * 1e6)
            decide[key] = {"median_us": float(np.median(ts)), "p95_us": float(np.percentile(ts, 95))}
        eng.stop()
    out["direct"] = direct
    summary = {k: {"median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95)), "mean_ms": float(np.mean(v))}
               for k, v in out.items()}
    summary["ipc_overhead_ms"] = summary["aios pinned"]["median_ms"] - summary["direct"]["median_ms"]
    summary["scheduling_overhead_ms"] = summary["aios scheduled"]["median_ms"] - summary["aios pinned"]["median_ms"]
    summary["decision_time"] = decide
    print(f"[E5] direct {summary['direct']['median_ms']:.3f} ms  pinned {summary['aios pinned']['median_ms']:.3f} ms  "
          f"scheduled {summary['aios scheduled']['median_ms']:.3f} ms  decision "
          + ", ".join(f"{k} {v['median_us']:.0f}us" for k, v in decide.items()))
    ctx.save("e5_overhead.json", {"summary": summary, "samples": out})
    return summary


EXPERIMENTS = {"e1": e1_steady, "e2": e2_dynamic, "e3": e3_migration, "e4": e4_cache, "e5": e5_overhead}
