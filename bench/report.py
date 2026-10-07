import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aios.config import PROJECT_ROOT

COLORS = {"local-cpu": "#8c6d31", "local-npu": "#2b8a3e", "edge": "#e67700", "cloud": "#1c7ed6", None: "#c92a2a"}
APP = "#adb5bd"
AIOS = "#5f3dc4"


def load(out, name):
    p = os.path.join(out, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def f(v, nd=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.{nd}f}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def bar_colors(names):
    return [AIOS if n.startswith("aios") else APP for n in names]


def plot_e1(e1, out):
    names = list(e1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    mean = [e1[n]["mean_ms"] for n in names]
    p95 = [e1[n]["p95_ms"] for n in names]
    x = np.arange(len(names))
    axes[0].bar(x, mean, color=bar_colors(names), label="mean")
    axes[0].scatter(x, p95, color="black", marker="_", s=300, label="p95", zorder=3)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("latency (ms, log)")
    axes[0].set_title("Inference latency")
    axes[0].legend()
    axes[1].bar(x, [e1[n]["device_mj_per_req"] for n in names], color=bar_colors(names))
    axes[1].set_ylabel("device energy per request (mJ)")
    axes[1].set_title("End-device energy")
    axes[2].bar(x, [e1[n]["deadline_miss_rate"] * 100 for n in names], color=bar_colors(names))
    axes[2].set_ylabel("deadline misses (%)")
    axes[2].set_title("Deadline misses")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out, "e1_steady.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bottom = np.zeros(len(names))
    for t in ("local-cpu", "local-npu", "edge", "cloud"):
        v = np.array([e1[n]["placements"].get(t, 0) / max(1, e1[n]["completed"]) * 100 for n in names])
        ax.bar(x, v, bottom=bottom, color=COLORS[t], label=t)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("share of requests (%)")
    ax.set_title("Where inference ran")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.45))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "e1_placement.png"), dpi=130)
    plt.close(fig)
    return path


def plot_e2(e2, out):
    res = e2["results"]
    phase = e2["phase_s"]
    names = list(res)
    top = max([4000.0] + [r["latency_ms"] * 1.5 for n in names for r in res[n]["timeline"] if r["ok"]])
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 2.3 * len(names)), sharex=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        tl = [r for r in res[name]["timeline"] if r["stream"] == "camera"]
        for tgt in ("local-cpu", "local-npu", "edge", "cloud", None):
            pts = [r for r in tl if (r["target"] if r["ok"] else None) == tgt and (r["ok"] or tgt is None)]
            if not pts:
                continue
            ys = [r["latency_ms"] if r["ok"] else top / 1.6 for r in pts]
            ax.scatter([r["t"] for r in pts], ys, s=6, color=COLORS[tgt], label=tgt or "failed")
        ax.axhline(80, color="black", lw=0.8, ls="--")
        for i in range(1, 5):
            ax.axvline(i * phase, color="grey", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(5, top)
        ax.set_ylabel("ms")
        ax.set_title(name, fontsize=9, loc="left")
        ax.legend(fontsize=7, loc="upper left", ncol=5, markerscale=2)
    for i, p in enumerate(e2["phases"]):
        np.atleast_1d(axes)[0].text((i + 0.5) * phase, top / 2.2, p, ha="center", fontsize=7)
    np.atleast_1d(axes)[-1].set_xlabel("time (s)  camera stream, 15 fps, deadline 80 ms (dashed)")
    fig.tight_layout()
    path = os.path.join(out, "e2_dynamic.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_e3(e3, out):
    groups = {}
    for t in e3["trials"]:
        groups.setdefault((t["mode"], t["disturbed"]), []).append(t["latency_ms"])
    keys = [("no-migration", False), ("migration", False), ("no-migration", True), ("migration", True)]
    labels = ["static\nundisturbed", "migration\nundisturbed", "static\nedge overloaded", "migration\nedge overloaded"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    vals = [groups.get(k, [np.nan]) for k in keys]
    ax.bar(range(4), [np.mean(v) for v in vals], yerr=[np.std(v) for v in vals],
           color=[APP, AIOS, APP, AIOS], capsize=4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("completion time (ms)")
    ax.set_title("doc-encoder (4.8 GFLOP) with mid-run edge overload")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out, "e3_migration.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_e4(e4, out):
    names = list(e4)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(names))
    axes[0].bar(x, [e4[n]["mean_ms"] for n in names], color=AIOS)
    axes[0].scatter(x, [e4[n]["p95_ms"] for n in names], color="black", marker="_", s=300, zorder=3)
    axes[0].set_ylabel("latency ms (bar mean, tick p95)")
    axes[1].bar(x, [e4[n]["model_push_mb"] for n in names], color=APP)
    axes[1].set_ylabel("model data pushed to remote (MB)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out, "e4_cache.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def loc(path):
    try:
        with open(path) as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return None


def build_report(out):
    meta = load(out, "meta.json") or {}
    e1, e2, e3, e4, e5 = (load(out, n) for n in ("e1_steady.json", "e2_dynamic.json", "e3_migration.json",
                                                 "e4_cache.json", "e5_overhead.json"))
    md = ["# AIOS evaluation results", ""]
    md.append(f"Run started {meta.get('started', '?')}, duration {meta.get('duration_s', '?')} s, "
              f"{'quick' if meta.get('quick') else 'full'} mode, seed {meta.get('seed')}, {meta.get('cpus')} host CPUs, "
              f"Python {meta.get('python')}.")
    loads = [v for k in ("loadavg_before", "loadavg_after") for v in (meta.get(k) or [])[:1]]
    if loads:
        busy = max(loads) > (meta.get("cpus") or 1) / 2
        md.append("")
        md.append(f"Host 1-minute load average {' -> '.join(f'{v:.1f}' for v in loads)}"
                  + (" — the host was shared with other heavy workloads during this run, which slows the real "
                     "numpy compute inside every emulated device (most visible on the cloud profile, whose nominal "
                     "speed is above what this host delivers)." if busy else "."))
    md.append("")
    md.append("All placements go through the same daemon and execution path; application-managed baselines pin the "
              "target (`pin=`) the way an application would hard-code a device, AIOS-managed runs let the OS layer decide. "
              "Devices are emulated on one host: every inference really executes (numpy), compute time is shaped to the "
              "device profile, links are shaped for RTT and shared bandwidth, and thermal state evolves with utilization. "
              "Energy is estimated from the device power model since RAPL counters are not readable without root.")
    md.append("")
    cfg = meta.get("config", {})
    if cfg.get("targets"):
        md.append("## Testbed")
        md.append("")
        md.append(table(["target", "tier", "GFLOPS", "slots", "trusted", "memory MB", "cache MB", "active W", "RTT ms", "Mbit/s"],
                        [[t["name"], t.get("tier"), t["gflops"], t["slots"], t.get("trusted"), t["memory_mb"], t["cache_mb"],
                          t.get("active_w"), t.get("rtt_ms", "-"), t.get("bandwidth_mbps", "-")] for t in cfg["targets"]]))
        md.append("")
    if e1:
        plot_e1(e1, out)
        md.append("## E1  Latency, throughput, utilization, energy under a steady mixed workload")
        md.append("")
        md.append("Open-loop Poisson load: sensor-mlp 12/s (15 ms deadline), kws-cnn 6/s (30 ms), vision-cnn 12/s (80 ms), "
                  "batch-encoder 2/s (300 ms). Warm-up excluded.")
        md.append("")
        rows = []
        for n, s in e1.items():
            util = ", ".join(f"{k} {v * 100:.0f}%" for k, v in s.get("utilization", {}).items() if v >= 0.005)
            rows.append([n, f(s["mean_ms"]), f(s["p50_ms"]), f(s["p95_ms"]), f(s["p99_ms"]),
                         f(s["deadline_miss_rate"] * 100), f(s["throughput_rps"]), f(s["device_mj_per_req"], 2),
                         f(s["system_mj_per_req"], 1), s["failed"], util or "-", f(s.get("host_cpu_util", 0) * 100)])
        md.append(table(["policy", "mean ms", "p50", "p95", "p99", "miss %", "req/s", "device mJ/req", "system mJ/req",
                         "failed", "target utilization", "host CPU %"], rows))
        md.append("")
        md.append("![E1](e1_steady.png)")
        md.append("")
        md.append("![E1 placement](e1_placement.png)")
        md.append("")
    if e2:
        plot_e2(e2, out)
        md.append("## E2  Adaptation to changing system conditions")
        md.append("")
        md.append(f"Camera stream (vision-cnn, 15 fps, session-sticky, 80 ms deadline) plus sensor-mlp at 10/s. "
                  f"Every {e2['phase_s']:.0f} s the system changes: " + "; ".join(e2["phases"]) + ".")
        md.append("")
        rows = []
        for n, r in e2["results"].items():
            row = [n]
            for p in e2["phases"]:
                s = r["phases"][p]
                row.append(f"{f(s['mean_ms'], 0)} / {f(s['deadline_miss_rate'] * 100, 0)}%")
            o = r["overall"]
            row += [f(o["mean_ms"]), f(o["deadline_miss_rate"] * 100), o["failed"], f(o["device_mj_per_req"], 2)]
            rows.append(row)
        md.append(table(["policy"] + [f"{p} (mean ms / miss)" for p in e2["phases"]] +
                        ["overall mean ms", "overall miss %", "failed", "device mJ/req"], rows))
        md.append("")
        md.append("![E2](e2_dynamic.png)")
        md.append("")
    if e3:
        plot_e3(e3, out)
        md.append("## E3  Inference migration and its overhead")
        md.append("")
        md.append(f"doc-encoder (16 layers, 4.8 GFLOP, 768 KiB activation state). In disturbed trials the edge is driven "
                  f"to 90% background load {e3['trigger_ms']:.0f} ms after submission.")
        md.append("")
        rows = []
        groups = {}
        for t in e3["trials"]:
            groups.setdefault((t["mode"], t["disturbed"]), []).append(t)
        for (mode, dist), ts in groups.items():
            lat = [t["latency_ms"] for t in ts]
            migs = [m for t in ts for m in t["migrations"]]
            paths = sorted({" > ".join(t["path"]) for t in ts})
            rows.append([mode, "yes" if dist else "no", len(ts), f(np.mean(lat)), f(np.std(lat)), len(migs),
                         f(np.mean([m["checkpoint_ms"] for m in migs])) if migs else "-",
                         f(np.mean([m["state_bytes"] for m in migs]) / 1024, 0) if migs else "-",
                         f(np.mean([m.get("resume_network_ms", 0) for m in migs])) if migs else "-",
                         "; ".join(paths)])
        md.append(table(["mode", "edge overloaded", "trials", "mean ms", "std", "migrations", "checkpoint ms",
                         "state KiB", "resume transfer ms", "paths"], rows))
        md.append("")
        md.append("Checkpoint time runs from the preempt signal to the activation arriving at the daemon and includes "
                  "finishing the layer in progress on the overloaded node. Resume transfer is the network time of the "
                  "first segment on the new target.")
        md.append("")
        md.append("![E3](e3_migration.png)")
        md.append("")
    if e4:
        plot_e4(e4, out)
        md.append("## E4  Model availability and caching")
        md.append("")
        md.append("Cold start, five models with skewed popularity, including two 36 MB encoders that do not fit the NPU "
                  "cache together.")
        md.append("")
        rows = [[n, f(s["mean_ms"]), f(s["p95_ms"]), f(s["deadline_miss_rate"] * 100), s["model_pushes"],
                 f(s["model_push_mb"]), s["cold_requests"], f(s["model_ms_total"], 0), len(s.get("prefetches", []))]
                for n, s in e4.items()]
        md.append(table(["mode", "mean ms", "p95", "miss %", "model loads", "MB pushed", "requests that waited for a model",
                         "total model wait ms", "prefetches"], rows))
        md.append("")
        md.append("![E4](e4_cache.png)")
        md.append("")
    if e5:
        s = e5["summary"]
        md.append("## E5  Cost of the OS-level abstraction")
        md.append("")
        md.append(table(["path", "median ms", "p95 ms"],
                        [[k, f(v["median_ms"], 3), f(v["p95_ms"], 3)] for k, v in s.items() if isinstance(v, dict) and "median_ms" in v]))
        md.append("")
        md.append(f"IPC and request handling add {s['ipc_overhead_ms']:.3f} ms over calling the executor in-process; "
                  f"letting AIOS choose the target adds a further {s['scheduling_overhead_ms']:.3f} ms.")
        md.append("")
        md.append(table(["model", "placement decision median us", "p95 us"],
                        [[k, f(v["median_us"], 0), f(v["p95_us"], 0)] for k, v in s["decision_time"].items()]))
        md.append("")
    app_loc = loc(os.path.join(PROJECT_ROOT, "examples", "app_managed.py"))
    aios_loc = loc(os.path.join(PROJECT_ROOT, "examples", "app_aios.py"))
    if app_loc and aios_loc:
        md.append("## Application-side resource-management code")
        md.append("")
        md.append(f"`examples/app_managed.py` (application probes devices, ships models, picks a target and falls back on "
                  f"failure) has {app_loc} non-blank lines; `examples/app_aios.py` doing the same job through AIOS has "
                  f"{aios_loc}.")
        md.append("")
    path = os.path.join(out, "REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(md))
    return path


if __name__ == "__main__":
    print(build_report(sys.argv[1]))
