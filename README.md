# AIOS

A kernel-level AI inference scheduling and migration framework for edge–cloud computing.

Applications ask the operating system for *an inference* — a model and an input, optionally with a deadline,
an objective (latency, energy, balanced) and a privacy level — and never name a device. AIOS watches the
system (CPU/NPU utilization, memory, temperature, battery, network RTT and bandwidth, which models are
cached where), places the request on the local CPU, local NPU, an edge server or the cloud, and moves running
inferences between them when conditions change.

```
 application ──► /dev/aios (kernel broker)  or  unix socket
                        │
                ┌───────▼──────────────────────── aiosd ───────────────────────────┐
                │ Resource Monitor   host /proc + /sys, per-target telemetry,       │
                │                    RTT / bandwidth probes, device battery         │
                │ Inference Scheduler  cost model: queue + compute + network +      │
                │                    model transfer + energy, privacy/memory/       │
                │                    deadline constraints, feedback correction      │
                │ Model Manager      model store, per-target LRU caches,            │
                │                    popularity tracking, cost-aware prefetch       │
                │ Migration Manager  preempt at layer boundary, ship activation,    │
                │                    resume elsewhere; failover on target loss      │
                └───────┬───────────────┬────────────────┬────────────────┬─────────┘
                    local-cpu       local-npu          edge             cloud
```

## Mapping to the proposal

| Proposal item | Where |
|---|---|
| Unified OS-level inference interface (objective 1) | `kernel/aios.c` (`/dev/aios` ioctl broker, records caller pid/uid/comm), `aios/daemon.py`, `aios/api.py` |
| AI-aware scheduling (objective 2, §6.3) | `aios/scheduler.py` |
| System information for placement (objective 3, §6.2) | `aios/monitor.py` |
| Inference migration (objective 4, §6.4) | `aios/engine.py` (migration loop, checkpoint/resume), `aios/node.py` (layer-boundary preemption) |
| Model availability and caching (objective 5, §6.5) | `aios/models.py`, `aios/node.py` (`ModelCache`), `aios/modelmgr.py` |
| Prototype and comparison with application-managed inference (objective 6, §9) | `bench/`, `examples/app_managed.py` vs `examples/app_aios.py` |
| Policy constraints | `privacy` per request, admin rules in `[policy]` matched on uid / process name |

## Layout

```
aios/            daemon, scheduler, monitor, model manager, execution nodes, client API, CLI
kernel/          /dev/aios kernel module, UAPI header, aiosstat tool
config/aios.toml targets, device power model, scheduler weights, prefetch and policy settings
bench/           workload generator, experiments E1–E5, report generator
examples/        AIOS app, the same app managing devices itself, kernel client, adaptation demo
tests/           unit, scheduler and full-system tests
results/         evaluation output (REPORT.md + figures + raw JSON)
```

## Quick start

```
make venv                 # uv venv + editable install with eval/test extras
make test                 # 40 tests, ~20 s
.venv/bin/aiosd --spawn-nodes   # daemon plus emulated edge and cloud nodes
```

In another terminal:

```
.venv/bin/aiosctl status
.venv/bin/aiosctl models
.venv/bin/aiosctl explain vision-cnn --objective latency
.venv/bin/aiosctl infer vision-cnn --session cam --deadline 80 --repeat 20
.venv/bin/aiosctl set local-npu load=0.8          # another app hogs the NPU
.venv/bin/aiosctl link edge rtt_ms=70 bandwidth_mbps=6
.venv/bin/aiosctl set edge online=false
.venv/bin/aiosctl device battery_level=0.1
.venv/bin/aiosctl policy --objective energy --migration on --prefetch on
python examples/demo_adaptation.py
```

From Python:

```python
import aios, numpy as np
with aios.Client() as c:
    r = c.infer("vision-cnn", np.random.rand(1, 3, 64, 64), deadline_ms=80, privacy="trusted")
    print(r.target, r.latency_ms, r.meta["segments"], r.migrations)
```

## Kernel module

`kernel/aios.c` registers the misc device `/dev/aios` and `/proc/aios`. Applications `AIOS_IOC_SUBMIT` a
request and `AIOS_IOC_WAIT` for the answer; the placement daemon `AIOS_IOC_REGISTER`s, blocks in
`AIOS_IOC_FETCH`, and returns results with `AIOS_IOC_COMPLETE`. The kernel stamps every request with the
submitting process's pid, uid and command name, which the daemon uses for admin policy, so an application
cannot lie about its identity. Requests of a process that exits are reclaimed; if the daemon exits, waiting
callers are failed with `-EPIPE`/`-EHOSTDOWN` instead of hanging.

```
make kernel               # builds aios.ko against the running kernel and the aiosstat tool
sudo insmod kernel/aios.ko
.venv/bin/aiosd --spawn-nodes      # picks up /dev/aios automatically
python examples/kernel_client.py
kernel/aiosstat ; cat /proc/aios
```

Without the module the same protocol runs over a unix socket, and clients pick the transport automatically
(`Client(transport="auto")`). The module builds cleanly against Linux 7.1.9 and the Python ioctl layout is
checked against the C header by the test suite; it has not been loaded on this machine because that needs
root.

## Testbed

Devices are emulated on one host so experiments are repeatable:

* every inference really executes (numpy CNN/MLP models with deterministic weights), layer by layer;
* compute time is shaped to the target's GFLOPS, reduced by background load and thermal throttling;
* temperature follows a heating/cooling model driven by utilization and throttles above a threshold;
* edge and cloud are separate processes behind TCP with emulated RTT, jitter and *shared* bandwidth, so
  model pushes and probes contend with inference traffic;
* the daemon measures RTT and bandwidth with probes and passive observation rather than reading the
  configured values;
* end-device energy uses a power model (compute power for local targets, radio tx/rx/idle power for offload)
  because RAPL counters need root; host CPU, memory, thermal zones and battery are read from `/proc` and `/sys`.

Profiles live in `config/aios.toml`. Setting `emulate = false` on a local target runs at the host's real speed,
and `host_coupled = true` feeds real host CPU load and battery into the scheduler.

## Evaluation

```
make eval          # full run, ~8 minutes, writes results/<timestamp>/REPORT.md
make eval-quick    # shorter phases
```

| Experiment | Proposal metric |
|---|---|
| E1 steady mixed workload: 5 application-managed placements vs 3 AIOS objectives | latency, throughput, utilization, energy |
| E2 conditions change every phase (NPU contention, link degradation, edge failure, low battery) | adaptation, static vs AIOS |
| E3 4.8 GFLOP job, edge overloaded mid-run | migration overhead, with vs without migration |
| E4 cold start with skewed model popularity | model availability and caching |
| E5 IPC and decision cost | cost of the abstraction |

The last full run is in `results/full/REPORT.md` (figures and raw JSON alongside). Headlines from it:

* E1: AIOS (13.9–15.5 ms mean, ≤1.1% deadline misses) matches the best hand-tuned application rule
  (13.5 ms, 0%) and beats every single-device placement (17.5–221 ms, up to 85% misses), with no
  application code for device selection.
* E2: when the edge link degrades and then the edge fails, AIOS moves the camera stream to the cloud and
  completes every request; edge-pinned and heuristic applications fail 145 and 91 requests, and the
  NPU-pinned application falls seconds behind once another process loads the NPU.
* E3: with the edge overloaded mid-run, migrating the 768 KiB activation at a layer boundary finishes in
  533 ms against 1514 ms without migration; when nothing changes, migration support costs nothing
  (247 vs 245 ms) and never triggers.
* E4: cost-aware prefetch lowers mean latency to 42.9 ms from 54.5 ms (cache-aware only) and 150 ms
  (cache-oblivious, 115 MB of model traffic).
* E5: the daemon round trip costs about 1 ms and a placement decision 0.3–0.6 ms.

That run shared the host with unrelated CPU-heavy jobs (load average ≈ 9 on 12 cores), which slows the real
compute inside every emulated device; the report records the load.
