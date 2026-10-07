import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from aios.api import AIOSError, Client


class ClientPool:
    def __init__(self, socket_path):
        self.socket_path = socket_path
        self._local = threading.local()
        self._all = []
        self._lock = threading.Lock()

    def get(self):
        c = getattr(self._local, "c", None)
        if c is None:
            c = self._local.c = Client(self.socket_path, transport="unix")
            with self._lock:
                self._all.append(c)
        return c

    def close(self):
        with self._lock:
            for c in self._all:
                c.close()
            self._all.clear()


def inputs_for(shapes, seed=0):
    rng = np.random.default_rng(seed)
    return {m: rng.random(shape, dtype=np.float32) for m, shape in shapes.items()}


def poisson_arrivals(rate, duration, rng):
    t, out = 0.0, []
    while True:
        t += rng.exponential(1.0 / rate)
        if t >= duration:
            return out
        out.append(t)


def periodic_arrivals(fps, duration, offset=0.0):
    return list(np.arange(offset, duration, 1.0 / fps))


class Workload:
    def __init__(self, socket_path, inputs, workers=64):
        self.pool = ClientPool(socket_path)
        self.inputs = inputs
        self.workers = workers

    def run(self, streams, requirements_fn, duration, events=None, on_tick=None):
        jobs = []
        for s in streams:
            for t in s["arrivals"]:
                jobs.append((t, s["name"], s["model"], s.get("session"), s.get("deadline_ms")))
        jobs.sort(key=lambda j: j[0])
        records = []
        lock = threading.Lock()
        events = sorted(events or [], key=lambda e: e[0])
        t0 = time.perf_counter()

        def fire(job):
            t_sched, stream, model, session, deadline = job
            req = requirements_fn(model, stream)
            if session and "pin" not in req:
                req = dict(req, session=session)
            if deadline is not None and "pin" not in req and "deadline_ms" not in req:
                req = dict(req, deadline_ms=deadline)
            t_start = time.perf_counter()
            rec = {"t": t_sched, "stream": stream, "model": model, "deadline_ms": deadline}
            try:
                r = self.pool.get().infer(model, self.inputs[model], **req)
                m = r.meta
                rec.update(ok=True, target=m["target"], server_ms=m["latency_ms"], queue_ms=m["queue_ms"],
                           compute_ms=m["compute_ms"], network_ms=m["network_ms"], model_ms=m["model_ms"],
                           device_energy_j=m["device_energy_j"], system_energy_j=m["system_energy_j"],
                           migrations=len(m["migrations"]), migration_detail=m["migrations"],
                           predicted_ms=m["predicted_ms"])
            except AIOSError as exc:
                rec.update(ok=False, target=req.get("pin"), error=str(exc)[:200])
            t_end = time.perf_counter()
            rec["latency_ms"] = (t_end - (t0 + t_sched)) * 1000
            rec["client_ms"] = (t_end - t_start) * 1000
            rec["t_done"] = t_end - t0
            with lock:
                records.append(rec)

        def run_events():
            for t_ev, label, fn in events:
                delay = t0 + t_ev - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                try:
                    fn()
                except Exception as exc:
                    print(f"  event {label} failed: {exc}")

        ev_thread = threading.Thread(target=run_events, daemon=True)
        ev_thread.start()
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for job in jobs:
                delay = t0 + job[0] - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                ex.submit(fire, job)
        ev_thread.join(timeout=1.0)
        wall = time.perf_counter() - t0
        return records, wall

    def close(self):
        self.pool.close()


def percentile(vals, q):
    return float(np.percentile(vals, q)) if len(vals) else float("nan")


def summarize(records, duration, deadline_key="deadline_ms"):
    ok = [r for r in records if r.get("ok")]
    lat = [r["latency_ms"] for r in ok]
    misses = sum(1 for r in records if not r.get("ok") or (r.get(deadline_key) and r["latency_ms"] > r[deadline_key]))
    with_deadline = sum(1 for r in records if r.get(deadline_key))
    dev_e = sum(r.get("device_energy_j", 0) for r in ok)
    sys_e = sum(r.get("system_energy_j", 0) for r in ok)
    placements = {}
    for r in ok:
        placements[r["target"]] = placements.get(r["target"], 0) + 1
    return {
        "requests": len(records),
        "completed": len(ok),
        "failed": len(records) - len(ok),
        "throughput_rps": len(ok) / duration if duration else float("nan"),
        "mean_ms": float(np.mean(lat)) if lat else float("nan"),
        "p50_ms": percentile(lat, 50),
        "p95_ms": percentile(lat, 95),
        "p99_ms": percentile(lat, 99),
        "deadline_miss_rate": misses / with_deadline if with_deadline else float("nan"),
        "device_energy_j": dev_e,
        "device_mj_per_req": dev_e / len(ok) * 1000 if ok else float("nan"),
        "system_energy_j": sys_e,
        "system_mj_per_req": sys_e / len(ok) * 1000 if ok else float("nan"),
        "migrations": sum(r.get("migrations", 0) for r in ok),
        "placements": placements,
    }
