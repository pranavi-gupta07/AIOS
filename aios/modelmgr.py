import threading
import time
from collections import defaultdict, deque

from .scheduler import Requirements
from .targets import TargetError


class ModelManager:
    def __init__(self, engine, cfg):
        m = cfg.get("models", {})
        self.engine = engine
        self.enabled = m.get("prefetch", True)
        self.min_rate = m.get("prefetch_min_rate", 0.5)
        self.window = m.get("prefetch_window_s", 10.0)
        self.interval = m.get("prefetch_interval_ms", 500) / 1000
        self.margin = m.get("prefetch_margin", 0.1)
        self.horizon = m.get("prefetch_horizon_s", 30.0)
        self.history = defaultdict(deque)
        self.last_req = {}
        self.busy = set()
        self.prefetches = []
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._loop, daemon=True, name="model-manager").start()
        return self

    def stop(self):
        self._stop.set()

    def record(self, key, req):
        now = time.monotonic()
        with self._lock:
            h = self.history[key]
            h.append(now)
            while h and now - h[0] > self.window:
                h.popleft()
            if not req.pin:
                self.last_req[key] = req.to_dict()

    def rates(self):
        now = time.monotonic()
        with self._lock:
            out = {}
            for key, h in self.history.items():
                while h and now - h[0] > self.window:
                    h.popleft()
                out[key] = len(h) / self.window
            return out

    def _loop(self):
        while not self._stop.wait(self.interval):
            if not self.enabled:
                continue
            try:
                self.plan()
            except Exception:
                continue

    def plan(self):
        sched = self.engine.scheduler
        if not sched.use_cache_state:
            return []
        launched = []
        for key, rate in sorted(self.rates().items(), key=lambda kv: -kv[1]):
            if rate < self.min_rate or key not in self.last_req:
                continue
            model = self.engine.store.get(key)
            req = Requirements.from_dict(self.last_req[key])
            req.session = None
            try:
                _, _, warm, _ = sched.rank(model, req, assume_cached=True)
                _, _, cold, _ = sched.rank(model, req)
            except Exception:
                continue
            if not warm or not cold:
                continue
            best = warm[0]
            if best["target"] == cold[0]["target"]:
                continue
            st = self.engine.monitor.state(best["target"])
            tel = st.telemetry or {}
            if key in tel.get("cache", {}).get("models", []) or key in st.pending_models:
                continue
            if not self.worth_it(model, req, rate, best, cold[0]):
                continue
            job = (best["target"], key)
            with self._lock:
                if job in self.busy:
                    continue
                self.busy.add(job)
            threading.Thread(target=self._prefetch, args=job, daemon=True).start()
            launched.append(job)
        return launched

    def worth_it(self, model, req, rate, warm_best, cold_best):
        sched = self.engine.scheduler
        _, w = sched.weights_for(req)
        cold_now = sched.estimate(model, cold_best["target"], assume_cached=True)
        d_lat = cold_now["latency_s"] - warm_best["latency_s"]
        d_e = cold_now["device_energy_j"] - warm_best["device_energy_j"]
        if d_lat < self.margin * cold_now["latency_s"] and d_e < self.margin * cold_now["device_energy_j"]:
            return False
        push = sched.estimate(model, warm_best["target"])
        cost_s = max(1e-6, push["model_s"])
        cost_j = max(1e-9, push["device_energy_j"] - warm_best["device_energy_j"])
        gain_s = rate * self.horizon * d_lat
        gain_j = rate * self.horizon * d_e
        net = w["latency"] * (gain_s - cost_s) / cost_s + w["energy"] * (gain_j - cost_j) / cost_j
        return net > 0

    def _prefetch(self, target, key):
        t0 = time.perf_counter()
        try:
            seconds, nbytes = self.engine.ensure_model(target, self.engine.store.get(key), background=True)
            self.engine.monitor.refresh(target)
            with self._lock:
                self.prefetches.append({"target": target, "model": key, "ms": round((time.perf_counter() - t0) * 1000, 2),
                                        "bytes": nbytes, "t": time.time()})
                self.prefetches = self.prefetches[-100:]
        except (TargetError, OSError, KeyError, MemoryError):
            pass
        finally:
            with self._lock:
                self.busy.discard((target, key))

    def snapshot(self):
        with self._lock:
            recent = list(self.prefetches[-10:])
        return {"enabled": self.enabled, "rates": {k: round(v, 3) for k, v in self.rates().items()},
                "recent_prefetches": recent}
