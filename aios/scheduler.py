import threading
from dataclasses import dataclass, field

OBJECTIVES = ("latency", "energy", "balanced")
PRIVACY = ("any", "trusted", "device")


@dataclass
class Requirements:
    deadline_ms: float | None = None
    objective: str | None = None
    privacy: str = "any"
    session: str | None = None
    pin: str | None = None
    migrate: bool = True
    deny: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        r = cls(deadline_ms=d.get("deadline_ms"), objective=d.get("objective"), privacy=d.get("privacy") or "any",
                session=d.get("session"), pin=d.get("pin"), migrate=d.get("migrate", True), deny=list(d.get("deny", [])))
        if r.objective is not None and r.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        if r.privacy not in PRIVACY:
            raise ValueError(f"privacy must be one of {PRIVACY}")
        return r

    def to_dict(self):
        return {"deadline_ms": self.deadline_ms, "objective": self.objective, "privacy": self.privacy,
                "session": self.session, "pin": self.pin, "migrate": self.migrate, "deny": self.deny}


class Scheduler:
    def __init__(self, cfg, monitor, store):
        s = cfg.get("scheduler", {})
        self.cfg = cfg
        self.monitor = monitor
        self.store = store
        self.default_objective = s.get("objective", "balanced")
        self.session_hysteresis = s.get("session_hysteresis", 0.15)
        self.migration_gain = s.get("migration_gain", 0.25)
        self.migration_min_gain_s = s.get("migration_min_gain_ms", 20) / 1000
        self.thermal_lookahead = s.get("thermal_lookahead", 0.85)
        self.weights = {k: dict(v) for k, v in cfg.get("objectives", {}).items()}
        for k, v in {"latency": (1.0, 0.0), "energy": (0.15, 0.85), "balanced": (0.6, 0.4)}.items():
            self.weights.setdefault(k, {"latency": v[0], "energy": v[1]})
        self.use_cache_state = True
        self.use_feedback = True
        self.corrections = {}
        self._lock = threading.Lock()

    def correction(self, target, key):
        if not self.use_feedback:
            return 1.0
        with self._lock:
            return self.corrections.get((target, key), 1.0)

    def observe(self, target, key, raw_predicted_s, actual_s):
        if raw_predicted_s <= 1e-4 or actual_s <= 0:
            return
        ratio = max(0.2, min(20.0, actual_s / raw_predicted_s))
        with self._lock:
            prev = self.corrections.get((target, key))
            self.corrections[(target, key)] = ratio if prev is None else prev * 0.7 + ratio * 0.3

    def weights_for(self, req):
        obj = req.objective or self.default_objective
        w = dict(self.weights[obj])
        if self.monitor.device.low_battery:
            w["energy"] = min(1.0, w["energy"] + 0.35)
            w["latency"] = max(0.0, 1.0 - w["energy"])
        return obj, w

    def allowed(self, target, req):
        if target.name in req.deny:
            return False, "denied by policy"
        if req.privacy == "trusted" and not target.trusted:
            return False, "untrusted target for private request"
        if req.privacy == "device" and target.tier != "device":
            return False, "request restricted to on-device execution"
        return True, ""

    def estimate(self, model, name, start=0, input_bytes=None, running=False, extra_s=0.0, assume_cached=False):
        st = self.monitor.state(name)
        target = st.target
        prof = target.profile
        tel = (target.telemetry() if not target.remote else st.telemetry) or {}
        dev = self.monitor.device
        est = {"target": name, "feasible": True, "reason": ""}
        if not st.online or tel.get("online") is False:
            return {**est, "feasible": False, "reason": "offline"}
        if model.memory_mb > prof.get("memory_mb", 1e9):
            return {**est, "feasible": False, "reason": f"needs {model.memory_mb:.1f} MB > {prof['memory_mb']} MB"}
        if model.nbytes > prof.get("cache_mb", 1e9) * 2 ** 20:
            return {**est, "feasible": False, "reason": "model larger than model cache"}
        gflops = tel.get("effective_gflops", prof["gflops"])
        flops = model.flops_between(start)
        compute = flops / (max(gflops, 1e-3) * 1e9)
        temp, thr = tel.get("temperature_c"), tel.get("throttle_c", prof.get("throttle_c"))
        if temp is not None and thr and not tel.get("throttled") and temp >= self.thermal_lookahead * thr:
            tf = prof.get("throttle_factor", 0.5)
            closeness = min(1.0, (temp - self.thermal_lookahead * thr) / max(1e-6, thr * (1 - self.thermal_lookahead)))
            compute *= 1 + (1 / tf - 1) * closeness * 0.75
        raw_compute = compute
        compute *= self.correction(name, model.key)
        queue = 0.0
        if not running:
            slots = prof.get("slots", 1)
            if target.remote:
                external = max(0, tel.get("active", 0) + tel.get("queued", 0) - st.inflight_at_refresh)
                waiting = st.inflight + external
            else:
                waiting = max(st.inflight, tel.get("active", 0) + tel.get("queued", 0))
            if waiting >= slots:
                svc = (tel.get("service_ms") or compute * 1000) / 1000
                queue = (waiting - slots + 1) * svc / slots
        in_bytes = model.activation_bytes(start) if input_bytes is None else input_bytes
        out_bytes = model.output_bytes
        network = up = down = 0.0
        model_s = 0.0
        cached = True
        if self.use_cache_state and not assume_cached:
            cached = model.key in (tel.get("cache", {}).get("models", [])) or model.key in st.pending_models
        if target.remote:
            bw = max(0.05, st.bandwidth_mbps or prof.get("bandwidth_mbps", 100)) * 1e6
            up = (in_bytes + 512) * 8 / bw
            down = (out_bytes + 512) * 8 / bw
            network = st.rtt_s + up + down + tel.get("link_backlog_ms", 0.0) / 1000
            if not cached and not running:
                model_s = self._blob_size(model) * 8 / bw + st.rtt_s
        elif not cached and not running:
            model_s = self._blob_size(model) / (prof.get("disk_mbps", 400) * 1e6)
        latency = queue + compute + network + model_s + extra_s
        if target.remote:
            device_e = dev.radio_tx_w * (up + (model_s if not cached else 0)) + dev.radio_rx_w * down \
                + dev.radio_idle_w * max(0.0, latency - up - down)
            system_e = device_e + prof.get("active_w", 0) * compute
        else:
            device_e = prof.get("active_w", 1.0) * compute + dev.radio_idle_w * 0.0
            system_e = device_e
        return {**est, "latency_s": latency, "queue_s": queue, "compute_s": compute, "raw_compute_s": raw_compute, "network_s": network,
                "model_s": model_s, "extra_s": extra_s, "cached": cached, "device_energy_j": device_e,
                "system_energy_j": system_e}

    def _blob_size(self, model):
        try:
            return len(self.store.blob(model.key))
        except (KeyError, OSError):
            return model.nbytes

    def checkpoint_cost(self, model, layer, current):
        st = self.monitor.state(current)
        if not st.target.remote:
            return 0.0
        bw = max(0.05, st.bandwidth_mbps or 100) * 1e6
        return st.rtt_s * 1.5 + model.activation_bytes(layer) * 8 / bw

    def rank(self, model, req, start=0, input_bytes=None, exclude=(), current=None, running_current=False,
             assume_cached=False):
        obj, w = self.weights_for(req)
        rows = []
        ckpt = self.checkpoint_cost(model, start, current) if running_current and current else 0.0
        for name, st in self.monitor.states.items():
            if name in exclude:
                continue
            ok, why = self.allowed(st.target, req)
            if not ok:
                rows.append({"target": name, "feasible": False, "reason": why})
                continue
            if running_current and name == current:
                rows.append(self.estimate(model, name, start, input_bytes=0, running=True))
            else:
                rows.append(self.estimate(model, name, start, input_bytes, extra_s=ckpt, assume_cached=assume_cached))
        feasible = [r for r in rows if r["feasible"]]
        if req.deadline_ms is not None:
            meets = [r for r in feasible if r["latency_s"] * 1000 <= req.deadline_ms]
            for r in feasible:
                r["meets_deadline"] = r in meets
            if meets:
                feasible = meets
        if feasible:
            lmin = max(1e-6, min(r["latency_s"] for r in feasible))
            emin = max(1e-6, min(r["device_energy_j"] for r in feasible))
            for r in feasible:
                r["score"] = w["latency"] * r["latency_s"] / lmin + w["energy"] * r["device_energy_j"] / emin
        feasible.sort(key=lambda r: r["score"])
        return obj, w, feasible, rows

    def place(self, model, req, start=0, input_bytes=None, exclude=(), current=None):
        if req.pin:
            st = self.monitor.states.get(req.pin)
            if st is None:
                raise ValueError(f"unknown target {req.pin}")
            est = self.estimate(model, req.pin, start, input_bytes)
            return {"target": req.pin, "reason": "pinned by application", "objective": "pinned", "estimate": est,
                    "candidates": [est]}
        obj, w, feasible, rows = self.rank(model, req, start, input_bytes, exclude)
        if not feasible:
            reasons = {r["target"]: r.get("reason") for r in rows}
            raise RuntimeError(f"no feasible target for {model.key}: {reasons}")
        best = feasible[0]
        reason = "lowest weighted cost"
        if current and current != best["target"]:
            cur = next((r for r in feasible if r["target"] == current), None)
            if cur is not None and cur["score"] <= best["score"] * (1 + self.session_hysteresis):
                best = cur
                reason = "session affinity within hysteresis"
        return {"target": best["target"], "reason": reason, "objective": obj, "weights": w, "estimate": best,
                "candidates": _round_rows(rows)}

    def migration_choice(self, model, req, layer, current):
        obj, w, feasible, rows = self.rank(model, req, layer, current=current, running_current=True)
        if not feasible:
            return None
        cur = next((r for r in feasible if r["target"] == current), None)
        best = feasible[0]
        if best["target"] == current:
            return None
        if cur is None:
            return {"target": best["target"], "reason": f"{current} no longer feasible", "estimate": best, "current": None}
        if cur["score"] < best["score"] * (1 + self.migration_gain):
            return None
        if w["latency"] > 0 and cur["latency_s"] - best["latency_s"] < self.migration_min_gain_s:
            return None
        return {"target": best["target"], "reason": "predicted remaining cost lower elsewhere", "estimate": best, "current": cur}


def _round_rows(rows):
    out = []
    for r in rows:
        out.append({(k[:-2] + "_ms" if k.endswith("_s") else k):
                    (round(v * 1000, 3) if k.endswith("_s") else round(v, 6) if isinstance(v, float) else v)
                    for k, v in r.items()})
    return out
