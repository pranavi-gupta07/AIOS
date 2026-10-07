import itertools
import os
import threading
import time
from collections import Counter, deque

import numpy as np

from .config import load_config
from .modelmgr import ModelManager
from .models import ModelStore
from .monitor import DeviceState, HostMonitor, ResourceMonitor
from .scheduler import Requirements, Scheduler, _round_rows
from .targets import TargetError, TargetUnavailable, build_targets


class InferenceError(Exception):
    pass


class Request:
    def __init__(self, rid, model, x, req, client):
        self.id = rid
        self.model = model
        self.x = x
        self.req = req
        self.client = client
        self.created = time.perf_counter()
        self.target = None
        self.start_layer = 0
        self.state = "pending"
        self.migrate_to = None
        self.preempt_at = None
        self.migrations = []
        self.segments = []
        self.lock = threading.Lock()


class Engine:
    def __init__(self, cfg=None, store=None, targets=None):
        self.cfg = cfg or load_config()
        d = self.cfg.get("daemon", {})
        self.store = store or ModelStore()
        self.targets = targets or build_targets(self.cfg, self.store)
        self.host = HostMonitor()
        self.device = DeviceState(self.cfg, self.host)
        self.monitor = ResourceMonitor(self.cfg, self.targets, self.host, self.device)
        self.scheduler = Scheduler(self.cfg, self.monitor, self.store)
        self.models = ModelManager(self, self.cfg)
        self.migration_enabled = True
        self.migration_interval = d.get("migration_check_ms", 25) / 1000
        self.max_migrations = d.get("max_migrations", 3)
        self.max_retries = d.get("max_retries", 2)
        self.rules = self.cfg.get("policy", {}).get("rules", [])
        self.sessions = {}
        self.active = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._model_locks = {}
        self._stop = threading.Event()
        self.started = time.time()
        self.reset_stats()

    def start(self, warm=True):
        self.store.ensure_zoo()
        if warm:
            for name in self.targets:
                self.monitor.refresh(name)
                self.monitor.measure_link(name)
        self.monitor.start()
        self.models.start()
        threading.Thread(target=self._migration_loop, daemon=True, name="migration").start()
        return self

    def stop(self):
        self._stop.set()
        self.models.stop()
        self.monitor.stop()
        for t in self.targets.values():
            t.close()

    def reset_stats(self):
        with self._lock:
            self.stats = {"requests": 0, "completed": 0, "failed": 0, "migrations": 0, "failovers": 0,
                          "session_switches": 0, "model_pushes": 0, "model_push_bytes": 0,
                          "placements": Counter(), "device_energy_j": 0.0, "system_energy_j": 0.0,
                          "since": time.time()}
            self.records = deque(maxlen=20000)
        for t in self.targets.values():
            if not t.remote:
                with t.node._lock:
                    t.node.busy_s = 0.0
                    t.node.started = time.perf_counter()

    def apply_policy(self, req, client):
        for rule in self.rules:
            match = rule.get("match", {})
            if "uid" in match and client.get("uid") != match["uid"]:
                continue
            if "comm" in match and client.get("comm") != match["comm"]:
                continue
            if "privacy" in rule:
                order = ["any", "trusted", "device"]
                if order.index(rule["privacy"]) > order.index(req.privacy):
                    req.privacy = rule["privacy"]
            if "objective" in rule and req.objective is None:
                req.objective = rule["objective"]
            req.deny = list(set(req.deny) | set(rule.get("deny", [])))
            if rule.get("allow_pin", True) is False:
                req.pin = None
        return req

    def _model_lock(self, target, key):
        with self._lock:
            return self._model_locks.setdefault((target, key), threading.Lock())

    def ensure_model(self, name, model, force=False, background=False):
        target = self.targets[name]
        st = self.monitor.state(name)
        tel = st.telemetry or {}
        if not force and model.key in tel.get("cache", {}).get("models", []):
            return 0.0, 0
        with self._model_lock(name, model.key):
            if force:
                st.pending_models.discard(model.key)
            elif model.key in st.pending_models or target.has_model(model.key):
                return 0.0, 0
            blob = self.store.blob(model.key)
            if background and target.remote:
                res = target.push_model(model.key, blob, background=True, bandwidth_mbps=st.bandwidth_mbps)
            else:
                res = target.push_model(model.key, blob)
            st.pending_models.add(model.key)
            if target.remote:
                if not background:
                    self.monitor.observe_transfer(name, len(blob), res["seconds"])
                bw = max(0.05, st.bandwidth_mbps or 100) * 1e6
                self.device.drain(self.device.radio_tx_w * min(res["seconds"], len(blob) * 8 / bw))
            with self._lock:
                self.stats["model_pushes"] += 1
                self.stats["model_push_bytes"] += len(blob) if target.remote else 0
            return res["seconds"], len(blob) if target.remote else 0

    def _decide(self, model, req, start, input_bytes, exclude, current=None):
        if req.session and not req.pin and current is None:
            current = self.sessions.get(req.session)
            if current in exclude:
                current = None
        decision = self.scheduler.place(model, req, start, input_bytes, exclude, current)
        self.monitor.begin(decision["target"])
        if req.session and not req.pin:
            prev = self.sessions.get(req.session)
            if prev is not None and prev != decision["target"]:
                with self._lock:
                    self.stats["session_switches"] += 1
            self.sessions[req.session] = decision["target"]
        return decision

    def infer(self, model_ref, x, requirements=None, client=None):
        client = client or {}
        model = self.store.get(model_ref)
        x = np.asarray(x, dtype=np.float32)
        if tuple(x.shape) != model.input_shape:
            if x.size == int(np.prod(model.input_shape)):
                x = x.reshape(model.input_shape)
            else:
                raise InferenceError(f"input shape {list(x.shape)} does not match {list(model.input_shape)}")
        req = self.apply_policy(Requirements.from_dict(requirements), client)
        r = Request(next(self._ids), model, x, req, client)
        self.models.record(model.key, req)
        with self._lock:
            self.stats["requests"] += 1
            self.active[r.id] = r
        try:
            out, meta = self._run(r)
        except Exception:
            with self._lock:
                self.stats["failed"] += 1
            raise
        finally:
            with self._lock:
                self.active.pop(r.id, None)
        return out, meta

    def _run(self, r):
        model, req = r.model, r.req
        x, start = r.x, 0
        exclude = set()
        retries = 0
        decision = self._decide(model, req, 0, None, exclude)
        first_decision = decision
        model_ms = 0.0
        while True:
            name = decision["target"]
            target = self.targets[name]
            with r.lock:
                r.target, r.start_layer, r.state = name, start, "running"
                r.migrate_to = None
            seg_t0 = time.perf_counter()
            try:
                load_s, _ = self.ensure_model(name, model)
                model_ms += load_s * 1000
                try:
                    info, y = target.execute(f"{os.getpid()}-{r.id}", model.key, x, start)
                except KeyError:
                    load_s, _ = self.ensure_model(name, model, force=True)
                    model_ms += load_s * 1000
                    info, y = target.execute(f"{os.getpid()}-{r.id}", model.key, x, start)
            except (TargetUnavailable, TargetError) as exc:
                self.monitor.end(name)
                if isinstance(exc, TargetUnavailable):
                    self.monitor.mark_offline(name)
                retries += 1
                if req.pin or retries > self.max_retries:
                    raise InferenceError(f"{name} failed: {exc}") from exc
                exclude.add(name)
                with self._lock:
                    self.stats["failovers"] += 1
                r.segments.append({"target": name, "start": start, "end": start, "error": str(exc)})
                decision = self._decide(model, req, start, x.nbytes, exclude)
                continue
            except Exception:
                self.monitor.end(name)
                raise
            self.monitor.end(name)
            seg_wall = (time.perf_counter() - seg_t0) * 1000
            net_ms = max(0.0, info["wall_ms"] - info["queue_ms"] - info["compute_ms"] - info["load_ms"]) if target.remote else 0.0
            if target.remote:
                self.monitor.observe_transfer(name, info.get("sent_bytes", 0) + info.get("recv_bytes", 0), net_ms / 1000)
            if info.get("modeled_ms") and info["flops"]:
                self.scheduler.observe(name, model.key, info["modeled_ms"] / 1000, info["compute_ms"] / 1000)
            dev_e, sys_e = self._account(target, info, net_ms)
            seg = {"target": name, "start": start, "end": info["next_layer"], "queue_ms": round(info["queue_ms"], 3),
                   "compute_ms": round(info["compute_ms"], 3), "network_ms": round(net_ms, 3),
                   "model_ms": round(load_s * 1000 + info["load_ms"], 3), "wall_ms": round(seg_wall, 3),
                   "device_energy_j": round(dev_e, 6), "system_energy_j": round(sys_e, 6)}
            r.segments.append(seg)
            if r.migrations and "resume_network_ms" not in r.migrations[-1]:
                r.migrations[-1]["resume_network_ms"] = round(net_ms, 3)
                r.migrations[-1]["overhead_ms"] = round(r.migrations[-1]["checkpoint_ms"] + net_ms, 3)
            if info["status"] == "done":
                break
            start = info["next_layer"]
            x = y
            now = time.perf_counter()
            dest = r.migrate_to
            mig = {"from": name, "layer": start, "state_bytes": int(x.nbytes),
                   "checkpoint_ms": round((now - (r.preempt_at or now)) * 1000, 3)}
            try:
                if dest and self.monitor.state(dest).online and dest not in exclude:
                    decision = {"target": dest, "reason": "migration", "estimate": self.scheduler.estimate(model, dest, start, x.nbytes)}
                    self.monitor.begin(dest)
                else:
                    decision = self._decide(model, req, start, x.nbytes, exclude)
            except RuntimeError:
                decision = {"target": name, "reason": "no better target", "estimate": self.scheduler.estimate(model, name, start, x.nbytes)}
                self.monitor.begin(name)
            mig["to"] = decision["target"]
            r.migrations.append(mig)
            with self._lock:
                self.stats["migrations"] += 1
            with r.lock:
                r.preempt_at = None
        total_ms = (time.perf_counter() - r.created) * 1000
        meta = self._finish(r, first_decision, total_ms, model_ms)
        return y, meta

    def _account(self, target, info, net_ms):
        dev = self.device
        compute_s = info["compute_ms"] / 1000
        if target.remote:
            st = self.monitor.state(target.name)
            bw = max(0.05, st.bandwidth_mbps or 100) * 1e6
            up = info.get("sent_bytes", 0) * 8 / bw
            down = info.get("recv_bytes", 0) * 8 / bw
            wall = info["wall_ms"] / 1000
            dev_e = dev.radio_tx_w * up + dev.radio_rx_w * down + dev.radio_idle_w * max(0.0, wall - up - down)
            sys_e = dev_e + target.profile.get("active_w", 0) * compute_s
        else:
            dev_e = target.profile.get("active_w", 1.0) * compute_s
            sys_e = dev_e
        dev.drain(dev_e)
        with self._lock:
            self.stats["device_energy_j"] += dev_e
            self.stats["system_energy_j"] += sys_e
        return dev_e, sys_e

    def _finish(self, r, decision, total_ms, model_ms):
        final = r.segments[-1]["target"]
        with self._lock:
            self.stats["completed"] += 1
            self.stats["placements"][final] += 1
        meta = {
            "request_id": r.id,
            "model": r.model.key,
            "target": final,
            "initial_target": decision["target"],
            "placement_reason": decision.get("reason"),
            "objective": decision.get("objective"),
            "latency_ms": round(total_ms, 3),
            "predicted_ms": round(decision["estimate"].get("latency_s", 0) * 1000, 3) if decision["estimate"].get("feasible", True) else None,
            "queue_ms": round(sum(s.get("queue_ms", 0) for s in r.segments), 3),
            "compute_ms": round(sum(s.get("compute_ms", 0) for s in r.segments), 3),
            "network_ms": round(sum(s.get("network_ms", 0) for s in r.segments), 3),
            "model_ms": round(model_ms, 3),
            "device_energy_j": round(sum(s.get("device_energy_j", 0) for s in r.segments), 6),
            "system_energy_j": round(sum(s.get("system_energy_j", 0) for s in r.segments), 6),
            "migrations": r.migrations,
            "segments": r.segments,
            "requirements": r.req.to_dict(),
            "battery_level": round(self.device.level, 5),
        }
        self.records.append({k: meta[k] for k in ("request_id", "model", "target", "latency_ms", "device_energy_j",
                                                  "system_energy_j")} | {"t": time.time(), "migrations": len(r.migrations)})
        return meta

    def _migration_loop(self):
        while not self._stop.wait(self.migration_interval):
            if not self.migration_enabled:
                continue
            with self._lock:
                reqs = list(self.active.values())
            for r in reqs:
                try:
                    self._consider_migration(r)
                except Exception:
                    continue

    def _consider_migration(self, r):
        if not r.req.migrate or r.req.pin or r.state != "running" or r.migrate_to is not None:
            return
        if len(r.migrations) >= self.max_migrations:
            return
        st = self.monitor.state(r.target)
        tel = (st.target.telemetry() if not st.target.remote else st.telemetry) or {}
        run = tel.get("running", {}).get(f"{os.getpid()}-{r.id}")
        if run is None:
            return
        layer = max(run["layer"], r.start_layer)
        if layer >= len(r.model.layers) - 1:
            return
        choice = self.scheduler.migration_choice(r.model, r.req, layer, r.target)
        if choice is None:
            return
        with r.lock:
            if r.migrate_to is not None or r.state != "running":
                return
            r.migrate_to = choice["target"]
            r.preempt_at = time.perf_counter()
        target = self.targets[r.target]
        threading.Thread(target=self._send_preempt, args=(target, f"{os.getpid()}-{r.id}"), daemon=True).start()

    def _send_preempt(self, target, rid):
        try:
            target.preempt(rid)
        except (TargetError, OSError):
            pass

    def explain(self, model_ref, requirements=None, client=None):
        model = self.store.get(model_ref)
        req = self.apply_policy(Requirements.from_dict(requirements), client or {})
        current = self.sessions.get(req.session) if req.session else None
        try:
            decision = self.scheduler.place(model, req, current=current)
        except RuntimeError as exc:
            return {"error": str(exc)}
        decision = dict(decision)
        decision["estimate"] = _round_rows([decision["estimate"]])[0]
        return decision

    def control(self, target=None, conditions=None, link=None, device=None):
        out = {}
        if target:
            t = self.targets[target]
            if conditions:
                out["conditions"] = t.control(**conditions)
            if link:
                out["link"] = t.link(**link)
            self.monitor.refresh(target)
            if link:
                self.monitor.measure_link(target)
        if device:
            self.device.set(**device)
            out["device"] = self.device.snapshot()
        return out

    def evict(self, target, key):
        ok = self.targets[target].evict(self.store.resolve(key))
        self.monitor.refresh(target)
        return ok

    def status(self):
        with self._lock:
            stats = dict(self.stats)
            stats["placements"] = dict(self.stats["placements"])
            active = [{"id": r.id, "model": r.model.key, "target": r.target, "state": r.state,
                       "migrations": len(r.migrations)} for r in self.active.values()]
            sessions = dict(self.sessions)
        lat = [rec["latency_ms"] for rec in self.records]
        if lat:
            stats["latency_ms"] = {"mean": round(float(np.mean(lat)), 3), "p50": round(float(np.percentile(lat, 50)), 3),
                                   "p95": round(float(np.percentile(lat, 95)), 3), "n": len(lat)}
        targets = self.monitor.snapshot()
        for name, t in self.targets.items():
            targets[name]["profile"] = t.describe()
        return {"uptime_s": round(time.time() - self.started, 2), "pid": os.getpid(), "host": self.monitor.host_snapshot,
                "device": self.device.snapshot(), "targets": targets, "stats": stats, "active": active,
                "sessions": sessions, "migration_enabled": self.migration_enabled, "model_manager": self.models.snapshot(),
                "objective": self.scheduler.default_objective,
                "corrections": {f"{k[0]}/{k[1]}": round(v, 4) for k, v in self.scheduler.corrections.items()}}

    def set_policy(self, objective=None, migration=None, cache_aware=None, feedback=None, weights=None, prefetch=None):
        if objective:
            if objective not in self.scheduler.weights:
                raise ValueError(f"unknown objective {objective}")
            self.scheduler.default_objective = objective
        if migration is not None:
            self.migration_enabled = bool(migration)
        if cache_aware is not None:
            self.scheduler.use_cache_state = bool(cache_aware)
        if feedback is not None:
            self.scheduler.use_feedback = bool(feedback)
        if prefetch is not None:
            self.models.enabled = bool(prefetch)
        if weights:
            for k, v in weights.items():
                self.scheduler.weights[k] = dict(v)
        return {"objective": self.scheduler.default_objective, "migration": self.migration_enabled,
                "cache_aware": self.scheduler.use_cache_state, "feedback": self.scheduler.use_feedback,
                "prefetch": self.models.enabled,
                "weights": self.scheduler.weights}
