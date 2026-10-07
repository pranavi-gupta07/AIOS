from ._threads import limit_blas_threads

limit_blas_threads(2)

import argparse
import json
import random
import socket
import socketserver
import sys
import threading
import time
from collections import OrderedDict

from . import protocol
from .config import load_config, target_profile
from .models import Model


class Link:
    def __init__(self, rtt_ms, bandwidth_mbps, jitter_ms=0.0):
        self.rtt_ms = rtt_ms
        self.bandwidth_mbps = bandwidth_mbps
        self.jitter_ms = jitter_ms
        self._lock = threading.Lock()
        self._free_at = 0.0

    def backlog(self):
        with self._lock:
            return max(0.0, self._free_at - time.perf_counter())

    def transmit(self, nbytes):
        with self._lock:
            now = time.perf_counter()
            start = max(now, self._free_at)
            self._free_at = start + nbytes * 8 / (self.bandwidth_mbps * 1e6)
            done = self._free_at
        jitter = random.gauss(0, self.jitter_ms) / 1000 if self.jitter_ms else 0.0
        target = done + max(0.0, self.rtt_ms / 2000 + jitter)
        delay = target - time.perf_counter()
        if delay > 0:
            time.sleep(delay)


class ModelCache:
    def __init__(self, capacity_mb):
        self.capacity = capacity_mb * 2 ** 20
        self.items = OrderedDict()
        self.used = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            model = self.items.get(key)
            if model is None:
                self.misses += 1
                return None
            self.items.move_to_end(key)
            self.hits += 1
            return model

    def put(self, model):
        with self._lock:
            if model.key in self.items:
                self.items.move_to_end(model.key)
                return []
            if model.nbytes > self.capacity:
                raise MemoryError(f"{model.key} ({model.nbytes / 2 ** 20:.1f} MB) exceeds cache capacity")
            evicted = []
            while self.used + model.nbytes > self.capacity and self.items:
                key, old = self.items.popitem(last=False)
                self.used -= old.nbytes
                self.evictions += 1
                evicted.append(key)
            self.items[model.key] = model
            self.used += model.nbytes
            return evicted

    def evict(self, key):
        with self._lock:
            model = self.items.pop(key, None)
            if model is not None:
                self.used -= model.nbytes
            return model is not None

    def keys(self):
        with self._lock:
            return list(self.items)

    def snapshot(self):
        with self._lock:
            return {"models": list(self.items), "used_mb": round(self.used / 2 ** 20, 3),
                    "capacity_mb": round(self.capacity / 2 ** 20, 3), "hits": self.hits,
                    "misses": self.misses, "evictions": self.evictions}


class Thermal:
    def __init__(self, ambient_c, throttle_c, heat_rate, cool_rate, throttle_factor):
        self.ambient = ambient_c
        self.throttle_c = throttle_c
        self.heat_rate = heat_rate
        self.cool_rate = cool_rate
        self.throttle_factor = throttle_factor
        self.temperature = ambient_c
        self.offset = 0.0
        self.throttled = False

    def step(self, dt, utilization):
        target_t = self.temperature + dt * (self.heat_rate * utilization - self.cool_rate * (self.temperature - self.ambient))
        self.temperature = max(self.ambient, target_t)
        t = self.temperature + self.offset
        if t >= self.throttle_c:
            self.throttled = True
        elif t < self.throttle_c - 5:
            self.throttled = False

    @property
    def reading(self):
        return self.temperature + self.offset

    @property
    def factor(self):
        return self.throttle_factor if self.throttled else 1.0


class Preempted(Exception):
    pass


class ExecutionNode:
    def __init__(self, profile, store=None):
        self.profile = profile
        self.name = profile["name"]
        self.store = store
        self.cache = ModelCache(profile["cache_mb"])
        self.slots = threading.Semaphore(profile["slots"])
        self.thermal = Thermal(profile.get("ambient_c", 40.0), profile.get("throttle_c", 85.0),
                               profile.get("heat_rate", 1.0), profile.get("cool_rate", 0.05),
                               profile.get("throttle_factor", 0.5))
        self.conditions = {"load": 0.0, "online": True, "gflops_scale": 1.0}
        self.emulate = profile.get("emulate", True)
        self._lock = threading.Lock()
        self.active = 0
        self.queued = 0
        self.running = {}
        self.completed = 0
        self.preempted = 0
        self.busy_s = 0.0
        self.compute_s = 0.0
        self.flops_done = 0
        self.service_ewma = None
        self.util_ewma = 0.0
        self.started = time.perf_counter()
        self._last_tick = self.started
        self._stop = threading.Event()
        self._ticker = threading.Thread(target=self._tick_loop, daemon=True)
        self._ticker.start()

    def close(self):
        self._stop.set()

    def _tick_loop(self):
        while not self._stop.wait(0.05):
            self.tick()

    def tick(self):
        with self._lock:
            now = time.perf_counter()
            dt = now - self._last_tick
            self._last_tick = now
            util = min(1.0, self.active / self.profile["slots"] + self.conditions["load"])
            self.busy_s += dt * min(1.0, self.active / self.profile["slots"])
            self.util_ewma += (util - self.util_ewma) * min(1.0, dt / 1.0)
            self.thermal.step(dt, util)

    @property
    def effective_gflops(self):
        g = self.profile["gflops"] * self.conditions["gflops_scale"]
        return max(1e-3, g * (1.0 - min(0.95, self.conditions["load"])) * self.thermal.factor)

    def set_conditions(self, **cond):
        with self._lock:
            for k, v in cond.items():
                if k == "temperature_offset":
                    self.thermal.offset = float(v)
                elif k == "temperature":
                    self.thermal.temperature = float(v)
                elif k in ("load", "gflops_scale"):
                    self.conditions[k] = float(v)
                elif k == "online":
                    self.conditions["online"] = bool(v)
            return dict(self.conditions)

    def has_model(self, key):
        return key in self.cache.keys()

    def load_model_blob(self, blob):
        model = Model.from_bytes(blob)
        evicted = self.cache.put(model)
        return model.key, evicted

    def load_from_store(self, key):
        t0 = time.perf_counter()
        blob = self.store.blob(key)
        model = Model.from_bytes(blob)
        if self.emulate:
            delay = len(blob) / (self.profile.get("disk_mbps", 400) * 1e6) - (time.perf_counter() - t0)
            if delay > 0:
                time.sleep(delay)
        evicted = self.cache.put(model)
        return model, evicted, time.perf_counter() - t0

    def preempt(self, req_id):
        with self._lock:
            run = self.running.get(req_id)
            if run is None:
                return False
            run["stop"].set()
            return True

    def execute(self, req_id, key, x, start=0, end=None):
        model = self.cache.get(key)
        load_s = 0.0
        if model is None:
            if self.store is None:
                raise KeyError(f"model {key} not cached on {self.name}")
            model, _, load_s = self.load_from_store(key)
        end = len(model.layers) if end is None else end
        stop = threading.Event()
        with self._lock:
            self.queued += 1
            self.running[req_id] = {"model": key, "layer": start, "total": end, "stop": stop, "state": "queued"}
        t_q = time.perf_counter()
        self.slots.acquire()
        t_start = time.perf_counter()
        with self._lock:
            self.queued -= 1
            self.active += 1
            self.running[req_id]["state"] = "running"
        budget = [t_start]
        modeled = [0.0]

        def on_layer(i, _x):
            nominal = model.layer_flops[i - 1] / (self.effective_gflops * 1e9)
            modeled[0] += nominal
            if self.emulate:
                budget[0] += nominal
                delay = budget[0] - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    budget[0] = time.perf_counter()
            with self._lock:
                self.running[req_id]["layer"] = i

        try:
            out, next_layer = model.run(x, start, end, should_stop=stop.is_set, on_layer=on_layer)
        finally:
            t_end = time.perf_counter()
            self.slots.release()
            with self._lock:
                self.active -= 1
                self.running.pop(req_id, None)
                self.compute_s += t_end - t_start
                svc = t_end - t_start
                self.service_ewma = svc if self.service_ewma is None else self.service_ewma * 0.8 + svc * 0.2
        done_flops = model.flops_between(start, next_layer)
        with self._lock:
            self.flops_done += done_flops
            if next_layer >= end:
                self.completed += 1
            else:
                self.preempted += 1
        return {"status": "done" if next_layer >= end else "preempted", "next_layer": next_layer,
                "queue_ms": (t_start - t_q) * 1000, "compute_ms": (t_end - t_start) * 1000,
                "load_ms": load_s * 1000, "modeled_ms": modeled[0] * 1000, "flops": done_flops,
                "node": self.name}, out

    def telemetry(self):
        self.tick()
        with self._lock:
            uptime = time.perf_counter() - self.started
            return {
                "name": self.name,
                "online": self.conditions["online"],
                "gflops": self.profile["gflops"],
                "effective_gflops": round(self.effective_gflops, 4),
                "load": self.conditions["load"],
                "temperature_c": round(self.thermal.reading, 2),
                "throttled": self.thermal.throttled,
                "throttle_c": self.thermal.throttle_c,
                "slots": self.profile["slots"],
                "active": self.active,
                "queued": self.queued,
                "utilization": round(self.util_ewma, 4),
                "busy_fraction": round(self.busy_s / uptime, 4) if uptime else 0.0,
                "memory_mb": self.profile["memory_mb"],
                "cache": self.cache.snapshot(),
                "running": {k: {"model": v["model"], "layer": v["layer"], "total": v["total"], "state": v["state"]}
                            for k, v in self.running.items()},
                "completed": self.completed,
                "preempted": self.preempted,
                "compute_s": round(self.compute_s, 4),
                "gflop_done": round(self.flops_done / 1e9, 4),
                "service_ms": round(self.service_ewma * 1000, 3) if self.service_ewma else None,
            }


class NodeHandler(socketserver.BaseRequestHandler):
    def setup(self):
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def handle(self):
        server = self.server
        node = server.node
        while True:
            try:
                header, arrays = protocol.recv(self.request)
            except (ConnectionError, OSError):
                return
            admin = header.get("admin", False)
            if not admin:
                if not node.conditions["online"]:
                    self.request.close()
                    return
                server.uplink.transmit(header["_wire_bytes"])
            reply, out = self.dispatch(node, header, arrays)
            reply["rid"] = header.get("rid")
            blob = protocol.encode(reply, out)
            if not admin:
                if not node.conditions["online"]:
                    self.request.close()
                    return
                server.downlink.transmit(len(blob))
            try:
                self.request.sendall(blob)
            except OSError:
                return

    def dispatch(self, node, header, arrays):
        op = header.get("op")
        try:
            if op == "ping":
                return {"ok": True, "t": time.time()}, None
            if op == "probe":
                return {"ok": True, "bytes": header["_wire_bytes"]}, None
            if op == "telemetry":
                tel = node.telemetry()
                tel["link_backlog_ms"] = round(max(self.server.uplink.backlog(), self.server.downlink.backlog()) * 1000, 3)
                return {"ok": True, "telemetry": tel}, None
            if op == "has_model":
                return {"ok": True, "present": node.has_model(header["key"])}, None
            if op == "load_model":
                key, evicted = node.load_model_blob(arrays["blob"].tobytes())
                return {"ok": True, "key": key, "evicted": evicted}, None
            if op == "model_chunk":
                buf = self.server.partial.setdefault(header["key"], bytearray())
                if header["offset"] != len(buf):
                    self.server.partial.pop(header["key"], None)
                    return {"ok": False, "error": "chunk out of order"}, None
                buf.extend(arrays["chunk"].tobytes())
                if not header.get("final"):
                    return {"ok": True, "received": len(buf)}, None
                blob = bytes(self.server.partial.pop(header["key"]))
                key, evicted = node.load_model_blob(blob)
                return {"ok": True, "key": key, "evicted": evicted}, None
            if op == "evict":
                return {"ok": True, "evicted": node.cache.evict(header["key"])}, None
            if op == "execute":
                if not node.has_model(header["key"]):
                    return {"ok": False, "error": "model_missing", "key": header["key"]}, None
                info, out = node.execute(header["req_id"], header["key"], arrays["x"], header.get("start", 0), header.get("end"))
                return {"ok": True, **info}, {"y": out}
            if op == "preempt":
                return {"ok": True, "preempted": node.preempt(header["req_id"])}, None
            if op == "control":
                return {"ok": True, "conditions": node.set_conditions(**header.get("conditions", {}))}, None
            if op == "link":
                for k in ("rtt_ms", "bandwidth_mbps", "jitter_ms"):
                    if k in header:
                        setattr(self.server.uplink, k, float(header[k]))
                        setattr(self.server.downlink, k, float(header[k]))
                return {"ok": True, "link": self.server.link_state()}, None
            if op == "link_state":
                return {"ok": True, "link": self.server.link_state()}, None
            return {"ok": False, "error": f"unknown op {op}"}, None
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, None


class NodeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, profile, host="127.0.0.1", port=0):
        self.node = ExecutionNode(profile)
        self.uplink = Link(profile.get("rtt_ms", 0), profile.get("bandwidth_mbps", 1000), profile.get("jitter_ms", 0))
        self.downlink = Link(profile.get("rtt_ms", 0), profile.get("bandwidth_mbps", 1000), profile.get("jitter_ms", 0))
        self.partial = {}
        super().__init__((host, port), NodeHandler)

    def link_state(self):
        return {"rtt_ms": self.uplink.rtt_ms, "bandwidth_mbps": self.uplink.bandwidth_mbps, "jitter_ms": self.uplink.jitter_ms}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aios-node")
    ap.add_argument("--config", default=None)
    ap.add_argument("--target", required=True)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--overrides", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, json.loads(args.overrides) if args.overrides else None)
    profile = target_profile(cfg, args.target)
    host = args.host or profile.get("host", "127.0.0.1")
    port = args.port if args.port is not None else profile["port"]
    server = NodeServer(profile, host, port)
    print(f"aios-node {profile['name']} listening on {host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.node.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
