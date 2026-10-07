import itertools
import socket
import threading
import time

import numpy as np

from . import protocol
from .node import ExecutionNode


class TargetError(Exception):
    pass


class TargetUnavailable(TargetError):
    pass


class Target:
    remote = False

    def __init__(self, profile):
        self.profile = profile
        self.name = profile["name"]
        self.tier = profile.get("tier", "device")
        self.trusted = profile.get("trusted", True)

    def describe(self):
        keys = ("name", "kind", "tier", "gflops", "slots", "trusted", "memory_mb", "cache_mb", "active_w", "idle_w",
                "rtt_ms", "bandwidth_mbps", "host", "port")
        return {k: self.profile[k] for k in keys if k in self.profile}


class LocalTarget(Target):
    remote = False

    def __init__(self, profile, store):
        super().__init__(profile)
        self.store = store
        self.node = ExecutionNode(profile, store=store)

    def telemetry(self):
        return self.node.telemetry()

    def ping(self):
        return 0.0

    def probe(self, nbytes):
        return 0.0

    def has_model(self, key):
        return self.node.has_model(key)

    def push_model(self, key, blob=None, background=False, **kw):
        _, evicted, seconds = self.node.load_from_store(key)
        return {"evicted": evicted, "seconds": seconds}

    def evict(self, key):
        return self.node.cache.evict(key)

    def execute(self, req_id, key, x, start=0, end=None):
        if not self.node.conditions["online"]:
            raise TargetUnavailable(f"{self.name} offline")
        info, y = self.node.execute(req_id, key, x, start, end)
        info["wall_ms"] = info["queue_ms"] + info["compute_ms"] + info["load_ms"]
        return info, y

    def preempt(self, req_id):
        return self.node.preempt(req_id)

    def control(self, **conditions):
        return self.node.set_conditions(**conditions)

    def link(self, **kw):
        return None

    def close(self):
        self.node.close()


class RemoteTarget(Target):
    remote = True

    def __init__(self, profile, connect_timeout=2.0):
        super().__init__(profile)
        self.host = profile.get("host", "127.0.0.1")
        self.port = profile["port"]
        self.connect_timeout = connect_timeout
        self._pool = []
        self._lock = threading.Lock()
        self._rid = itertools.count(1)

    def _acquire(self):
        with self._lock:
            if self._pool:
                return self._pool.pop()
        try:
            return protocol.connect_tcp(self.host, self.port, timeout=self.connect_timeout)
        except OSError as exc:
            raise TargetUnavailable(f"{self.name} unreachable: {exc}") from exc

    def _release(self, sock):
        with self._lock:
            if len(self._pool) < 32:
                self._pool.append(sock)
                return
        sock.close()

    def call(self, header, arrays=None, timeout=None):
        sock = self._acquire()
        header = dict(header)
        header["rid"] = next(self._rid)
        try:
            sock.settimeout(timeout)
            t0 = time.perf_counter()
            sent = protocol.send(sock, header, arrays)
            reply, out = protocol.recv(sock)
            elapsed = time.perf_counter() - t0
            sock.settimeout(None)
        except (OSError, ConnectionError, socket.timeout) as exc:
            sock.close()
            raise TargetUnavailable(f"{self.name}: {exc}") from exc
        self._release(sock)
        if not reply.get("ok", False):
            raise TargetError(reply.get("error", "unknown error"))
        reply["_sent_bytes"] = sent
        reply["_elapsed"] = elapsed
        return reply, out

    def telemetry(self):
        reply, _ = self.call({"op": "telemetry"}, timeout=3.0)
        return reply["telemetry"]

    def ping(self):
        reply, _ = self.call({"op": "ping"}, timeout=3.0)
        return reply["_elapsed"]

    def probe(self, nbytes):
        reply, _ = self.call({"op": "probe"}, {"pad": np.zeros(nbytes, dtype=np.uint8)}, timeout=10.0)
        return reply["_elapsed"]

    def has_model(self, key):
        reply, _ = self.call({"op": "has_model", "key": key}, timeout=5.0)
        return reply["present"]

    def push_model(self, key, blob, background=False, chunk=256 * 1024, share=0.5, bandwidth_mbps=None):
        t0 = time.perf_counter()
        if not background:
            reply, _ = self.call({"op": "load_model", "key": key}, {"blob": np.frombuffer(blob, dtype=np.uint8)})
            return {"evicted": reply.get("evicted", []), "seconds": time.perf_counter() - t0}
        data = np.frombuffer(blob, dtype=np.uint8)
        bw = (bandwidth_mbps or self.profile.get("bandwidth_mbps", 100)) * 1e6
        reply = {}
        for off in range(0, len(data), chunk):
            part = data[off:off + chunk]
            final = off + chunk >= len(data)
            reply, _ = self.call({"op": "model_chunk", "key": key, "offset": off, "final": final}, {"chunk": part})
            if not final:
                time.sleep(len(part) * 8 / bw * (1 / share - 1))
        return {"evicted": reply.get("evicted", []), "seconds": time.perf_counter() - t0}

    def evict(self, key):
        reply, _ = self.call({"op": "evict", "key": key}, timeout=5.0)
        return reply["evicted"]

    def execute(self, req_id, key, x, start=0, end=None):
        header = {"op": "execute", "req_id": req_id, "key": key, "start": start, "end": end}
        try:
            reply, out = self.call(header, {"x": x})
        except TargetError as exc:
            if "model_missing" in str(exc):
                raise KeyError(key) from exc
            raise
        info = {k: reply[k] for k in ("status", "next_layer", "queue_ms", "compute_ms", "load_ms", "modeled_ms", "flops", "node")}
        info["wall_ms"] = reply["_elapsed"] * 1000
        info["sent_bytes"] = reply["_sent_bytes"]
        info["recv_bytes"] = reply["_wire_bytes"]
        return info, out["y"]

    def preempt(self, req_id):
        reply, _ = self.call({"op": "preempt", "req_id": req_id}, timeout=5.0)
        return reply["preempted"]

    def control(self, **conditions):
        reply, _ = self.call({"op": "control", "admin": True, "conditions": conditions}, timeout=5.0)
        return reply["conditions"]

    def link(self, **kw):
        reply, _ = self.call({"op": "link", "admin": True, **kw}, timeout=5.0)
        return reply["link"]

    def link_state(self):
        reply, _ = self.call({"op": "link_state", "admin": True}, timeout=5.0)
        return reply["link"]

    def close(self):
        with self._lock:
            for s in self._pool:
                s.close()
            self._pool.clear()


def build_targets(cfg, store):
    targets = {}
    for profile in cfg["targets"]:
        if profile.get("enabled", True) is False:
            continue
        if profile.get("kind", "local") == "local":
            targets[profile["name"]] = LocalTarget(profile, store)
        else:
            targets[profile["name"]] = RemoteTarget(profile)
    return targets
