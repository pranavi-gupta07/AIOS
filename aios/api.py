import itertools
import socket
import threading

import numpy as np

from . import kdev, protocol
from .config import load_config


class AIOSError(Exception):
    pass


class Result:
    def __init__(self, output, meta):
        self.output = output
        self.meta = meta

    @property
    def target(self):
        return self.meta["target"]

    @property
    def latency_ms(self):
        return self.meta["latency_ms"]

    @property
    def migrations(self):
        return self.meta["migrations"]

    def argmax(self):
        return np.argmax(self.output, axis=-1)

    def __repr__(self):
        return f"Result(target={self.target!r}, latency_ms={self.latency_ms}, shape={tuple(self.output.shape)})"


def _socket_path(path):
    if path:
        return path
    return load_config()["daemon"]["socket"]


class Client:
    def __init__(self, socket_path=None, transport="auto", device=kdev.DEFAULT_DEVICE):
        self._lock = threading.Lock()
        self._rid = itertools.count(1)
        self._sock = None
        self._chan = None
        self.transport = None
        if transport in ("auto", "kernel") and kdev.available(device):
            try:
                chan = kdev.KernelChannel(device)
                if chan.stats()["daemon_pid"] > 0:
                    self._chan = chan
                    self.transport = "kernel"
                else:
                    chan.close()
            except OSError:
                pass
        if self._chan is None:
            if transport == "kernel":
                raise AIOSError(f"kernel transport requested but {device} has no registered daemon")
            self.socket_path = _socket_path(socket_path)
            self._connect_socket()
            self.transport = "unix"

    def _connect_socket(self):
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        if ":" in str(self.socket_path) or not hasattr(socket, "AF_UNIX"):
            if ":" in str(self.socket_path):
                host, port = str(self.socket_path).rsplit(":", 1)
                port = int(port)
            else:
                host, port = "127.0.0.1", 9090
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self._sock.connect((host, port))
            except OSError as exc:
                raise AIOSError(f"cannot reach aiosd at {self.socket_path}: {exc}") from exc
        else:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                self._sock.connect(self.socket_path)
            except OSError as exc:
                raise AIOSError(f"cannot reach aiosd at {self.socket_path}: {exc}") from exc

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None
        if self._chan:
            self._chan.close()
            self._chan = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def call(self, header, arrays=None):
        header = dict(header)
        header["rid"] = next(self._rid)
        if self._chan is not None:
            status, blob = self._chan.call(protocol.encode(header, arrays))
            reply, out = protocol.decode(blob)
        else:
            with self._lock:
                try:
                    protocol.send(self._sock, header, arrays)
                    reply, out = protocol.recv(self._sock)
                except (ConnectionError, OSError):
                    self._connect_socket()
                    protocol.send(self._sock, header, arrays)
                    reply, out = protocol.recv(self._sock)
        if not reply.get("ok"):
            raise AIOSError(reply.get("error", "request failed"))
        return reply, out

    def infer(self, model, x, *, deadline_ms=None, objective=None, privacy=None, session=None, pin=None, migrate=True):
        req = {"deadline_ms": deadline_ms, "objective": objective, "privacy": privacy, "session": session,
               "pin": pin, "migrate": migrate}
        reply, out = self.call({"op": "infer", "model": model, "requirements": req},
                               {"x": np.ascontiguousarray(x, dtype=np.float32)})
        return Result(out["y"], reply["meta"])

    def explain(self, model, **requirements):
        return self.call({"op": "explain", "model": model, "requirements": requirements})[0]["decision"]

    def status(self):
        return self.call({"op": "status"})[0]["status"]

    def models(self):
        return self.call({"op": "models"})[0]["models"]

    def register_model(self, model):
        blob = np.frombuffer(model.to_bytes(), dtype=np.uint8)
        return self.call({"op": "register_model"}, {"blob": blob})[0]["key"]

    def control(self, target=None, link=None, device=None, **conditions):
        header = {"op": "control", "target": target, "conditions": conditions or None, "link": link, "device": device}
        return self.call(header)[0]["result"]

    def set_policy(self, **kw):
        return self.call({"op": "policy", **kw})[0]["policy"]

    def evict(self, target, model):
        return self.call({"op": "evict", "target": target, "model": model})[0]["evicted"]

    def reset_stats(self):
        self.call({"op": "reset_stats"})

    def records(self, n=100):
        return self.call({"op": "records", "n": n})[0]["records"]

    def shutdown(self):
        self.call({"op": "shutdown"})


_local = threading.local()


def default_client():
    c = getattr(_local, "client", None)
    if c is None:
        c = _local.client = Client()
    return c


def infer(model, x, **requirements):
    return default_client().infer(model, x, **requirements)
