from ._threads import limit_blas_threads

limit_blas_threads(4)

import argparse
import json
import logging
import os
import signal
import socket
import socketserver
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import kdev, protocol
from .config import load_config
from .engine import Engine, InferenceError
from .models import Model
from .testbed import Testbed

log = logging.getLogger("aiosd")


def _comm(pid):
    try:
        with open(f"/proc/{pid}/comm") as fh:
            return fh.read().strip()
    except OSError:
        return None


def peer_credentials(sock):
    if hasattr(socket, "SO_PEERCRED"):
        try:
            raw = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
            pid, uid, gid = struct.unpack("3i", raw)
            return {"pid": pid, "uid": uid, "gid": gid, "comm": _comm(pid), "transport": "unix"}
        except OSError:
            pass
    return {"transport": "unix"}


def dispatch(engine, header, arrays, client, server=None):
    op = header.get("op")
    try:
        if op == "ping":
            return {"ok": True, "pid": os.getpid()}, None
        if op == "infer":
            if "x" not in arrays:
                raise InferenceError("missing input array 'x'")
            y, meta = engine.infer(header["model"], arrays["x"], header.get("requirements"), client)
            return {"ok": True, "meta": meta}, {"y": y}
        if op == "explain":
            return {"ok": True, "decision": engine.explain(header["model"], header.get("requirements"), client)}, None
        if op == "status":
            st = engine.status()
            if server is not None:
                st["transports"] = server.transports()
            return {"ok": True, "status": st}, None
        if op == "models":
            return {"ok": True, "models": [engine.store.get(k).describe() for k in engine.store.keys()]}, None
        if op == "register_model":
            model = Model.from_bytes(arrays["blob"].tobytes())
            return {"ok": True, "key": engine.store.put(model)}, None
        if op == "control":
            return {"ok": True, "result": engine.control(header.get("target"), header.get("conditions"),
                                                          header.get("link"), header.get("device"))}, None
        if op == "policy":
            kw = {k: header[k] for k in ("objective", "migration", "cache_aware", "feedback", "weights", "prefetch") if k in header}
            return {"ok": True, "policy": engine.set_policy(**kw)}, None
        if op == "evict":
            return {"ok": True, "evicted": engine.evict(header["target"], header["model"])}, None
        if op == "reset_stats":
            engine.reset_stats()
            return {"ok": True}, None
        if op == "records":
            n = int(header.get("n", 100))
            return {"ok": True, "records": list(engine.records)[-n:]}, None
        if op == "shutdown":
            if client.get("uid") not in (None, os.getuid()):
                raise PermissionError("only the daemon owner may shut it down")
            if server is not None:
                threading.Thread(target=server.shutdown_all, daemon=True).start()
            return {"ok": True}, None
        return {"ok": False, "error": f"unknown op {op}"}, None
    except Exception as exc:
        log.debug("request failed", exc_info=True)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, None


class UnixHandler(socketserver.BaseRequestHandler):
    def handle(self):
        client = peer_credentials(self.request)
        while True:
            try:
                header, arrays = protocol.recv(self.request)
            except (ConnectionError, OSError):
                return
            reply, out = dispatch(self.server.engine, header, arrays, client, self.server.owner)
            reply["rid"] = header.get("rid")
            try:
                protocol.send(self.request, reply, out)
            except OSError:
                return


class UnixServer:
    def __new__(cls, path, engine, owner):
        if ":" in str(path) or not hasattr(socket, "AF_UNIX"):
            if ":" in str(path):
                host, port = str(path).rsplit(":", 1)
                port = int(port)
            else:
                host, port = "127.0.0.1", 9090
            server = socketserver.ThreadingTCPServer((host, port), UnixHandler)
            server.daemon_threads = True
            server.engine = engine
            server.owner = owner
            return server
        else:
            if os.path.exists(path):
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(path)
                    s.close()
                    raise RuntimeError(f"another aiosd is already serving {path}")
                except ConnectionRefusedError:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            server = socketserver.ThreadingUnixStreamServer(path, UnixHandler)
            server.daemon_threads = True
            server.engine = engine
            server.owner = owner
            try:
                os.chmod(path, 0o666)
            except OSError:
                pass
            return server


class KernelBridge:
    def __init__(self, engine, owner, path=kdev.DEFAULT_DEVICE, workers=32):
        self.engine = engine
        self.owner = owner
        self.chan = kdev.KernelChannel(path)
        self.chan.register()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kreq")
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="kernel-bridge")

    def start(self):
        self.thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                item = self.chan.fetch(timeout_ms=250)
            except OSError as exc:
                log.error("kernel fetch failed: %s", exc)
                self._stop.wait(0.5)
                continue
            if item is not None:
                self.pool.submit(self._serve, item)

    def _serve(self, item):
        client = {"pid": item["pid"], "uid": item["uid"], "comm": item["comm"], "transport": "kernel"}
        try:
            header, arrays = protocol.decode(item["blob"])
            reply, out = dispatch(self.engine, header, arrays, client, self.owner)
            status = 0 if reply.get("ok") else -1
        except Exception as exc:
            reply, out, status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, None, -1
        try:
            self.chan.complete(item["id"], protocol.encode(reply, out), status)
        except OSError as exc:
            log.warning("kernel complete failed for %s: %s", item["id"], exc)

    def stop(self):
        self._stop.set()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.chan.close()


class Daemon:
    def __init__(self, cfg, socket_path=None, kernel=True, testbed=None):
        self.cfg = cfg
        self.engine = Engine(cfg)
        self.socket_path = socket_path or cfg["daemon"]["socket"]
        self.kernel_wanted = kernel
        self.testbed = testbed
        self.unix = None
        self.kernel = None
        self._done = threading.Event()

    def transports(self):
        return {"unix": self.socket_path, "kernel": self.kernel.chan.path if self.kernel else None}

    def start(self):
        self.engine.start()
        self.unix = UnixServer(self.socket_path, self.engine, self)
        threading.Thread(target=self.unix.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True, name="unix").start()
        dev = self.cfg["daemon"].get("kernel_device", kdev.DEFAULT_DEVICE)
        if self.kernel_wanted and kdev.available(dev):
            try:
                self.kernel = KernelBridge(self.engine, self, dev).start()
                log.info("kernel transport active on %s", dev)
            except OSError as exc:
                log.warning("kernel transport unavailable: %s", exc)
        log.info("aiosd ready on %s with targets %s", self.socket_path, ", ".join(self.engine.targets))
        return self

    def shutdown_all(self):
        self._done.set()

    def wait(self):
        self._done.wait()

    def stop(self):
        if self.kernel:
            self.kernel.stop()
        if self.unix:
            self.unix.shutdown()
            self.unix.server_close()
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
        self.engine.stop()
        if self.testbed:
            self.testbed.stop()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aiosd", description="AIOS inference placement daemon")
    ap.add_argument("--config", default=None)
    ap.add_argument("--socket", default=None)
    ap.add_argument("--no-kernel", action="store_true")
    ap.add_argument("--spawn-nodes", action="store_true")
    ap.add_argument("--objective", choices=["latency", "energy", "balanced"], default=None)
    ap.add_argument("--no-migration", action="store_true")
    ap.add_argument("--overrides", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    overrides = json.loads(args.overrides) if args.overrides else None
    cfg = load_config(args.config, overrides)
    if args.socket:
        cfg["daemon"]["socket"] = args.socket
    testbed = Testbed(cfg, args.config, overrides=overrides).start_nodes() if args.spawn_nodes else None
    daemon = Daemon(cfg, kernel=not args.no_kernel, testbed=testbed)
    if args.objective or args.no_migration:
        daemon.engine.set_policy(objective=args.objective, migration=False if args.no_migration else None)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: daemon.shutdown_all())
    try:
        daemon.start()
        daemon.wait()
    finally:
        daemon.stop()
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
