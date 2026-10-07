import json
import os
import socket
import subprocess
import sys
import time

from .config import DEFAULT_CONFIG, PROJECT_ROOT


def wait_port(host, port, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def wait_unix(path, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if os.path.exists(path):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(path)
                s.close()
                return True
            except OSError:
                pass
        time.sleep(0.05)
    return False


def _env(threads=None):
    env = dict(os.environ)
    if threads:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
            env[var] = str(threads)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


class Testbed:
    def __init__(self, cfg, config_path=None, log_dir=None, overrides=None):
        self.cfg = cfg
        self.overrides = overrides
        self.config_path = str(config_path or os.environ.get("AIOS_CONFIG") or DEFAULT_CONFIG)
        self.log_dir = log_dir
        self.procs = {}

    def _log(self, name):
        if not self.log_dir:
            return subprocess.DEVNULL
        os.makedirs(self.log_dir, exist_ok=True)
        return open(os.path.join(self.log_dir, f"{name}.log"), "ab")

    def _override_args(self):
        return ["--overrides", json.dumps(self.overrides)] if self.overrides else []

    def start_nodes(self):
        for t in self.cfg["targets"]:
            if t.get("kind") != "remote" or t.get("enabled", True) is False:
                continue
            if t["name"] in self.procs:
                continue
            p = subprocess.Popen([sys.executable, "-m", "aios.node", "--config", self.config_path, "--target", t["name"],
                                  *self._override_args()],
                                 stdout=self._log(t["name"]), stderr=subprocess.STDOUT,
                                 env=_env(t.get("threads", 2)), cwd=PROJECT_ROOT)
            self.procs[t["name"]] = p
        for t in self.cfg["targets"]:
            if t.get("kind") == "remote" and t["name"] in self.procs:
                if not wait_port(t.get("host", "127.0.0.1"), t["port"]):
                    raise RuntimeError(f"node {t['name']} did not start")
        return self

    def start_daemon(self, extra=None):
        sock = self.cfg["daemon"]["socket"]
        args = [sys.executable, "-m", "aios.daemon", "--config", self.config_path, "--socket", sock]
        args += self._override_args() + list(extra or [])
        p = subprocess.Popen(args, stdout=self._log("aiosd"), stderr=subprocess.STDOUT, env=_env(4), cwd=PROJECT_ROOT)
        self.procs["aiosd"] = p
        if not wait_unix(sock):
            raise RuntimeError("aiosd did not start")
        return self

    def stop(self):
        for name, p in reversed(list(self.procs.items())):
            if p.poll() is None:
                p.terminate()
        for p in self.procs.values():
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        self.procs.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
