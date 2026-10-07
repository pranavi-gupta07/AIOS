import glob
import sys
import time

import numpy as np

from aios.config import load_config, target_profile
from aios.models import ModelStore
from aios.targets import LocalTarget, RemoteTarget, TargetError

MODEL = "vision-cnn:1"
DEADLINE_MS = 80


def battery_level():
    for bat in glob.glob("/sys/class/power_supply/BAT*/capacity"):
        try:
            return int(open(bat).read()) / 100
        except (OSError, ValueError):
            pass
    return 1.0


class DeviceManager:
    def __init__(self):
        cfg = load_config()
        self.store = ModelStore()
        self.store.ensure_zoo()
        self.npu = LocalTarget(target_profile(cfg, "local-npu"), self.store)
        self.cpu = LocalTarget(target_profile(cfg, "local-cpu"), self.store)
        self.edge = RemoteTarget(target_profile(cfg, "edge"))
        self.cloud = RemoteTarget(target_profile(cfg, "cloud"))
        self.blob = self.store.blob(MODEL)
        self.shipped = set()
        self.rtt = {}
        self.last_probe = 0.0

    def probe(self):
        if time.monotonic() - self.last_probe < 1.0:
            return
        self.last_probe = time.monotonic()
        for t in (self.edge, self.cloud):
            try:
                self.rtt[t.name] = t.ping() * 1000
            except TargetError:
                self.rtt[t.name] = None

    def ensure_model(self, t):
        if t.remote and t.name not in self.shipped:
            if not t.has_model(MODEL):
                t.push_model(MODEL, self.blob)
            self.shipped.add(t.name)

    def candidates(self):
        self.probe()
        order = []
        npu_tel = self.npu.telemetry()
        npu_ok = not npu_tel["throttled"] and npu_tel["load"] < 0.5
        edge_rtt, cloud_rtt = self.rtt.get("edge"), self.rtt.get("cloud")
        if battery_level() < 0.2 and edge_rtt is not None:
            order.append(self.edge)
        if npu_ok:
            order.append(self.npu)
        if edge_rtt is not None and edge_rtt < 20:
            order.append(self.edge)
        if cloud_rtt is not None and cloud_rtt < 60:
            order.append(self.cloud)
        order += [self.npu, self.cpu]
        seen, out = set(), []
        for t in order:
            if t.name not in seen:
                seen.add(t.name)
                out.append(t)
        return out

    def infer(self, x, req_id):
        last = None
        for t in self.candidates():
            try:
                self.ensure_model(t)
                t0 = time.perf_counter()
                _, y = t.execute(req_id, MODEL, x)
                ms = (time.perf_counter() - t0) * 1000
                if ms > DEADLINE_MS and t is self.edge:
                    self.rtt["edge"] = None
                return y, t.name, ms
            except (TargetError, KeyError) as exc:
                last = exc
                self.shipped.discard(t.name)
                if t.remote:
                    self.rtt[t.name] = None
        raise RuntimeError(f"all devices failed: {last}")


def main(frames=60, fps=15):
    rng = np.random.default_rng(1)
    dm = DeviceManager()
    for i in range(frames):
        frame = rng.random((1, 3, 64, 64), dtype=np.float32)
        y, where, ms = dm.infer(frame, f"frame-{i}")
        print(f"frame {i:3d}  class {int(np.argmax(y))}  {ms:7.2f} ms  on {where}")
        time.sleep(1 / fps)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
