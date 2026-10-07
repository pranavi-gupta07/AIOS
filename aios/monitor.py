import glob
import os
import threading
import time
from pathlib import Path

from .targets import TargetError


def _read(path, default=None):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


class HostMonitor:
    def __init__(self):
        self._cpu_prev = self._cpu_times()
        self._rapl_prev = self._rapl()
        self._lock = threading.Lock()

    @staticmethod
    def _cpu_times():
        line = _read("/proc/stat", "cpu 0 0 0 0").splitlines()[0].split()
        vals = [int(v) for v in line[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        return sum(vals), idle

    @staticmethod
    def _rapl():
        total = 0
        found = False
        for p in glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj"):
            if p.count(":") != 1:
                continue
            v = _read(p)
            if v is not None and v.isdigit():
                total += int(v)
                found = True
        return total if found else None

    def cpu_utilization(self):
        with self._lock:
            total, idle = self._cpu_times()
            pt, pi = self._cpu_prev
            self._cpu_prev = (total, idle)
        dt = total - pt
        return 0.0 if dt <= 0 else 1.0 - (idle - pi) / dt

    def memory(self):
        info = {}
        for line in (_read("/proc/meminfo", "") or "").splitlines():
            k, _, v = line.partition(":")
            parts = v.split()
            if parts:
                info[k] = int(parts[0]) / 1024
        return {"total_mb": round(info.get("MemTotal", 0), 1), "available_mb": round(info.get("MemAvailable", 0), 1)}

    def thermal(self):
        temps = {}
        for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
            t = _read(f"{zone}/temp")
            kind = _read(f"{zone}/type", os.path.basename(zone))
            if t and t.lstrip("-").isdigit():
                temps[kind] = int(t) / 1000
        return {"max_c": max(temps.values()) if temps else None, "zones": temps}

    def battery(self):
        for bat in sorted(glob.glob("/sys/class/power_supply/BAT*")):
            cap = _read(f"{bat}/capacity")
            status = _read(f"{bat}/status")
            power = _read(f"{bat}/power_now")
            if power is None:
                cur, volt = _read(f"{bat}/current_now"), _read(f"{bat}/voltage_now")
                if cur and volt and cur.isdigit() and volt.isdigit():
                    power = str(int(cur) * int(volt) // 1_000_000)
            return {"present": True, "level": int(cap) / 100 if cap and cap.isdigit() else None, "status": status,
                    "power_w": int(power) / 1e6 if power and power.isdigit() else None}
        return {"present": False}

    def network(self):
        rx = tx = 0
        for line in (_read("/proc/net/dev", "") or "").splitlines()[2:]:
            name, _, rest = line.partition(":")
            if name.strip() == "lo":
                continue
            f = rest.split()
            if len(f) >= 9:
                rx += int(f[0])
                tx += int(f[8])
        return {"rx_bytes": rx, "tx_bytes": tx}

    def energy_j(self):
        v = self._rapl()
        return None if v is None else v / 1e6

    def snapshot(self):
        load1, load5, load15 = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
        return {"cpu_utilization": round(self.cpu_utilization(), 4), "loadavg": [load1, load5, load15],
                "cpus": os.cpu_count(), "memory": self.memory(), "thermal": self.thermal(),
                "battery": self.battery(), "network": self.network(),
                "rapl_available": self._rapl_prev is not None}


class DeviceState:
    def __init__(self, cfg, host):
        dev = cfg.get("device", {})
        self.battery_wh = dev.get("battery_wh", 12.0)
        self.level = dev.get("battery_level", 1.0)
        self.low = dev.get("low_battery", 0.2)
        self.radio_tx_w = dev.get("radio_tx_w", 1.3)
        self.radio_rx_w = dev.get("radio_rx_w", 0.9)
        self.radio_idle_w = dev.get("radio_idle_w", 0.15)
        self.host_coupled = dev.get("host_coupled", False)
        self.charging = False
        self.host = host
        self.energy_used_j = 0.0
        self._lock = threading.Lock()

    def drain(self, joules):
        with self._lock:
            self.energy_used_j += joules
            if not self.charging:
                self.level = max(0.0, self.level - joules / (self.battery_wh * 3600))

    def set(self, **kw):
        with self._lock:
            if "battery_level" in kw:
                self.level = float(kw["battery_level"])
            if "charging" in kw:
                self.charging = bool(kw["charging"])

    def refresh_from_host(self):
        if not self.host_coupled:
            return
        b = self.host.battery()
        if b.get("present") and b.get("level") is not None:
            with self._lock:
                self.level = b["level"]
                self.charging = b.get("status") in ("Charging", "Full", "Not charging")

    @property
    def low_battery(self):
        return self.level < self.low and not self.charging

    def snapshot(self):
        with self._lock:
            return {"battery_level": round(self.level, 5), "charging": self.charging, "low_battery": self.low_battery,
                    "energy_used_j": round(self.energy_used_j, 4), "battery_wh": self.battery_wh}


class TargetState:
    def __init__(self, target):
        self.target = target
        self.name = target.name
        self.telemetry = None
        self.online = True
        self.rtt_s = target.profile.get("rtt_ms", 0) / 1000 if target.remote else 0.0
        self.bandwidth_mbps = target.profile.get("bandwidth_mbps", 1000) if target.remote else None
        self.updated = 0.0
        self.failures = 0
        self.inflight = 0
        self.inflight_at_refresh = 0
        self.pending_models = set()

    def view(self):
        tel = self.telemetry or {}
        return {"name": self.name, "online": self.online, "rtt_ms": round(self.rtt_s * 1000, 3),
                "bandwidth_mbps": None if self.bandwidth_mbps is None else round(self.bandwidth_mbps, 2),
                "inflight": self.inflight, "age_ms": round((time.monotonic() - self.updated) * 1000, 1) if self.updated else None,
                "telemetry": tel}


class ResourceMonitor:
    def __init__(self, cfg, targets, host=None, device=None):
        d = cfg.get("daemon", {})
        self.interval = d.get("monitor_interval_ms", 100) / 1000
        self.probe_interval = d.get("probe_interval_ms", 1000) / 1000
        self.targets = targets
        self.host = host or HostMonitor()
        self.device = device or DeviceState(cfg, self.host)
        self.states = {name: TargetState(t) for name, t in targets.items()}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._threads = []
        self._listeners = []
        self.host_snapshot = {}

    def on_change(self, fn):
        self._listeners.append(fn)

    def _emit(self, name, what):
        for fn in self._listeners:
            try:
                fn(name, what)
            except Exception:
                pass

    def start(self):
        for name in self.states:
            th = threading.Thread(target=self._poll_target, args=(name,), daemon=True, name=f"monitor-{name}")
            th.start()
            self._threads.append(th)
        th = threading.Thread(target=self._poll_host, daemon=True, name="monitor-host")
        th.start()
        self._threads.append(th)

    def stop(self):
        self._stop.set()

    def refresh(self, name):
        st = self.states[name]
        t = st.target
        try:
            inflight = st.inflight
            tel = t.telemetry()
        except (TargetError, OSError) as exc:
            self._fail(name, exc)
            return False
        with self._lock:
            was_online = st.online and tel.get("online", True)
            prev = st.telemetry
            st.telemetry = tel
            st.inflight_at_refresh = min(inflight, st.inflight)
            st.online = tel.get("online", True)
            st.updated = time.monotonic()
            st.failures = 0
            for k in list(st.pending_models):
                if k in tel["cache"]["models"]:
                    st.pending_models.discard(k)
        if not was_online and st.online:
            self._emit(name, "online")
        if prev and (prev.get("throttled") != tel.get("throttled") or abs(prev.get("load", 0) - tel.get("load", 0)) > 0.05):
            self._emit(name, "conditions")
        return True

    def _fail(self, name, exc):
        st = self.states[name]
        with self._lock:
            st.failures += 1
            changed = st.online and st.failures >= 2
            if st.failures >= 2:
                st.online = False
        if changed:
            self._emit(name, "offline")

    def mark_offline(self, name):
        with self._lock:
            st = self.states[name]
            changed = st.online
            st.online = False
            st.failures = max(st.failures, 2)
        if changed:
            self._emit(name, "offline")

    def measure_link(self, name, bandwidth=True):
        st = self.states[name]
        t = st.target
        if not t.remote:
            return None
        try:
            rtt = min(t.ping() for _ in range(2))
            with self._lock:
                st.rtt_s = rtt if st.updated == 0 else st.rtt_s * 0.6 + rtt * 0.4
            if bandwidth:
                nbytes = 96 * 1024
                elapsed = t.probe(nbytes)
                xfer = max(1e-4, elapsed - rtt)
                bw = nbytes * 8 / xfer / 1e6
                with self._lock:
                    st.bandwidth_mbps = st.bandwidth_mbps * 0.5 + bw * 0.5
        except (TargetError, OSError) as exc:
            self._fail(name, exc)
            return None
        return {"rtt_ms": st.rtt_s * 1000, "bandwidth_mbps": st.bandwidth_mbps}

    def observe_transfer(self, name, nbytes, seconds):
        st = self.states[name]
        if not st.target.remote or nbytes < 32 * 1024:
            return
        xfer = seconds - st.rtt_s
        if xfer <= 1e-4:
            return
        bw = nbytes * 8 / xfer / 1e6
        with self._lock:
            st.bandwidth_mbps = st.bandwidth_mbps * 0.7 + bw * 0.3

    def observe_rtt(self, name, seconds):
        st = self.states[name]
        if st.target.remote and seconds > 0:
            with self._lock:
                st.rtt_s = st.rtt_s * 0.8 + seconds * 0.2

    def _poll_target(self, name):
        last_probe = 0.0
        while not self._stop.is_set():
            ok = self.refresh(name)
            now = time.monotonic()
            st = self.states[name]
            if st.target.remote and (now - last_probe >= self.probe_interval or (not st.online and ok)):
                self.measure_link(name, bandwidth=True)
                last_probe = now
            self._stop.wait(self.interval if st.online else max(self.interval, 0.25))

    def _poll_host(self):
        while not self._stop.is_set():
            snap = self.host.snapshot()
            self.device.refresh_from_host()
            if self.device.host_coupled:
                load = max(0.0, min(0.9, snap["cpu_utilization"]))
                for st in self.states.values():
                    if not st.target.remote:
                        st.target.control(load=load)
            self.host_snapshot = snap
            self._stop.wait(max(0.5, self.interval * 5))

    def snapshot(self):
        with self._lock:
            return {name: st.view() for name, st in self.states.items()}

    def state(self, name):
        return self.states[name]

    def begin(self, name):
        with self._lock:
            self.states[name].inflight += 1

    def end(self, name):
        with self._lock:
            self.states[name].inflight = max(0, self.states[name].inflight - 1)
