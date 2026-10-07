import pytest

from aios.config import load_config
from aios.monitor import DeviceState, HostMonitor, TargetState
from aios.scheduler import Requirements, Scheduler


class FakeTarget:
    def __init__(self, profile, remote):
        self.profile = profile
        self.name = profile["name"]
        self.remote = remote
        self.tier = profile.get("tier", "device")
        self.trusted = profile.get("trusted", True)
        self.tel = {"online": True, "effective_gflops": profile["gflops"], "active": 0, "queued": 0,
                    "temperature_c": 40.0, "throttle_c": profile.get("throttle_c", 85), "throttled": False,
                    "cache": {"models": []}, "service_ms": None, "load": 0.0}

    def telemetry(self):
        return self.tel


class FakeMonitor:
    def __init__(self, cfg):
        self.device = DeviceState(cfg, HostMonitor())
        self.states = {}
        for p in cfg["targets"]:
            t = FakeTarget(p, p["kind"] == "remote")
            st = TargetState(t)
            st.telemetry = t.tel
            self.states[p["name"]] = st

    def state(self, name):
        return self.states[name]

    def cache_everywhere(self, key):
        for st in self.states.values():
            st.target.tel["cache"]["models"].append(key)


@pytest.fixture
def sched(store):
    cfg = load_config()
    mon = FakeMonitor(cfg)
    for k in store.keys():
        mon.cache_everywhere(k)
    return Scheduler(cfg, mon, store), mon


def test_requirements_validation():
    with pytest.raises(ValueError):
        Requirements.from_dict({"objective": "fastest"})
    with pytest.raises(ValueError):
        Requirements.from_dict({"privacy": "secret"})
    assert Requirements.from_dict(None).privacy == "any"


def test_small_model_stays_on_device(sched, store):
    s, _ = sched
    d = s.place(store.get("sensor-mlp"), Requirements(objective="latency"))
    assert d["target"] in ("local-npu", "local-cpu")


def test_heavy_model_prefers_edge_for_latency(sched, store):
    s, _ = sched
    d = s.place(store.get("doc-encoder"), Requirements(objective="latency"))
    assert d["target"] == "edge"


def test_privacy_excludes_untrusted_and_remote(sched, store):
    s, _ = sched
    m = store.get("doc-encoder")
    d = s.place(m, Requirements(objective="latency", privacy="trusted"))
    assert d["target"] != "cloud"
    d = s.place(m, Requirements(objective="latency", privacy="device"))
    assert d["target"] in ("local-npu", "local-cpu")
    reasons = {r["target"]: r.get("reason") for r in d["candidates"] if not r["feasible"]}
    assert "edge" in reasons and "cloud" in reasons


def test_offline_target_never_chosen(sched, store):
    s, mon = sched
    mon.states["edge"].online = False
    d = s.place(store.get("doc-encoder"), Requirements(objective="latency"))
    assert d["target"] != "edge"


def test_load_shifts_placement(sched, store):
    s, mon = sched
    m = store.get("doc-encoder")
    assert s.place(m, Requirements(objective="latency"))["target"] == "edge"
    mon.states["edge"].target.tel["effective_gflops"] = 1.0
    assert s.place(m, Requirements(objective="latency"))["target"] != "edge"


def test_network_degradation_shifts_placement(sched, store):
    s, mon = sched
    m = store.get("vision-cnn")
    mon.states["local-npu"].target.tel["effective_gflops"] = 2.0
    assert s.place(m, Requirements(objective="latency"))["target"] == "edge"
    mon.states["edge"].bandwidth_mbps = 1.0
    mon.states["edge"].rtt_s = 0.3
    assert s.place(m, Requirements(objective="latency"))["target"] != "edge"


def test_memory_feasibility(sched, store):
    s, mon = sched
    m = store.get("doc-encoder")
    est = s.estimate(m, "local-npu")
    assert est["feasible"]
    mon.states["local-npu"].target.profile["memory_mb"] = 10
    assert not s.estimate(m, "local-npu")["feasible"]


def test_uncached_model_costs_transfer(sched, store):
    s, mon = sched
    m = store.get("batch-encoder")
    warm = s.estimate(m, "cloud")
    mon.states["cloud"].target.tel["cache"]["models"].remove(m.key)
    cold = s.estimate(m, "cloud")
    assert cold["model_s"] > 1.0 and warm["model_s"] == 0
    s.use_cache_state = False
    assert s.estimate(m, "cloud")["model_s"] == 0


def test_deadline_filters_candidates(sched, store):
    s, _ = sched
    m = store.get("vision-cnn")
    d = s.place(m, Requirements(objective="energy", deadline_ms=30))
    assert d["estimate"]["latency_s"] * 1000 <= 30


def test_low_battery_increases_energy_weight(sched, store):
    s, mon = sched
    req = Requirements(objective="balanced")
    _, w = s.weights_for(req)
    mon.device.level = 0.05
    _, w_low = s.weights_for(req)
    assert w_low["energy"] > w["energy"]


def test_session_hysteresis_keeps_current(sched, store):
    s, mon = sched
    m = store.get("vision-cnn")
    best = s.place(m, Requirements(objective="latency"))
    other = next(r["target"] for r in sorted(
        [r for r in best["candidates"] if r["feasible"]], key=lambda r: r["score"]) if r["target"] != best["target"])
    rows = {r["target"]: r for r in best["candidates"] if r["feasible"]}
    if rows[other]["score"] <= rows[best["target"]]["score"] * (1 + s.session_hysteresis):
        assert s.place(m, Requirements(objective="latency"), current=other)["target"] == other


def test_thermal_lookahead_penalizes_hot_target(sched, store):
    s, mon = sched
    m = store.get("vision-cnn")
    cool = s.estimate(m, "local-npu")["compute_s"]
    mon.states["local-npu"].target.tel["temperature_c"] = 79.5
    hot = s.estimate(m, "local-npu")["compute_s"]
    assert hot > cool * 1.5


def test_migration_choice_when_current_slows(sched, store):
    s, mon = sched
    m = store.get("doc-encoder")
    req = Requirements(objective="latency")
    assert s.migration_choice(m, req, 4, "edge") is None
    mon.states["edge"].target.tel["effective_gflops"] = 0.5
    choice = s.migration_choice(m, req, 4, "edge")
    assert choice is not None and choice["target"] != "edge"


def test_feedback_correction(sched, store):
    s, _ = sched
    m = store.get("vision-cnn")
    base = s.estimate(m, "cloud")["compute_s"]
    for _ in range(10):
        s.observe("cloud", m.key, 0.002, 0.010)
    assert s.estimate(m, "cloud")["compute_s"] > base * 3
