import threading
import time

import numpy as np
import pytest

from aios import protocol
from aios.config import load_config, target_profile
from aios.models import Model, build_sensor_mlp, conv2d, maxpool
from aios.node import ExecutionNode, Link, ModelCache, Thermal


def test_protocol_roundtrip():
    arrays = {"x": np.arange(12, dtype=np.float32).reshape(3, 4), "m": np.array([1, 2, 3], dtype=np.int64)}
    header, out = protocol.decode(protocol.encode({"op": "infer", "k": [1, 2]}, arrays))
    assert header["op"] == "infer" and header["k"] == [1, 2]
    for k in arrays:
        assert out[k].dtype == arrays[k].dtype
        np.testing.assert_array_equal(out[k], arrays[k])


def test_protocol_truncated():
    blob = protocol.encode({"op": "x"}, {"a": np.zeros(10)})
    with pytest.raises(protocol.ProtocolError):
        protocol.decode(blob[:-3])


def naive_conv(x, w, b, stride, pad):
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    n, c, h, wd = x.shape
    f, _, k, _ = w.shape
    ho, wo = (h - k) // stride + 1, (wd - k) // stride + 1
    out = np.zeros((n, f, ho, wo), dtype=np.float64)
    for i in range(ho):
        for j in range(wo):
            patch = x[:, :, i * stride:i * stride + k, j * stride:j * stride + k]
            out[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3])) + b
    return out


@pytest.mark.parametrize("stride,pad", [(1, 1), (2, 0), (2, 1)])
def test_conv_matches_naive(stride, pad):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 9, 9)).astype(np.float32)
    w = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    b = rng.standard_normal(4).astype(np.float32)
    np.testing.assert_allclose(conv2d(x, w, b, stride, pad), naive_conv(x, w, b, stride, pad), rtol=1e-4, atol=1e-4)


def test_maxpool():
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    np.testing.assert_array_equal(maxpool(x, 2)[0, 0], [[5, 7], [13, 15]])


def test_model_shapes_and_flops(store):
    for key in store.keys():
        m = store.get(key)
        x = np.random.default_rng(1).random(m.input_shape, dtype=np.float32)
        y, nxt = m.run(x)
        assert nxt == len(m.layers)
        assert tuple(y.shape) == tuple(m.output_shapes[-1])
        assert m.flops == sum(m.layer_flops) > 0
        assert np.isfinite(y).all()


def test_split_execution_equals_full(store):
    m = store.get("vision-cnn")
    x = np.random.default_rng(2).random(m.input_shape, dtype=np.float32)
    full, _ = m.run(x)
    for cut in (1, 4, 9, len(m.layers) - 1):
        mid, at = m.run(x, 0, cut)
        assert at == cut
        assert mid.nbytes == m.activation_bytes(cut)
        rest, _ = m.run(mid, cut)
        np.testing.assert_allclose(rest, full, rtol=1e-5)


def test_model_serialization_roundtrip():
    m = build_sensor_mlp()
    m2 = Model.from_bytes(m.to_bytes())
    assert m2.key == m.key and m2.flops == m.flops
    x = np.ones(m.input_shape, dtype=np.float32)
    np.testing.assert_array_equal(m.run(x)[0], m2.run(x)[0])


def test_store_resolves_latest(store):
    assert store.resolve("vision-cnn") == "vision-cnn:1"
    with pytest.raises(KeyError):
        store.resolve("nope")


def test_cache_lru_eviction():
    c = ModelCache(1)

    class M:
        def __init__(self, key, n):
            self.key, self.nbytes = key, n

    c.put(M("a", 400_000))
    c.put(M("b", 400_000))
    assert c.get("a") is not None
    evicted = c.put(M("c", 400_000))
    assert evicted == ["b"]
    assert set(c.keys()) == {"a", "c"}
    with pytest.raises(MemoryError):
        c.put(M("huge", 2 * 2 ** 20))


def test_thermal_throttles_and_recovers():
    t = Thermal(40, 80, heat_rate=50, cool_rate=0.5, throttle_factor=0.5)
    for _ in range(20):
        t.step(0.1, 1.0)
    assert t.throttled and t.factor == 0.5
    for _ in range(200):
        t.step(0.1, 0.0)
    assert not t.throttled


def test_link_shapes_bandwidth_and_rtt():
    link = Link(rtt_ms=10, bandwidth_mbps=8)
    t0 = time.perf_counter()
    link.transmit(10_000)
    elapsed = time.perf_counter() - t0
    assert 0.014 < elapsed < 0.05


def test_node_emulates_profile_speed(store):
    prof = dict(target_profile(load_config(), "local-cpu"))
    node = ExecutionNode(prof, store=store)
    m = store.get("vision-cnn")
    x = np.zeros(m.input_shape, dtype=np.float32)
    node.load_from_store(m.key)
    info, _ = node.execute("r1", m.key, x)
    expected = m.flops / (prof["gflops"] * 1e9) * 1000
    assert info["status"] == "done"
    assert expected * 0.9 < info["compute_ms"] < expected * 1.6
    node.set_conditions(load=0.5)
    info2, _ = node.execute("r2", m.key, x)
    assert info2["compute_ms"] > info["compute_ms"] * 1.3
    node.close()


def test_node_preemption_checkpoint_resume(store):
    prof = dict(target_profile(load_config(), "local-cpu"))
    node = ExecutionNode(prof, store=store)
    m = store.get("batch-encoder")
    x = np.random.default_rng(4).random(m.input_shape, dtype=np.float32)
    node.load_from_store(m.key)
    ref, _ = m.run(x)
    result = {}
    th = threading.Thread(target=lambda: result.update(zip(("info", "y"), node.execute("p1", m.key, x))))
    th.start()
    time.sleep(0.05)
    assert node.preempt("p1")
    th.join()
    info = result["info"]
    assert info["status"] == "preempted"
    assert 0 < info["next_layer"] < len(m.layers)
    other = ExecutionNode(dict(target_profile(load_config(), "local-npu")), store=store)
    info2, y = other.execute("p1b", m.key, result["y"], info["next_layer"])
    assert info2["status"] == "done"
    np.testing.assert_allclose(y, ref, rtol=1e-5)
    node.close()
    other.close()
