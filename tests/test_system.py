import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from aios import kdev
from aios.api import AIOSError, Client
from aios.config import load_config
from aios import testbed

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def system(tmp_path_factory, store):
    runtime = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    sock = os.path.join(runtime, f"aios-test-{os.getpid()}.sock")
    overrides = {"daemon": {"socket": sock}, "targets": {"edge": {"port": 7301}, "cloud": {"port": 7302}},
                 "policy": {"rules": [{"match": {"comm": "aios-private"}, "privacy": "device"}]}}
    cfg = load_config(None, overrides)
    tb = testbed.Testbed(cfg, overrides=overrides, log_dir=str(tmp_path_factory.mktemp("logs")))
    tb.start_nodes().start_daemon(["--no-kernel"])
    c = Client(sock, transport="unix")
    time.sleep(0.5)
    yield c, sock
    c.close()
    tb.stop()


def test_infer_matches_local_reference(system, store):
    c, _ = system
    for key in store.keys():
        m = store.get(key)
        x = np.random.default_rng(9).random(m.input_shape, dtype=np.float32)
        r = c.infer(m.name, x)
        np.testing.assert_allclose(r.output, m.run(x)[0], rtol=1e-4, atol=1e-5)
        assert r.meta["latency_ms"] > 0 and r.target in ("local-cpu", "local-npu", "edge", "cloud")


def test_pinned_execution_on_every_target(system, store):
    c, _ = system
    x = np.zeros((1, 3, 64, 64), dtype=np.float32)
    for t in ("local-cpu", "local-npu", "edge", "cloud"):
        assert c.infer("vision-cnn", x, pin=t).target == t


def test_shape_validation(system):
    c, _ = system
    with pytest.raises(AIOSError):
        c.infer("vision-cnn", np.zeros((2, 2), dtype=np.float32))


def test_privacy_device_only(system):
    c, _ = system
    x = np.zeros((32, 768), dtype=np.float32)
    assert c.infer("batch-encoder", x, privacy="device").target in ("local-cpu", "local-npu")


def test_failover_when_edge_goes_offline(system):
    c, _ = system
    x = np.zeros((256, 768), dtype=np.float32)
    try:
        c.control("edge", online=False)
        r = c.infer("doc-encoder", x, objective="latency")
        assert r.target != "edge"
        with pytest.raises(AIOSError):
            c.infer("vision-cnn", np.zeros((1, 3, 64, 64), dtype=np.float32), pin="edge")
    finally:
        c.control("edge", online=True)


def test_migration_preserves_result(system, store):
    c, _ = system
    m = store.get("doc-encoder")
    x = np.random.default_rng(11).random(m.input_shape, dtype=np.float32)
    for t in ("edge", "local-npu", "cloud"):
        c.infer("doc-encoder", x, pin=t)
    c.set_policy(migration=True)
    c.control("edge", load=0.0)
    time.sleep(1.0)
    out = {}
    _, sock = system
    worker = Client(sock, transport="unix")
    th = threading.Thread(target=lambda: out.setdefault("r", worker.infer("doc-encoder", x, objective="latency",
                                                                         privacy="trusted")))
    try:
        th.start()
        time.sleep(0.06)
        c.control("edge", load=0.9)
        th.join()
    finally:
        c.control("edge", load=0.0)
        worker.close()
    r = out["r"]
    assert r.meta["initial_target"] == "edge"
    assert len(r.migrations) >= 1
    assert r.migrations[0]["state_bytes"] > 0
    np.testing.assert_allclose(r.output, m.run(x)[0], rtol=1e-4, atol=1e-5)


def test_explain_and_status(system):
    c, _ = system
    d = c.explain("vision-cnn", objective="balanced")
    assert d["target"] and len(d["candidates"]) == 4
    st = c.status()
    assert set(st["targets"]) == {"local-cpu", "local-npu", "edge", "cloud"}
    assert st["stats"]["completed"] > 0
    assert st["targets"]["edge"]["rtt_ms"] > 3


def test_admin_policy_rule_by_process_name(system):
    _, sock = system
    script = ("import numpy as np, aios.api as a; c=a.Client(%r, transport='unix'); "
              "print(c.infer('doc-encoder', np.zeros((256,768),dtype=np.float32), objective='latency').target)") % sock
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    code = f"import ctypes,sys;ctypes.CDLL(None).prctl(15,b'aios-private',0,0,0);exec({script!r})"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() in ("local-cpu", "local-npu")


def test_kernel_abi_matches_header():
    header = (ROOT / "kernel" / "aios_uapi.h").read_text()
    assert re.search(r"AIOS_IOC_MAGIC 'A'", header)
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        pytest.skip("no C compiler")
    src = ('#include <stdio.h>\n#include "aios_uapi.h"\nint main(void){printf("%lu %lu %lu %lu %lu %lu",'
           "(unsigned long)AIOS_IOC_SUBMIT,(unsigned long)AIOS_IOC_WAIT,(unsigned long)AIOS_IOC_REGISTER,"
           "(unsigned long)AIOS_IOC_FETCH,(unsigned long)AIOS_IOC_COMPLETE,(unsigned long)AIOS_IOC_STATS);return 0;}")
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"aios_abi_{os.getpid()}"
    tmp.with_suffix(".c").write_text(src)
    subprocess.run([cc, "-I", str(ROOT / "kernel"), "-o", str(tmp), str(tmp.with_suffix(".c"))], check=True)
    vals = [int(v) for v in subprocess.run([str(tmp)], capture_output=True, text=True, check=True).stdout.split()]
    tmp.unlink()
    tmp.with_suffix(".c").unlink()
    assert vals == [kdev.IOC_SUBMIT, kdev.IOC_WAIT, kdev.IOC_REGISTER, kdev.IOC_FETCH, kdev.IOC_COMPLETE, kdev.IOC_STATS]


def test_background_chunked_model_push(system, store):
    from aios.config import target_profile
    from aios.targets import RemoteTarget
    prof = dict(target_profile(load_config(), "edge"), port=7301)
    t = RemoteTarget(prof)
    key = "kws-cnn:1"
    t.evict(key)
    assert not t.has_model(key)
    res = t.push_model(key, store.blob(key), background=True, chunk=64 * 1024)
    assert t.has_model(key) and res["seconds"] > 0
    t.close()
