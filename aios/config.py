import copy
import os
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "aios.toml"


def default_socket_path():
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/aios-{os.getuid()}"
    Path(runtime).mkdir(parents=True, exist_ok=True)
    return str(Path(runtime) / "aios.sock")


def load_config(path=None, overrides=None):
    path = Path(path or os.environ.get("AIOS_CONFIG") or DEFAULT_CONFIG)
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)
    if overrides:
        cfg = merge(cfg, overrides)
    cfg.setdefault("daemon", {})
    if not cfg["daemon"].get("socket"):
        cfg["daemon"]["socket"] = os.environ.get("AIOS_SOCKET") or default_socket_path()
    names = [t["name"] for t in cfg.get("targets", [])]
    if len(names) != len(set(names)):
        raise ValueError("duplicate target names in config")
    return cfg


def merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if k == "targets" and isinstance(v, dict):
            for t in out.get("targets", []):
                if t["name"] in v:
                    t.update(v[t["name"]])
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def target_profile(cfg, name):
    for t in cfg.get("targets", []):
        if t["name"] == name:
            return dict(t)
    raise KeyError(f"no target named {name}")
