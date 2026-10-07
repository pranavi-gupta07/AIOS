import os
import threading
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import protocol


def conv2d(x, w, b, stride, pad):
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    f, c, k, _ = w.shape
    win = sliding_window_view(x, (k, k), axis=(2, 3))[:, :, ::stride, ::stride]
    n, _, ho, wo = win.shape[:4]
    cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(n * ho * wo, c * k * k)
    out = cols @ w.reshape(f, -1).T + b
    return out.reshape(n, ho, wo, f).transpose(0, 3, 1, 2)


def maxpool(x, k):
    n, c, h, w = x.shape
    x = x[:, :, :h - h % k, :w - w % k]
    return x.reshape(n, c, h // k, k, w // k, k).max(axis=(3, 5))


class Model:
    def __init__(self, name, version, input_shape, layers, weights, description=""):
        self.name = name
        self.version = version
        self.input_shape = tuple(input_shape)
        self.layers = layers
        self.weights = weights
        self.description = description
        self._shapes = None
        self._flops = None

    @property
    def key(self):
        return f"{self.name}:{self.version}"

    @property
    def nbytes(self):
        return int(sum(w.nbytes for w in self.weights.values()))

    def _analyse(self):
        shape = self.input_shape
        shapes, flops = [], []
        for layer in self.layers:
            t = layer["type"]
            if t == "conv":
                f, c, k, _ = self.weights[layer["w"]].shape
                n, _, h, w = shape
                p, s = layer["pad"], layer["stride"]
                ho, wo = (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1
                shape = (n, f, ho, wo)
                flops.append(2 * n * ho * wo * f * c * k * k)
            elif t == "pool":
                k = layer["k"]
                n, c, h, w = shape
                shape = (n, c, h // k, w // k)
                flops.append(n * c * h * w)
            elif t == "flatten":
                shape = (shape[0], int(np.prod(shape[1:])))
                flops.append(0)
            elif t == "dense":
                i, o = self.weights[layer["w"]].shape
                shape = (shape[0], o)
                flops.append(2 * shape[0] * i * o)
            shapes.append(shape)
        self._shapes, self._flops = shapes, flops

    @property
    def output_shapes(self):
        if self._shapes is None:
            self._analyse()
        return self._shapes

    @property
    def layer_flops(self):
        if self._flops is None:
            self._analyse()
        return self._flops

    @property
    def flops(self):
        return int(sum(self.layer_flops))

    def flops_between(self, start, end=None):
        return int(sum(self.layer_flops[start:end]))

    def activation_bytes(self, layer_index):
        if layer_index <= 0:
            return int(np.prod(self.input_shape)) * 4
        return int(np.prod(self.output_shapes[layer_index - 1])) * 4

    @property
    def output_bytes(self):
        return int(np.prod(self.output_shapes[-1])) * 4

    @property
    def peak_activation_bytes(self):
        return max(int(np.prod(s)) * 4 for s in [self.input_shape, *self.output_shapes])

    @property
    def memory_mb(self):
        return (self.nbytes + 3 * self.peak_activation_bytes) / 2 ** 20

    def run_layer(self, i, x):
        layer = self.layers[i]
        t = layer["type"]
        if t == "conv":
            x = conv2d(x, self.weights[layer["w"]], self.weights[layer["b"]], layer["stride"], layer["pad"])
        elif t == "pool":
            return maxpool(x, layer["k"])
        elif t == "flatten":
            return x.reshape(x.shape[0], -1)
        elif t == "dense":
            x = x @ self.weights[layer["w"]] + self.weights[layer["b"]]
        if layer.get("relu"):
            np.maximum(x, 0, out=x)
        return x

    def run(self, x, start=0, end=None, should_stop=None, on_layer=None):
        end = len(self.layers) if end is None else end
        x = np.asarray(x, dtype=np.float32)
        i = start
        while i < end:
            if should_stop is not None and i > start and should_stop():
                return x, i
            x = self.run_layer(i, x).astype(np.float32, copy=False)
            i += 1
            if on_layer is not None:
                on_layer(i, x)
        return x, i

    def spec(self):
        return {"name": self.name, "version": self.version, "input_shape": list(self.input_shape),
                "layers": self.layers, "description": self.description}

    def to_bytes(self):
        return protocol.encode({"kind": "model", "spec": self.spec()}, self.weights)

    @classmethod
    def from_bytes(cls, blob):
        header, arrays = protocol.decode(blob)
        s = header["spec"]
        return cls(s["name"], s["version"], s["input_shape"], s["layers"], arrays, s.get("description", ""))

    def describe(self):
        return {"key": self.key, "input_shape": list(self.input_shape), "layers": len(self.layers),
                "gflop": round(self.flops / 1e9, 4), "size_mb": round(self.nbytes / 2 ** 20, 3),
                "memory_mb": round(self.memory_mb, 3), "description": self.description}


class Builder:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.layers = []
        self.weights = {}

    def conv(self, cin, cout, k=3, stride=1, pad=1, relu=True):
        i = len(self.layers)
        std = np.sqrt(2.0 / (cin * k * k))
        self.weights[f"l{i}.w"] = (self.rng.standard_normal((cout, cin, k, k)) * std).astype(np.float32)
        self.weights[f"l{i}.b"] = np.zeros(cout, dtype=np.float32)
        self.layers.append({"type": "conv", "w": f"l{i}.w", "b": f"l{i}.b", "stride": stride, "pad": pad, "relu": relu})
        return self

    def pool(self, k=2):
        self.layers.append({"type": "pool", "k": k})
        return self

    def flatten(self):
        self.layers.append({"type": "flatten"})
        return self

    def dense(self, cin, cout, relu=True):
        i = len(self.layers)
        std = np.sqrt(2.0 / cin)
        self.weights[f"l{i}.w"] = (self.rng.standard_normal((cin, cout)) * std).astype(np.float32)
        self.weights[f"l{i}.b"] = np.zeros(cout, dtype=np.float32)
        self.layers.append({"type": "dense", "w": f"l{i}.w", "b": f"l{i}.b", "relu": relu})
        return self


def build_sensor_mlp():
    b = Builder(11).dense(64, 256).dense(256, 256).dense(256, 10, relu=False)
    return Model("sensor-mlp", "1", (1, 64), b.layers, b.weights, "IoT sensor anomaly classifier")


def build_kws_cnn():
    b = (Builder(23).conv(1, 16).pool().conv(16, 32).pool().conv(32, 64).pool()
         .flatten().dense(64 * 5 * 5, 128).dense(128, 12, relu=False))
    return Model("kws-cnn", "1", (1, 1, 40, 40), b.layers, b.weights, "keyword spotting on 40x40 spectrogram")


def build_vision_cnn():
    b = (Builder(37).conv(3, 32).conv(32, 32).pool().conv(32, 64).conv(64, 64).pool()
         .conv(64, 128).pool().conv(128, 128).pool().flatten().dense(128 * 4 * 4, 256).dense(256, 10, relu=False))
    return Model("vision-cnn", "1", (1, 3, 64, 64), b.layers, b.weights, "image classifier on 64x64 RGB frames")


def build_batch_encoder():
    b = Builder(53)
    for _ in range(15):
        b.dense(768, 768)
    b.dense(768, 768, relu=False)
    return Model("batch-encoder", "1", (32, 768), b.layers, b.weights, "16-layer encoder over a batch of 32 embeddings")


def build_doc_encoder():
    b = Builder(71)
    for _ in range(15):
        b.dense(768, 768)
    b.dense(768, 768, relu=False)
    return Model("doc-encoder", "1", (256, 768), b.layers, b.weights, "16-layer encoder over a batch of 256 document chunks")


ZOO = {
    "sensor-mlp": build_sensor_mlp,
    "kws-cnn": build_kws_cnn,
    "vision-cnn": build_vision_cnn,
    "batch-encoder": build_batch_encoder,
    "doc-encoder": build_doc_encoder,
}


def default_store_dir():
    return Path(os.environ.get("AIOS_MODEL_DIR", Path(__file__).resolve().parent.parent / "var" / "models"))


class ModelStore:
    def __init__(self, root=None):
        self.root = Path(root or default_store_dir())
        self.root.mkdir(parents=True, exist_ok=True)
        self._models = {}
        self._blobs = {}
        self._lock = threading.Lock()

    def path(self, key):
        return self.root / (key.replace(":", "-") + ".aiosm")

    def ensure_zoo(self):
        for name, build in ZOO.items():
            key = f"{name}:1"
            if not self.path(key).exists():
                self.put(build())
        return self.keys()

    def put(self, model):
        blob = model.to_bytes()
        tmp = self.path(model.key).with_suffix(".tmp")
        tmp.write_bytes(blob)
        tmp.replace(self.path(model.key))
        with self._lock:
            self._models[model.key] = model
            self._blobs[model.key] = blob
        return model.key

    def keys(self):
        return sorted(p.stem.rsplit("-", 1)[0] + ":" + p.stem.rsplit("-", 1)[1] for p in self.root.glob("*.aiosm"))

    def resolve(self, ref):
        if ":" in ref:
            return ref
        versions = [k for k in self.keys() if k.split(":")[0] == ref]
        if not versions:
            raise KeyError(f"unknown model {ref}")
        return max(versions, key=lambda k: int(k.split(":")[1]) if k.split(":")[1].isdigit() else 0)

    def blob(self, ref):
        key = self.resolve(ref)
        with self._lock:
            if key in self._blobs:
                return self._blobs[key]
        blob = self.path(key).read_bytes()
        with self._lock:
            self._blobs[key] = blob
        return blob

    def get(self, ref):
        key = self.resolve(ref)
        with self._lock:
            if key in self._models:
                return self._models[key]
        model = Model.from_bytes(self.blob(key))
        with self._lock:
            self._models[key] = model
        return model
