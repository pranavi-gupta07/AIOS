import sys

import numpy as np

from aios import kdev, protocol


def main(device=kdev.DEFAULT_DEVICE):
    if not kdev.available(device):
        print(f"{device} is not available: build and load kernel/aios.ko, then start aiosd", file=sys.stderr)
        return 1
    with kdev.KernelChannel(device) as ch:
        st = ch.stats()
        if st["daemon_pid"] <= 0:
            print("no placement daemon registered with the kernel broker", file=sys.stderr)
            return 1
        x = np.random.default_rng(3).random((1, 64), dtype=np.float32)
        blob = protocol.encode({"op": "infer", "model": "sensor-mlp", "requirements": {"objective": "latency"}}, {"x": x})
        req_id = ch.submit(blob)
        status, reply_blob = ch.wait(req_id, timeout_ms=10000)
        reply, arrays = protocol.decode(reply_blob)
        if status != 0 or not reply.get("ok"):
            print(f"request {req_id} failed: {reply.get('error')}", file=sys.stderr)
            return 1
        meta = reply["meta"]
        print(f"kernel request {req_id}: ran on {meta['target']} in {meta['latency_ms']} ms, "
              f"top class {int(np.argmax(arrays['y']))}")
        print(f"broker stats: {ch.stats()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
