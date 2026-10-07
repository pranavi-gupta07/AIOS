import time

import numpy as np

import aios

STEPS = [
    ("baseline", {}),
    ("another app saturates the NPU", {"target": "local-npu", "load": 0.8}),
    ("edge Wi-Fi degrades", {"target": "edge", "link": {"rtt_ms": 70, "bandwidth_mbps": 6}}),
    ("edge server goes down", {"target": "edge", "online": False}),
    ("edge back, NPU free again", {"target": "edge", "online": True, "link": {"rtt_ms": 6, "bandwidth_mbps": 200}}),
    ("NPU free", {"target": "local-npu", "load": 0.0}),
    ("battery drops to 10%", {"device": {"battery_level": 0.10}}),
    ("charger connected", {"device": {"battery_level": 0.10, "charging": True}}),
]


def main(frames_per_step=20):
    rng = np.random.default_rng(5)
    frame = rng.random((1, 3, 64, 64), dtype=np.float32)
    with aios.Client() as c:
        for label, change in STEPS:
            if change:
                change = dict(change)
                target = change.pop("target", None)
                link = change.pop("link", None)
                device = change.pop("device", None)
                c.control(target, link=link, device=device, **change)
                time.sleep(0.4)
            seen = {}
            lat = []
            for _ in range(frames_per_step):
                r = c.infer("vision-cnn", frame, session="demo", deadline_ms=80)
                seen[r.target] = seen.get(r.target, 0) + 1
                lat.append(r.latency_ms)
                time.sleep(1 / 15)
            print(f"{label:<32} mean {np.mean(lat):7.2f} ms  p95 {np.percentile(lat, 95):7.2f} ms  placements {seen}")
        c.control(device={"battery_level": 0.9, "charging": False})


if __name__ == "__main__":
    main()
