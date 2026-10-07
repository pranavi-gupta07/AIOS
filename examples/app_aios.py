import sys
import time

import numpy as np

import aios


def main(frames=60, fps=15):
    rng = np.random.default_rng(1)
    with aios.Client() as client:
        print(f"connected to aiosd over {client.transport}")
        for i in range(frames):
            frame = rng.random((1, 3, 64, 64), dtype=np.float32)
            r = client.infer("vision-cnn", frame, session="camera-0", deadline_ms=80)
            print(f"frame {i:3d}  class {int(r.argmax()[0])}  {r.latency_ms:7.2f} ms  on {r.target}")
            time.sleep(1 / fps)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
