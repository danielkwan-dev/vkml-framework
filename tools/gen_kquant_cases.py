"""Generates tests/data/kquant_cases.json: random blocks of llama.cpp's
k-quant types and the values gguf-py (llama.cpp's own Python package)
dequantizes them to, so vkml's decoding is checked against the reference.

    python tools/gen_kquant_cases.py > tests/data/kquant_cases.json

Needs gguf (pip install gguf) and numpy.
"""

import json
import sys

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf.quants import dequantize

# Block layouts: bytes per block of 256 values, and where the f16 scales sit
# (random bytes there could be NaN or infinity).
LAYOUTS = {
    "q6_K": (T.Q6_K, 210, [208]),  # ql[128] qh[64] scales[16] d
    "q4_K": (T.Q4_K, 144, [0, 2]),  # d dmin scales[12] qs[128]
    "q5_K": (T.Q5_K, 176, [0, 2]),  # d dmin scales[12] qh[32] qs[128]
}


def main():
    rng = np.random.default_rng(7)
    cases = []
    for name, (qtype, size, halves) in LAYOUTS.items():
        blocks = 3
        data = rng.integers(0, 256, size=blocks * size, dtype=np.uint8)
        for b in range(blocks):
            for at in halves:
                scale = np.float16(rng.uniform(0.001, 0.05) * (1 if rng.random() < 0.7 else -1))
                data[b * size + at: b * size + at + 2] = np.frombuffer(scale.tobytes(), dtype=np.uint8)
        values = dequantize(data, qtype).astype(np.float32).reshape(-1)
        cases.append({"type": name, "bytes": data.tobytes().hex(), "values": [float(v) for v in values]})
    json.dump(cases, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
