"""Copies an HF checkpoint's first layers into a smaller checkpoint, for checks.

    python tools/slice_checkpoint.py --model <dir> --layers 4 --out <dir>

A 3B model does not fit transformers in f32 on a 16 GB machine; its first
few layers, with the embeddings, final norm and output projection, do. The
slice is a model of its own (its outputs mean little), but vkml and
transformers must agree on it exactly, which checks every architectural
detail on the real weights. Tensors are streamed one at a time; the other
files (tokenizer, templates) are copied as they are. Needs safetensors and
torch.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

# Per-layer lists in config.json, cut to the layers kept.
PER_LAYER = ("layer_types", "no_rope_layers", "sliding_layers")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    src, out = Path(args.model), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
            shutil.copy(f, out / f.name)

    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", config)  # image-text models nest the text model's
    text["num_hidden_layers"] = args.layers
    for key in PER_LAYER:
        if isinstance(text.get(key), list):
            text[key] = text[key][: args.layers]
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    layer = re.compile(r"\.layers\.(\d+)\.")
    tensors = {}
    for shard in sorted(src.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():
                m = layer.search(name)
                if m is None or int(m.group(1)) < args.layers:
                    tensors[name] = f.get_tensor(name)
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    print(f"{len(tensors)} tensors, {args.layers} layers, in {out}")


if __name__ == "__main__":
    main()
