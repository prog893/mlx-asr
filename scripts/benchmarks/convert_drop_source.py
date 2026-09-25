"""Quantize a Voxtral v1 source with mlx-audio's own steps, deleting the source before saving.

For a host too full to hold the bf16 shards and the quantized output at once. Small 24B at
8bit is 48GB of shards plus 27GB of output, and mlx-audio's `convert` reads the shards
lazily while it writes, so both have to be on disk together.

This runs the same calls as `mlx_audio.convert.convert(quantize=True, dtype="bfloat16")`
in the same order, and differs in one place only: it evaluates the quantized model into
memory, then (with --drop-source) deletes the downloaded snapshot, then saves. Evaluation
order does not change what MLX computes, so the output should equal a normal conversion
bit for bit. `--compare DIR` checks exactly that against a conversion made the normal way,
and should be run on a small model first.

    python scripts/benchmarks/convert_drop_source.py \\
        --source mistralai/Voxtral-Small-24B-2507 --precision 8bit --drop-source

Writes to the same cache path `mlx_asr.voxtral_v1.weights_dir` uses, so the CLI and the
benchmark pick the result up as an ordinary cached conversion.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from mlx_asr import voxtral_v1  # noqa: E402


def converted_path(source: str, precision: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE
    return (Path(HF_HUB_CACHE) / "mlx-asr-converted"
            / f"{source.replace('/', '--')}-{precision}")


def convert(source: str, precision: str, out: Path, drop_source: bool):
    from mlx_audio.convert import (MODEL_CONVERSION_DTYPES, build_quant_predicate,
                                   copy_model_files, detect_model_domain,
                                   get_model_class, get_model_type, load_config,
                                   load_weights)
    from mlx_lm.utils import quantize_model, save_config, save_model

    bits, group = voxtral_v1.QUANT_BITS[precision]
    snapshot = voxtral_v1._download(source)
    model_path = voxtral_v1._staged(snapshot)

    # --- mlx_audio.convert.convert, step for step ---
    config = load_config(model_path)
    domain = detect_model_domain(config, model_path)
    model_type = get_model_type(config, model_path, domain)
    model_class = get_model_class(model_type, domain)
    model_config = model_class.ModelConfig.from_dict(config)
    if hasattr(model_config, "model_path"):
        model_config.model_path = model_path
    weights = load_weights(model_path)
    model = model_class.Model(model_config)
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()))
    weights = dict(tree_flatten(model.parameters()))
    target_dtype = "bfloat16"
    assert target_dtype in MODEL_CONVERSION_DTYPES
    weights = {k: v.astype(getattr(mx, target_dtype)) for k, v in weights.items()}
    predicate = build_quant_predicate(model, None)
    model.load_weights(list(weights.items()))
    model, config = quantize_model(model, config, group, bits, mode="affine",
                                   quant_predicate=predicate)
    # --- the one difference: materialise before the source can go away ---
    mx.eval(model.parameters())
    del weights

    tmp = out.with_name(out.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    copy_model_files(model_path, tmp)
    if drop_source:
        repo_dir = snapshot.parents[1]            # .../models--owner--name
        assert repo_dir.name.startswith("models--"), repo_dir
        shutil.rmtree(repo_dir)
        print(f"[convert] dropped source {repo_dir}")
    save_model(tmp, model, donate_model=True)
    config["model_type"] = model_type
    save_config(config, config_path=tmp / "config.json")
    tmp.rename(out)
    print(f"[convert] wrote {out}")


def compare(a: Path, b: Path) -> bool:
    import json

    def tensors(d):
        out = {}
        for f in sorted(d.glob("*.safetensors")):
            out.update(mx.load(str(f)))
        return out

    ta, tb = tensors(a), tensors(b)
    ok = set(ta) == set(tb)
    print(f"keys: {len(ta)} vs {len(tb)}, {'same set' if ok else 'DIFFERENT sets'}")
    diffs = [k for k in sorted(set(ta) & set(tb))
             if ta[k].dtype != tb[k].dtype or ta[k].shape != tb[k].shape
             or not mx.array_equal(ta[k], tb[k]).item()]
    print(f"tensors differing: {len(diffs)}" + (f" (first: {diffs[:3]})" if diffs else ""))
    ca = json.loads((a / "config.json").read_text())
    cb = json.loads((b / "config.json").read_text())
    same_cfg = ca == cb
    print(f"config.json: {'identical' if same_cfg else 'DIFFERENT'}")
    return ok and not diffs and same_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--precision", required=True, choices=sorted(voxtral_v1.QUANT_BITS))
    p.add_argument("--out", help="default: the cache path the CLI reads")
    p.add_argument("--drop-source", action="store_true")
    p.add_argument("--compare", help="a normally converted directory to check against")
    a = p.parse_args()

    out = Path(a.out) if a.out else converted_path(a.source, a.precision)
    if out.exists():
        print(f"{out} already exists", file=sys.stderr)
        return 2
    convert(a.source, a.precision, out, a.drop_source)
    if a.compare:
        return 0 if compare(out, Path(a.compare)) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
