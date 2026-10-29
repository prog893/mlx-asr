"""Convert a transformers Whisper checkpoint to MLX format, without torch.

Why this exists: `--model kotoba` needs MLX weights, and the authors of
kotoba-whisper publish transformers weights only.

Why it is cheap: the two formats hold the same numbers. `mlx-examples`' converter
imports torch to read the checkpoint and to build the model, but neither is
necessary. safetensors is a plain tensor container that `mx.load` reads directly,
and the rest of the work is renaming keys (`self_attn` -> `attn`, `fc1` -> `mlp1`,
and so on) plus one axis swap on the conv weights, because torch stores Conv1d as
(out, in, k) and MLX expects (out, k, in). No matrix is recomputed, so this is not
"reimplementing a converter", it is a dictionary rewrite.

The result is written to the HF cache so it happens once per machine.
"""

import json
from pathlib import Path

import mlx.core as mx

# transformers key -> mlx-whisper key. Order matters: `.self_attn_layer_norm`
# must be rewritten before `.self_attn`, or the prefix match mangles it.
_KEY_REWRITES = (
    ("model.", ""),
    (".layers", ".blocks"),
    (".self_attn_layer_norm", ".attn_ln"),
    (".self_attn", ".attn"),
    (".encoder_attn_layer_norm", ".cross_attn_ln"),
    (".encoder_attn.", ".cross_attn."),
    (".final_layer_norm", ".mlp_ln"),
    (".q_proj", ".query"),
    (".k_proj", ".key"),
    (".v_proj", ".value"),
    (".out_proj", ".out"),
    (".fc1", ".mlp1"),
    (".fc2", ".mlp2"),
    ("embed_positions.weight", "positional_embedding"),
    ("decoder.embed_tokens", "decoder.token_embedding"),
    ("encoder.layer_norm", "encoder.ln_post"),
    ("decoder.layer_norm", "decoder.ln"),
)

# config.json field -> ModelDimensions field, which is what mlx-whisper's loader
# splats into a dataclass. A transformers config carries dozens of other keys and
# passing them through is exactly the TypeError that `chunked._is_mlx_format`
# exists to avoid.
_CONFIG_MAP = {
    "n_mels": "num_mel_bins",
    "n_audio_ctx": "max_source_positions",
    "n_audio_state": "d_model",
    "n_audio_head": "encoder_attention_heads",
    "n_audio_layer": "encoder_layers",
    "n_vocab": "vocab_size",
    "n_text_ctx": "max_target_positions",
    "n_text_state": "d_model",
    "n_text_head": "decoder_attention_heads",
    "n_text_layer": "decoder_layers",
}


def _rewrite_key(key: str) -> str:
    for old, new in _KEY_REWRITES:
        key = key.replace(old, new)
    return key


def convert_to_mlx(src: str, dst: str | Path, dtype=mx.float16, log=print) -> Path:
    """Write an MLX-format copy of a transformers Whisper checkpoint.

    ``src`` is a local directory or an HF repo id; ``dst`` is where the converted
    `config.json` and `weights.safetensors` go. Returns ``dst``.
    """
    dst = Path(dst)
    path = Path(src)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        log(f"[convert] fetching {src}")
        path = Path(snapshot_download(
            src, allow_patterns=["model.safetensors", "config.json"]))

    cfg_src = json.loads((path / "config.json").read_text())
    missing = [v for v in _CONFIG_MAP.values() if v not in cfg_src]
    if missing:
        raise ValueError(
            f"{src} does not look like a transformers Whisper checkpoint "
            f"(config.json lacks {', '.join(missing)})")
    cfg = {k: cfg_src[v] for k, v in _CONFIG_MAP.items()}
    cfg["model_type"] = "whisper"

    weights = mx.load(str(path / "model.safetensors"))
    log(f"[convert] {len(weights)} tensors, {cfg['n_audio_layer']}-layer encoder / "
        f"{cfg['n_text_layer']}-layer decoder")

    out = {}
    for key, value in weights.items():
        # Tied to the token embedding in mlx-whisper, and passing it would be an
        # unexpected weight.
        if key == "proj_out.weight":
            continue
        key = _rewrite_key(key)
        # torch Conv1d is (out, in, kernel); MLX is (out, kernel, in).
        if "conv" in key and value.ndim == 3:
            value = value.swapaxes(1, 2)
        out[key] = value.astype(dtype)

    # mlx-whisper computes these from the config rather than loading them.
    out.pop("encoder.positional_embedding", None)

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    mx.save_safetensors(str(dst / "weights.safetensors"), out)
    log(f"[convert] wrote {dst}")
    return dst


def cached_mlx_copy(repo: str, log=print) -> str:
    """Return a path to an MLX copy of ``repo``, converting on first use.

    Converted models live beside the downloaded ones under the HF cache, so this
    costs one conversion per machine rather than one per run. Returns a path
    suitable for `--model`.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    dst = Path(HF_HUB_CACHE) / "mlx-asr-converted" / repo.replace("/", "--")
    if (dst / "weights.safetensors").exists() and (dst / "config.json").exists():
        return str(dst)
    log(f"[convert] no MLX build of {repo} is published; converting once "
        f"(this is a key rename, not a re-quantization)")
    convert_to_mlx(repo, dst, log=log)
    return str(dst)


# Voxtral Realtime precisions converted on first use, with the exact recipe of the
# saved runs that measured them (mlx_audio.convert from the fp16 hub build).
# source: ultra-voxtral-precision-default.json#vox_w8bit_c30b128_kv8
REALTIME_RECIPES = {"8bit": {"q_mode": "affine", "q_bits": 8, "q_group_size": 64}}


def realtime_weights(repo: str, log=print) -> str:
    """A loadable path for a Voxtral Realtime registry value.

    A plain repo id is returned as is. `<fp16 repo>:<precision>` is quantized once into
    `<HF cache>/mlx-asr-converted/` and reused, because no loadable build of that
    precision is published (the hub's 8bit and 6bit repos crash mlx-audio's loader).
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    from .models import CONVERT_SEP

    if CONVERT_SEP not in repo:
        return repo
    source, precision = repo.rsplit(CONVERT_SEP, 1)
    if precision not in REALTIME_RECIPES:
        raise ValueError(f"no conversion recipe for {precision!r}; "
                         f"known: {', '.join(REALTIME_RECIPES)}")
    dst = (Path(HF_HUB_CACHE) / "mlx-asr-converted"
           / f"{source.replace('/', '--')}-{precision}")
    if (dst / "config.json").exists() and list(dst.glob("*.safetensors")):
        return str(dst)
    log(f"[convert] no loadable {precision} build of {source} is published; "
        f"quantizing once into {dst}")
    from mlx_audio.convert import convert

    # A half-written directory from an interrupted run must not pass the check above.
    tmp = dst.with_name(dst.name + ".partial")
    convert(hf_path=source, mlx_path=str(tmp), quantize=True,
            **REALTIME_RECIPES[precision])
    tmp.rename(dst)
    log(f"[convert] wrote {dst}")
    return str(dst)
