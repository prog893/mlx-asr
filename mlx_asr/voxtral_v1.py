"""Voxtral v1 (Mini 3B and Small 24B, both 2507), on MLX, without torch.

These are the first Voxtral generation: a Whisper-large-v3-shaped audio encoder feeding
a Mistral language model through a small projector, prompted in "transcription mode".
The `voxtral` family is the second generation (Realtime 4B), a different architecture
with its own streaming decoder, so the two do not share a code path.

mlx-audio has carried a loader for these weights since 0.4, and this module uses its
model classes for the forward pass. It does NOT use `Model.generate`, for three reasons,
each of which fails quietly or only on some installs:

1. **The prompt needs torch and soundfile.** `generate` builds its input with
   transformers' `VoxtralProcessor.apply_transcription_request`, which refuses any
   `return_tensors` other than `"pt"`, and mistral_common's `Audio` asserts soundfile is
   installed before it will hold an array. Neither package is a dependency here, and the
   Homebrew formula exists because nothing needs torch. So the prompt is built from
   mistral_common's pure pieces (`build_prompt`) and pinned against its
   `encode_transcription` by a test wherever soundfile happens to be available.

2. **`max_tokens` is per call, default 128.** Handed a whole file, the transcript stops
   after about a minute of speech and nothing says so. The same class of budget bug
   truncated Qwen3-ASR (see backends.TOKENS_PER_SECOND), so this drives its own window
   loop with a per-window budget.

3. **mistralai's repos ship two copies of the weights**, `consolidated.safetensors`
   (Mistral-native key names) beside HF-format shards, and mlx-audio's loader prefers the
   consolidated file, whose keys this model's `sanitize` does not map. Downloading only
   the shards also halves the download (48GB instead of 96GB for Small 24B).

Weights come from the authors' own repos, never a third-party conversion. bf16 loads
straight from the downloaded shards; a quantized precision is converted once per machine
into the HF cache beside kotoba's conversion, using mlx-audio's own converter, which
leaves the audio encoder unquantized (`model_quant_predicate`).
"""

import glob
import json
import math
from pathlib import Path

import numpy as np

from .models import CONVERT_SEP

SAMPLE_RATE = 16000
# The processor's `pad_to_multiple_of`, and the encoder's receptive field: every window
# is padded to a whole number of 30s chunks, each encoded independently.
CHUNK_SAMPLES = 480_000
# Mel frames per 30s chunk, the processor's `max_source_positions`.
FRAMES_PER_CHUNK = 3000

# Same tokens as the whole-file default in mlx-audio, which ends on any of them.
_EOS_TOKEN_IDS = (2, 4, 32000)


# Only the files a load needs, and never `consolidated*` (see module docstring).
_ALLOW_PATTERNS = ["*.json", "*.safetensors"]
_IGNORE_PATTERNS = ["consolidated*"]

# Precision -> (bits, group size) for a conversion. Affine, group 64: mlx-audio's and
# mlx-lm's defaults, and what every quantized build this project has measured used.
QUANT_BITS = {"8bit": (8, 64), "6bit": (6, 64), "4bit": (4, 64)}


def split_repo(repo: str) -> tuple[str, str | None]:
    """`mistralai/X:4bit` -> ("mistralai/X", "4bit"); a plain id or path -> (it, None)."""
    if CONVERT_SEP in repo and not Path(repo).exists():
        source, precision = repo.rsplit(CONVERT_SEP, 1)
        return source, precision
    return repo, None


def _download(source: str) -> Path:
    """Fetch the HF-format shards and processor files only, or use a local directory."""
    if Path(source).is_dir():
        return Path(source)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(source, allow_patterns=_ALLOW_PATTERNS,
                                  ignore_patterns=_IGNORE_PATTERNS))


def _staged(snapshot: Path) -> Path:
    """A directory holding the snapshot minus any `consolidated*` file.

    Needed only for the converter, which globs every `*.safetensors` it is pointed at and
    prefers a consolidated file if one is there. A snapshot this module downloaded never
    has one, but a user's cache might (anything else that fetched the whole repo), and
    the converter would then silently read the other key layout. Symlinks, so no copy.
    """
    stage = snapshot.parent / (snapshot.name + ".mlx-asr-shards")
    stage.mkdir(exist_ok=True)
    for f in snapshot.iterdir():
        if f.name.startswith("consolidated"):
            continue
        link = stage / f.name
        if not link.exists():
            link.symlink_to(f.resolve())
    return stage


def weights_dir(repo: str, log=print) -> Path:
    """A local directory with loadable weights for a registry repo value.

    bf16 (a plain source id) is the downloaded snapshot itself. A `source:precision`
    value is converted once into `<HF cache>/mlx-asr-converted/`, beside kotoba's
    conversion, and reused from then on.
    """
    source, precision = split_repo(repo)
    if precision is None:
        return _download(source)
    if precision not in QUANT_BITS:
        raise ValueError(f"no conversion recipe for {precision!r}; "
                         f"known: {', '.join(QUANT_BITS)}")
    from huggingface_hub.constants import HF_HUB_CACHE

    dst = (Path(HF_HUB_CACHE) / "mlx-asr-converted"
           / f"{source.replace('/', '--')}-{precision}")
    if (dst / "config.json").exists() and glob.glob(str(dst / "*.safetensors")):
        return dst
    bits, group = QUANT_BITS[precision]
    log(f"[convert] no {precision} MLX build of {source} is published by its "
        f"authors; quantizing once into {dst} (audio encoder stays bf16)")
    snapshot = _download(source)
    from mlx_audio.convert import convert

    # A half-written directory from an interrupted run must not pass the check above.
    tmp = dst.with_name(dst.name + ".partial")
    convert(hf_path=str(_staged(snapshot)), mlx_path=str(tmp), quantize=True,
            q_bits=bits, q_group_size=group, dtype="bfloat16")
    tmp.rename(dst)
    log(f"[convert] wrote {dst}")
    return dst


class Loaded:
    """Model, tokenizer and feature extractor for one set of weights."""

    def __init__(self, model, tokenizer, features, path: Path):
        self.model, self.tokenizer, self.features, self.path = (
            model, tokenizer, features, path)


def load(repo: str, log=print) -> Loaded:
    """Build the model from `config.json` and load only the HF-format shards.

    Mirrors mlx-audio's `base_load_model` step for step (config, sanitize, quantize from
    the config, strict load, eval) but skips its `post_load_hook`, which constructs a
    transformers `AutoProcessor` that nothing here uses.
    """
    import mlx.core as mx
    from mlx_audio.stt.models.voxtral.voxtral import Model, ModelConfig
    from mlx_audio.utils import apply_quantization
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from transformers import WhisperFeatureExtractor

    path = weights_dir(repo, log)
    config = json.loads((path / "config.json").read_text())
    model = Model(ModelConfig.from_dict(config))

    weights = {}
    for f in sorted(glob.glob(str(path / "*.safetensors"))):
        if Path(f).name.startswith("consolidated"):
            continue
        weights.update(mx.load(f))
    weights = model.sanitize(weights)
    apply_quantization(model, config, weights, model.model_quant_predicate)
    # strict: a key-layout mismatch is a crash here, not a model of random weights.
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()

    tokenizer = MistralTokenizer.from_file(str(path / "tekken.json"))
    # transformers' NumPy path, the exact extractor the processor wraps; no torch.
    features = WhisperFeatureExtractor.from_pretrained(str(path))
    return Loaded(model, tokenizer, features, path)


def build_prompt(tokenizer, n_samples: int, language: str | None) -> list[int]:
    """Token ids for one transcription request over `n_samples` of (padded) audio.

    `[BOS][INST][BEGIN_AUDIO][AUDIO]*n[/INST]lang:xx[TRANSCRIBE]`, which is what
    mistral_common's `_encode_instruct_transcription` emits for a single audio chunk. It
    is assembled here from the same objects (the audio encoder's own token count, the
    tokenizer's own special ids) because the public entry point requires soundfile.
    `tests/test_voxtral_v1.py` checks the two agree token for token.
    """
    instruct = tokenizer.instruct_tokenizer
    audio_tokens = instruct.audio_encoder._encode_audio_tokens(n_samples)
    tokens = [*instruct.start(), instruct.BEGIN_INST, *audio_tokens, instruct.END_INST]
    if language:
        tokens += instruct.tokenizer.encode(f"lang:{language}", bos=False, eos=False)
    tokens.append(instruct.TRANSCRIBE)
    return tokens


def input_features(extractor, window: np.ndarray):
    """Log-mel for one window, shaped (n_chunks, 3000, 128) for the encoder.

    Pads to a whole number of 30s chunks first and extracts once over the padded
    window, then splits into chunks, which is the processor's `_retrieve_input_features`
    order. It matters: Whisper's normalisation clamps at the maximum over the whole
    input minus 8, so extracting per chunk would change every value.
    """
    import mlx.core as mx

    padded_len = math.ceil(max(len(window), 1) / CHUNK_SAMPLES) * CHUNK_SAMPLES
    padded = np.zeros(padded_len, dtype=np.float32)
    padded[:len(window)] = window
    feats = extractor(padded, sampling_rate=SAMPLE_RATE, return_tensors="np",
                      padding=False, truncation=False)["input_features"][0]
    n_mels = feats.shape[0]
    chunks = feats.reshape(n_mels, -1, FRAMES_PER_CHUNK).transpose(1, 0, 2)
    return mx.array(chunks).transpose(0, 2, 1), padded_len


def transcribe_window(loaded: Loaded, window: np.ndarray, language: str | None,
                      max_tokens: int) -> str:
    """Greedy decode of one window. Stops at EOS or after `max_tokens` new tokens."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    feats, padded_len = input_features(loaded.features, window)
    ids = mx.array([build_prompt(loaded.tokenizer, padded_len, language)])
    embeds = loaded.model._merge_input_embeddings(input_ids=ids,
                                                  input_features=feats)[0]
    out = []
    for token, _ in generate_step(prompt=mx.array([]), input_embeddings=embeds,
                                  model=loaded.model.language_model,
                                  max_tokens=max_tokens,
                                  sampler=lambda logits: mx.argmax(logits, axis=-1)):
        token = int(token)
        if token in _EOS_TOKEN_IDS:
            break
        out.append(token)
    mx.clear_cache()
    return loaded.tokenizer.decode(out).strip()
