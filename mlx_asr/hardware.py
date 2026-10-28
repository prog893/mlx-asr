"""Machine detection and decode-config defaults.

Two decisions, made on different inputs because they have different causes:

  batch          sized from memory on every machine (``derive_batch``). Batch is
                 memory-bound: on both benchmarked machines end-to-end speed rises
                 with batch until memory runs out, accuracy shifts by about 0.1 point
                 (unresolved on both machines; source: ultra-voxtral-batch.json#c30b32,c30b128;
                 m4-voxtral-batch.json#c60b16,c60b32), and
                 larger batches only speed up files longer than one batch.
  chunk length   per chip, from profiles.json. It is a throughput choice set by
                 how expensive the encoder is on a given GPU, so it has to be
                 measured; chips with no profile fall back to 30s (not-measured: a config value).

## The memory model

Peak MLX memory over the 20-file corpus is linear in rows times chunk seconds, on
top of the weights and a fixed cost:

    M2 Ultra 128GB, 30s, 4-bit: 6.56 / 7.23 / 8.14 / 9.17 / 11.33 GB at B=16/32/48/64/128
                                (source: ultra-voxtral-batch.json#c30b16,c30b32,c30b48,c30b64,c30b128)
                                (0.0014 GB per row-second, 3.5 GB fixed beyond weights)
    M4 16GB,       60s, 4-bit: 6.60 / 7.34 / 8.08 GB at B=16/24/32
                                (source: m4-voxtral-batch.json#c60b16,c60b24,c60b32)
                                (0.0015 GB per row-second, 2.6 GB fixed beyond weights)

(The per-row-second and fixed terms are least-squares fits to these peaks, with the
registry's 2.5GB of 4-bit weights subtracted from the intercept; not-measured: 2.5GB is
the registry's weight constant.)

The Ultra points are not quite linear (B=48 to 64 is steeper than the rest), so the
constants below are set to cover every measured peak rather than to fit the mean:
the tightest is the Ultra at B=64 (9.36GB predicted, 9.17 measured), and the M4 at
B=32 is over-predicted by 16% (9.36 against 8.08; source: ultra-voxtral-batch.json#c30b64;
m4-voxtral-batch.json#c60b32).

Rows are sized on the nominal chunk length, although a batch is padded to its longest
row and a file's last chunk can reach 1.5x the target (not-measured: the splitter's design limit). That padding is inside the fit:
the corpus runs it was fitted on decoded padded batches (worst on the M4 at B=32: 25
rows x 84.1s against 32 x 60s nominal). The worst case itself was measured too: a file
cut so that all 32 rows pad to an 89.4s last chunk peaked at 8.31GB on the M4, under the
9.36 predicted from the nominal length (source: m4-voxtral-worstcase.json#worst32_c60b32;
the 25 x 84.1s and 89.4s row lengths have no saved record). Overlap is not in the fit, so it is added.

The budget is a fraction of the GPU working set, fitted to the one machine where
memory binds: on the M4 (12.7GB working set) B=32 completed the corpus seven times with
no errors at 8.08GB (source: m4-synthetic-decode.json#synth_nvfp4_r1,synth_nvfp4_r2,synth_nvfp4_r3;
m4-voxtral-batch.json#c60b32,c60b32_r1,c60b32_r4). 0.78 (a 9.9GB budget) admits it with 0.55GB to spare
and excludes B=48 (11.1GB predicted) at 60s. B=48 and B=64 did complete the corpus at
60s, four times each, but B=64 on a worst-case padded file peaked at 11.87GB, 93% of the
working set (source: m4-voxtral-batch.json#c60b48,c60b48_r1,c60b48_r2,c60b48_r3,c60b64,c60b64_r1,c60b64_r2,c60b64_r3;
m4-voxtral-worstcase.json#worst64_c60b64), which is too close to the limit to call
reliable. At the M4's 30s chunks the formula admits B=64 (9.36GB predicted), which
has never run there, so the M4 profile caps the batch at the measured 32
(``max_batch``). On the
Ultra (115.4GB, source: ultra-synthetic-decode.json#synth_ultra_4bit_r1) memory does not bind below the largest measured-good batch, 128.
Two machines is a thin fit; a third with a different RAM size is the test.

## Why the result gets snapped to a list

Throughput per step is not monotonic, and two dips reproduce on both machines
measured:

    B = 2..8   worse per step than B=1. On the M4, x-realtime is 6.2 / 6.3 / 6.2
               at B=2/4/8 versus 3.8 at B=1 (source: m4-synthetic-decode.json#synth_nvfp4_r1,synth_nvfp4_r2,synth_nvfp4_r3):
               the batch grows 8x and throughput does not follow. Never default here.
    B = 48     a regression in the synthetic sweep on both machines (M2 Ultra 898
               tok/s at 48 vs 940 at 32; M4 265 vs 322).

B=96 was once excluded as a dip on the Ultra; the synthetic sweep does not show one
(x-realtime 95.9 / 99.7 / 103.9 at B=64/96/128), so it is allowed.

(source: ultra-synthetic-decode.json#synth_ultra_4bit_r1,synth_ultra_4bit_r2,synth_ultra_4bit_r3;
m4-synthetic-decode.json#synth_nvfp4_r1,synth_nvfp4_r2,synth_nvfp4_r3, means of 3 runs)

These look like scheduling artifacts rather than a memory effect, so no formula
over specs will predict them. ``FAST_BATCHES`` encodes the sizes that measured
well, and the largest one that fits is the default.
"""

import functools
import json
import platform
import subprocess
from pathlib import Path

import mlx.core as mx

PROFILES_PATH = Path(__file__).with_name("profiles.json")

# Batch sizes that measured well. Excludes 2-8 (slower per step than 1) and 48
# (a measured regression). 96 sits between 64 and 128 on the Ultra synthetic sweep
# (99.7x between 95.9x and 103.9x). See the module docstring.
# source: ultra-synthetic-decode.json#synth_ultra_4bit_r1,synth_ultra_4bit_r2,synth_ultra_4bit_r3
FAST_BATCHES = (1, 12, 16, 24, 32, 64, 96, 128)

# GB of peak memory per decoded row per second of chunk audio. The two corpus fits
# give 0.0014 (Ultra) and 0.0015 (M4); 0.0018 is what it takes to cover every
# measured peak, including the Ultra's steeper B=48-64 step. See the module docstring.
GB_PER_ROW_SECOND = 0.0018

# Fixed peak beyond the weights: encoder activations, mel buffers, framework
# overhead. The two fits give 3.5 (Ultra) and 2.6 (M4); 3.4 sits just under the
# Ultra fit, and with GB_PER_ROW_SECOND's margin the formula still covers every
# measured peak. See the module docstring.
# source: ultra-voxtral-batch.json#c30b16,c30b32,c30b48,c30b64,c30b128;
# m4-voxtral-batch.json#c60b16,c60b24,c60b32
FIXED_OVERHEAD_GB = 3.4

# Fraction of the GPU working set a decode may plan to use. Fitted on the M4, the
# source: m4-synthetic-decode.json#synth_nvfp4_r1 (working set); m4-voxtral-batch.json#c60b32
# only machine where memory binds: on a 12.7GB working set it admits B=32 (8.08GB
# measured, 9.36 predicted) and excludes B=48 (11.1 predicted). B=64 completed the
# corpus but peaked at 93% of the working set on a worst-case padded file, too close
# to call reliable. See the module docstring.
# source: m4-voxtral-worstcase.json#worst64_c60b64
USABLE_FRACTION = 0.78


def _sh(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def chip_family(chip: str) -> str:
    """"Apple M4 Pro" -> "M4". Generation is what correlates with the quirks."""
    for gen in ("M1", "M2", "M3", "M4", "M5"):
        if gen in (chip or "").upper():
            return gen
    return "unknown"


@functools.lru_cache(maxsize=1)
def machine_info() -> dict:
    """Identify this machine. Cached: the shell-outs cost ~100ms (not-measured: a rough cost)."""
    try:
        gpu_gb = mx.device_info()["max_recommended_working_set_size"] / 1e9
    except Exception:
        gpu_gb = None
    ram = _sh("sysctl -n hw.memsize")
    cores = _sh(
        "system_profiler SPDisplaysDataType "
        "| awk '/Total Number of Cores/{print $NF; exit}'"
    )
    chip = _sh("sysctl -n machdep.cpu.brand_string") or platform.processor()
    return {
        "chip": chip,
        # e.g. "Mac16,1". Recorded because chip and memory do not determine
        # sustained throughput: cooling does, and the chip string cannot see it. A
        # MacBook Air is fanless, a MacBook Pro is not, a Mac Studio has more
        # headroom than either and no battery to throttle for. Two machines with the
        # same chip and RAM can therefore hold different clocks over a long decode,
        # and this benchmark's runs are minutes long.
        #
        # Not part of `match` today, because there is one profile per chip+RAM and no
        # evidence yet that the same pair needs different values per enclosure. If a
        # contributed profile ever disagrees with an existing one at equal chip and
        # memory, this field is what identifies the cause, and `match` can grow a
        # `model_id` key at that point.
        "model_id": _sh("sysctl -n hw.model") or None,
        "chip_family": chip_family(chip),
        # GiB, not decimal GB, so this matches the number on the machine's spec
        # sheet: a 16GB Mac reports 17.18e9 bytes and a 128GB one 137.4e9. Dividing
        # by 1e9 made every profile window a guess about which unit was meant, and
        # the M4 16GB entry only matched because its window happened to span 16-18.
        "ram_gb": round(int(ram) / 2**30) if ram.isdigit() else None,
        "gpu_cores": int(cores) if cores.isdigit() else None,
        "gpu_working_set_gb": round(gpu_gb, 1) if gpu_gb else None,
        "macos": platform.mac_ver()[0],
        "arch": platform.machine(),
        "mlx": mx.__version__,
    }


def snap_batch(want: int) -> int:
    """Largest measured-good batch size not exceeding ``want``."""
    ok = [b for b in FAST_BATCHES if b <= max(want, 1)]
    return max(ok) if ok else 1


def predicted_peak_gb(batch: int, weights_gb: float, chunk_seconds: float) -> float:
    """Peak MLX memory for one decode, from the measured fit in the module docstring."""
    return (weights_gb + FIXED_OVERHEAD_GB
            + GB_PER_ROW_SECOND * max(chunk_seconds, 1.0) * max(batch, 1))


def batch_budget_gb(gpu_gb: float) -> float:
    """Memory a decode may plan to use on a GPU with this working set."""
    return gpu_gb * USABLE_FRACTION


def derive_batch(gpu_gb: float, weights_gb: float, chunk_seconds: float) -> int:
    """Largest measured-good batch whose predicted peak fits the memory budget.

    Memory is the only input: batch size is memory-bound, and more rows than memory
    allows is the failure that matters (an OOM or a swap storm, not a few percent).
    Never returns 2-11 (2-8 measured slower per step than 1, 9-11 unmeasured), so the
    real choice is "12 or more" versus "1".
    """
    budget = batch_budget_gb(gpu_gb)
    fits = [b for b in FAST_BATCHES
            if predicted_peak_gb(b, weights_gb, chunk_seconds) <= budget]
    best = max(fits, default=1)
    return best if best >= 12 else 1


def _load_profiles() -> dict:
    try:
        with open(PROFILES_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"profiles": [], "derived": {}}


def resolve_profile(info: dict | None = None, weights_gb: float = 2.5,
                    chunk_seconds: float | None = None,
                    overlap_seconds: float | None = None) -> dict:
    """Pick a decode config for this machine.

    Batch is always sized from memory (``derive_batch``). ``matched`` says where the
    rest came from ("profile" for a benchmarked chip, "derived" for the defaults), so
    the CLI can tell the user whether chunk length is a measurement or an estimate.
    """
    info = info or machine_info()
    data = _load_profiles()
    chip = (info.get("chip") or "").strip()
    ram = info.get("ram_gb") or 0
    # Batch comes from memory on every path, profiled or not.
    gpu_gb = info.get("gpu_working_set_gb") or 10.0

    for prof in data.get("profiles", []):
        m = prof.get("match", {})
        if m.get("chip") and m["chip"].lower() not in chip.lower():
            continue
        if ram < m.get("ram_gb_min", 0) or ram > m.get("ram_gb_max", 10**9):
            continue
        chunk = chunk_seconds or float(prof["chunk_seconds"])
        overlap = (overlap_seconds if overlap_seconds is not None
                   else prof.get("overlap_seconds", 0.0))
        # every chunk after the first decodes its warm-up overlap too
        batch = derive_batch(gpu_gb, weights_gb, chunk + overlap)
        if prof.get("max_batch"):
            # the largest batch measured reliable here, where the formula admits more
            batch = min(batch, snap_batch(prof["max_batch"]))
        return {
            "batch": batch,
            "chunk_seconds": prof["chunk_seconds"],
            "kv_bits": prof.get("kv_bits"),
            "overlap_seconds": prof.get("overlap_seconds", 0.0),
            "matched": "profile",
            "notes": prof.get("notes", ""),
            "source": prof.get("source", ""),
        }

    # No measured profile: derive chunk length from what can be detected at runtime.
    cores = info.get("gpu_cores") or 0
    derived = data.get("derived", {})
    # Both values are 30s, so this branch is deliberately a no-op.  # not-measured: config values
    # The split was made on the belief that short chunks lose on a low-core GPU,
    # where the encoder is compute-bound. Both measured machines show the reverse:
    # 30s/B32 ran 6.37x against 4.66x for 60s/B16 on the 10-core M4, and 29.3x
    # against 20.4x on the 60-core Ultra, with accuracy tied on both (means of 3 runs
    # except the M4 60s/B16, one run).
    # source: m4-voxtral-chunking.json#c30b32,c30b32_r2,c30b32_r3,c60b16;
    # ultra-voxtral-batch.json#c30b32,c30b32_r2,c30b32_r3;ultra-voxtral-precision.json#vox_w4bit_c60b16_kv8,vox_w4bit_c60b16_kv8_r2;ultra-voxtral-kv.json#vox_kv8_c60b16_w4bit
    # The split is kept so a low-core measurement can change one value without
    # touching this code.
    chunk = chunk_seconds or float(
        derived.get("chunk_seconds_low_core", 30.0) if cores and cores <= 12
        else derived.get("chunk_seconds_default", 30.0)
    )
    batch = derive_batch(gpu_gb, weights_gb, chunk + (overlap_seconds or 0.0))
    return {
        "batch": batch,
        "chunk_seconds": chunk,
        "kv_bits": derived.get("kv_bits", 8),
        "overlap_seconds": 0.0,
        "matched": "derived",
        "notes": (f"no measured profile for {chip or 'this machine'}; chunk length "
                  f"{chunk:.0f}s is the unprofiled default, and batch {batch} is the "
                  f"largest measured-good size that fits {gpu_gb:.1f}GB of GPU "
                  f"working set. Run mlx-asr-bench to contribute a real profile."),
        "source": "",
    }
