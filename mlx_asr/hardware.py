"""Machine detection and decode-config defaults.

Two decisions, made on different inputs because they have different causes:

  batch          sized from memory on every machine (``derive_batch``). Batch is
                 memory-bound: on both benchmarked machines end-to-end speed rises
                 with batch until memory runs out, accuracy does not move, and
                 larger batches only speed up files longer than one batch.
  chunk length   per chip, from profiles.json. It is a throughput choice set by
                 how expensive the encoder is on a given GPU, so it has to be
                 measured; chips with no profile fall back to 60s.

## The memory model

Peak MLX memory over the 20-file corpus is linear in rows times chunk seconds, on
top of the weights and a fixed cost:

    M2 Ultra 128GB, 30s, 4-bit: 6.56 / 7.23 / 8.14 / 9.17 / 11.33 GB at B=16/32/48/64/128
                                (0.0014 GB per row-second, 3.4 GB fixed beyond weights)
    M4 16GB,       60s, 4-bit: 6.60 / 7.34 / 8.08 GB at B=16/24/32
                                (0.0015 GB per row-second, 2.6 GB fixed beyond weights)

The Ultra points are not quite linear (B=48 to 64 is steeper than the rest), so the
constants below are set to cover every measured peak rather than to fit the mean:
the tightest is the Ultra at B=64 (9.36GB predicted, 9.17 measured), and the M4 at
B=32 is over-predicted by 16% (9.36 against 8.08).

The budget is a fraction of the GPU working set, fitted to the one machine where
memory binds: on the M4 (12.7GB working set) B=32 completed the corpus twice with
no errors at 8.08GB, while B=48 is past the ~8.4GB wall where synthetic decode
throughput collapses. 0.78 (a 9.9GB budget) admits the former with 0.55GB to spare
and excludes the latter (11.1GB predicted). On the
Ultra (115.4GB) memory does not bind below the largest measured-good batch, 128.
Two machines is a thin fit; a third with a different RAM size is the test.

## Why the result gets snapped to a list

Throughput per step is not monotonic, and two dips reproduce on every machine
measured:

    B = 2..8   worse per step than B=1. On the M4, x-realtime is 6.1 / 6.2 / 5.8
               at B=2/4/8 versus 3.6 at B=1: the batch grows 8x and throughput
               does not follow. Never default here.
    B = 48     a regression in the synthetic sweep on both machines (M2 Ultra 903
               tok/s at 48 vs 945 at 32; M4 251 vs 309). B=96 dips on the Ultra too.

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
# and 96 (measured regressions). See the module docstring.
FAST_BATCHES = (1, 12, 16, 24, 32, 64, 128)

# GB of peak memory per decoded row per second of chunk audio. The two corpus fits
# give 0.0014 (Ultra) and 0.0015 (M4); 0.0018 is what it takes to cover every
# measured peak, including the Ultra's steeper B=48-64 step. See the module docstring.
GB_PER_ROW_SECOND = 0.0018

# Fixed peak beyond the weights: encoder activations, mel buffers, framework
# overhead. Rounded up from the two fits (3.4 Ultra, 2.6 M4).
FIXED_OVERHEAD_GB = 3.4

# Fraction of the GPU working set a decode may plan to use. Fitted on the M4, the
# only machine where memory binds: on a 12.7GB working set it admits B=32 (8.08GB
# measured, 9.36 predicted) and excludes B=48 (11.1 predicted).
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
    """Identify this machine. Cached: the shell-outs cost ~100ms."""
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
    Never returns 2-11: that range is the measured valley, so the real choice is
    "12 or more" versus "1".
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
                    chunk_seconds: float | None = None) -> dict:
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
        return {
            "batch": derive_batch(gpu_gb, weights_gb, chunk),
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
    # Both values are currently 60s, so this branch is deliberately a no-op: on a
    # low-core GPU the compute-bound encoder dominates wall clock, so shorter
    # chunks add encoder work without buying decode speed, but that is measured
    # only on the M4 and one machine is not enough to justify a different default.
    # The split is kept so a second low-core measurement can change one value
    # without touching this code.
    chunk = chunk_seconds or float(
        derived.get("chunk_seconds_low_core", 60.0) if cores and cores <= 12
        else derived.get("chunk_seconds_default", 60.0)
    )
    batch = derive_batch(gpu_gb, weights_gb, chunk)
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
