"""Machine detection and decode-config defaults.

Two-tier by design, because the two things being decided have different
characters:

  measured profiles   A table, matched on chip + RAM. Throughput is NOT monotonic
                      in batch size and the dips are not predictable from specs,
                      so where a machine has been benchmarked its numbers win
                      outright.
  derived fallback    A function, for machines nobody has benchmarked. It sizes
                      the batch from GPU working set, model footprint and chunk
                      length, then snaps the result onto the batch sizes that are
                      actually fast.

The old fallback was a bucket table keyed on GPU memory, which gave the same
answer for a 16GB M4 and a 36GB M3 Pro and had nothing to say about chunk length
or model size. The formula below is derived from the two full batch sweeps in
scripts/benchmarks/sweep_*.json.

## What the sweeps show

KV memory per decoded row is nearly constant per row-second, and converges as
batch grows (the fixed cost amortizes):

    M2 Ultra 128GB, 60s chunks: 0.152 GB/row at B=16 -> 0.114 at B=128
    M4 16GB,       60s chunks: 0.143 GB/row at B=16 -> 0.120 at B=48

so ~0.002 GB per row-second is a good planning figure, and batch capacity is
(usable memory) / (0.002 * chunk_seconds).

## Why the result gets snapped to a list

Throughput per step is not monotonic, and two dips reproduce on every machine
measured:

    B = 2..8   worse per step than B=1. On the M4, x-realtime is 6.1 / 6.2 / 5.8
               at B=2/4/8 versus 3.6 at B=1: the batch grows 8x and throughput
               does not follow. Never default here.
    B = 48     a real regression on both machines (M2 Ultra 903 tok/s at 48 vs
               945 at 32; M4 251 vs 309). B=96 dips on the Ultra too.

These look like scheduling artifacts rather than a memory effect, so no formula
over specs will predict them. ``FAST_BATCHES`` encodes the sizes that measured
well.

## Why memory alone is the wrong limit

Sizing purely to memory overshoots badly on large machines: 115GB of working set
allows batch 128, but the Ultra sweep only gains 7% from 64 to 128 and dips at 96
on the way. Throughput saturates before memory does, and it saturates as a
function of *GPU cores*:

    M2 Ultra 128GB, 60 cores: 90% of peak throughput first reached at batch 64  (1.1 rows/core)
    M4 16GB,       10 cores: 90% of peak throughput first reached at batch 32  (3.2 rows/core)

Neither machine benefits much past ~2-3 rows per core, so the derived batch is
the smaller of a memory limit and a compute limit. Memory then only binds on
small-memory machines, which is the correct behaviour: that is where getting it
wrong means an OOM rather than a few lost percent.

Generational differences (M2 vs M3 vs M4 at equal memory and cores) still need
measuring: the M4 encoder is compute-bound and eats 36% of wall clock, which
changes the best chunk length in a way core count alone does not reveal. Those
stay in profiles.json. Run `mlx-asr-bench` and open an issue to add one.
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

# GB of KV cache per decoded row per second of chunk audio, from the asymptote of
# both batch sweeps. Rounded up, since running out of memory is worse than
# leaving some unused.
GB_PER_ROW_SECOND = 0.0020

# Fraction of the GPU working set to plan for. MLX peaks above its steady state
# during the encoder pass and the OS wants headroom, so half is the safe share.
# The measured profiles sit at 54% (M4) and 8% (Ultra) of working set, so this
# only binds on small-memory machines, which is exactly where it matters.
USABLE_FRACTION = 0.50

# Non-weight fixed cost: encoder activations, mel buffers, framework overhead.
FIXED_OVERHEAD_GB = 0.6

# Rows per GPU core beyond which throughput stops improving. Both sweeps reach
# 90% of their peak at 1.1 (Ultra) and 3.2 (M4) rows/core, so 3 covers the
# small-GPU case without chasing the flat tail on large ones.
ROWS_PER_GPU_CORE = 3.0

# Assumed core count when detection fails (system_profiler can be slow or
# unavailable). A mid-range GPU, so the compute cap neither dominates nor
# disappears.
DEFAULT_GPU_CORES = 16


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


def derive_batch(gpu_gb: float, weights_gb: float, chunk_seconds: float,
                 gpu_cores: int | None = None) -> int:
    """Batch size for hardware with no measured profile.

    The smaller of two caps, then snapped onto a batch size that measured well:

      memory   how many rows of KV cache fit in the planning budget
      compute  how many rows the GPU can actually keep busy (rows/core)

    Never returns 2-11: that range is the measured valley, so the real choice is
    "12 or more" versus "1".
    """
    usable = max(gpu_gb * USABLE_FRACTION - weights_gb - FIXED_OVERHEAD_GB, 0.0)
    per_row = GB_PER_ROW_SECOND * max(chunk_seconds, 1.0)
    mem_cap = int(usable / per_row) if per_row > 0 else 0
    compute_cap = int(ROWS_PER_GPU_CORE * (gpu_cores or DEFAULT_GPU_CORES))
    want = min(mem_cap, compute_cap)
    if want < 12:
        return 1
    return snap_batch(want)


def _load_profiles() -> dict:
    try:
        with open(PROFILES_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"profiles": [], "derived": {}}


def resolve_profile(info: dict | None = None, weights_gb: float = 2.5,
                    chunk_seconds: float | None = None) -> dict:
    """Pick a decode config for this machine.

    ``matched`` says where the numbers came from ("profile" for a benchmarked
    machine, "derived" for the formula), so the CLI can tell the user whether it
    is reporting a measurement or an estimate. Keeping that honest is the point
    of having both paths.
    """
    info = info or machine_info()
    data = _load_profiles()
    chip = (info.get("chip") or "").strip()
    ram = info.get("ram_gb") or 0

    for prof in data.get("profiles", []):
        m = prof.get("match", {})
        if m.get("chip") and m["chip"].lower() not in chip.lower():
            continue
        if ram < m.get("ram_gb_min", 0) or ram > m.get("ram_gb_max", 10**9):
            continue
        return {
            "batch": prof["batch"],
            "chunk_seconds": prof["chunk_seconds"],
            "kv_bits": prof.get("kv_bits"),
            "overlap_seconds": prof.get("overlap_seconds", 0.0),
            "matched": "profile",
            "notes": prof.get("notes", ""),
            "source": prof.get("source", ""),
        }

    # No measured profile: derive from what can be detected at runtime.
    gpu_gb = info.get("gpu_working_set_gb") or 10.0
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
    batch = derive_batch(gpu_gb, weights_gb, chunk, cores or None)
    return {
        "batch": batch,
        "chunk_seconds": chunk,
        "kv_bits": derived.get("kv_bits", 8),
        "overlap_seconds": 0.0,
        "matched": "derived",
        "notes": (f"no measured profile for {chip or 'this machine'}; batch "
                  f"{batch} derived from {gpu_gb:.1f}GB GPU working set and "
                  f"{cores or DEFAULT_GPU_CORES} GPU cores at {chunk:.0f}s "
                  f"chunks, snapped to a measured-good size. "
                  f"Run mlx-asr-bench to contribute a real profile."),
        "source": "",
    }
