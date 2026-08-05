"""Tests for the two-tier default resolution.

The property worth protecting is that a *measured* machine keeps its measured
numbers no matter what the formula would say, and that the formula never lands in
the batch-size valley. Both are things a well-meaning refactor could break
silently, and the cost would be a large throughput regression that no test
currently catches.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_asr.hardware import (
    FAST_BATCHES,
    chip_family,
    derive_batch,
    machine_info,
    resolve_profile,
    snap_batch,
)

M4 = {"chip": "Apple M4", "ram_gb": 16, "gpu_cores": 10,
      "gpu_working_set_gb": 12.7}
ULTRA = {"chip": "Apple M2 Ultra", "ram_gb": 128, "gpu_cores": 60,
         "gpu_working_set_gb": 115.4}
UNTESTED = {"chip": "Apple M3 Pro", "ram_gb": 36, "gpu_cores": 18,
            "gpu_working_set_gb": 27.0}


def test_measured_machines_use_the_table_not_the_formula():
    for info in (M4, ULTRA):
        p = resolve_profile(info)
        assert p["matched"] == "profile", f"{info['chip']} lost its measured profile"
        assert p["batch"] in FAST_BATCHES


def test_untested_machine_is_derived_and_says_so():
    p = resolve_profile(UNTESTED)
    assert p["matched"] == "derived"
    # the note must tell the user this is an estimate, not a benchmark
    assert "no measured profile" in p["notes"]
    assert "mlx-asr-bench" in p["notes"]


@pytest.mark.parametrize("want", list(range(1, 200)))
def test_snap_never_lands_in_the_measured_valley(want):
    """B=2..11 is slower per step than B=1 on every machine measured."""
    b = snap_batch(want)
    assert b in FAST_BATCHES
    assert not (2 <= b <= 11)
    assert b <= max(want, 1)


def test_derive_never_returns_valley_batches():
    for gpu in (4, 5.3, 8, 12.7, 18, 27, 54, 115.4, 190):
        for cores in (7, 8, 10, 16, 18, 30, 40, 60, 76):
            for chunk in (15, 30, 45, 60, 90):
                b = derive_batch(gpu, 2.5, chunk, cores)
                assert not (2 <= b <= 11), (gpu, cores, chunk, b)


def test_small_memory_machines_fall_back_to_batch_one():
    """Below a useful batch, B=1 beats anything in the valley."""
    assert derive_batch(5.3, 2.5, 60, 7) == 1
    assert derive_batch(4.0, 3.1, 60, 8) == 1


def test_larger_model_gets_a_smaller_batch():
    """A 3.1GB model must not be given the same batch as a 0.5GB one on a
    memory-bound machine."""
    small = derive_batch(12.7, 0.5, 60, 10)
    large = derive_batch(12.7, 5.0, 60, 10)
    assert large <= small


def test_longer_chunks_get_a_smaller_batch():
    """KV cost scales with chunk length, so batch must fall as chunks grow."""
    b30 = derive_batch(12.7, 2.5, 30, 64)
    b120 = derive_batch(12.7, 2.5, 120, 64)
    assert b120 <= b30


def test_compute_cap_binds_on_big_memory_small_gpu():
    """A hypothetical huge-memory, few-core machine must not get a giant batch
    just because the memory allows it."""
    b = derive_batch(190.0, 2.5, 60, gpu_cores=8)
    assert b <= 32


def test_more_cores_allows_a_larger_batch():
    few = derive_batch(115.4, 2.5, 60, 10)
    many = derive_batch(115.4, 2.5, 60, 60)
    assert many > few


def test_missing_core_count_still_resolves():
    p = resolve_profile({"chip": "Apple M9 Ultra", "ram_gb": 64,
                         "gpu_cores": None, "gpu_working_set_gb": 54.0})
    assert p["matched"] == "derived"
    assert p["batch"] in FAST_BATCHES


def test_unknown_machine_with_no_info_at_all():
    p = resolve_profile({"chip": "", "ram_gb": None, "gpu_cores": None,
                         "gpu_working_set_gb": None})
    assert p["batch"] in FAST_BATCHES
    assert p["chunk_seconds"] > 0


@pytest.mark.parametrize("chip,expected", [
    ("Apple M4 Pro", "M4"), ("Apple M2 Ultra", "M2"), ("Apple M1", "M1"),
    ("Apple M3 Max", "M3"), ("Intel Core i9", "unknown"), ("", "unknown"),
])
def test_chip_family(chip, expected):
    assert chip_family(chip) == expected


def test_ram_is_reported_in_nameplate_gib():
    """A 16GB Mac must report 16, not 17.

    `hw.memsize` is bytes, so dividing by 1e9 gives decimal GB: 17.18 for a 16GiB
    machine and 137.4 for a 128GiB one. Profile windows are written against the
    number on the spec sheet, so the unit has to match or every window becomes a
    guess about which convention was meant.
    """
    info = machine_info()
    if info["ram_gb"] is None:
        pytest.skip("sysctl unavailable")
    # nameplate sizes are powers of two times 8 in this product line
    assert info["ram_gb"] in (8, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512), \
        info["ram_gb"]


def test_model_id_is_captured_for_provenance():
    """Cooling class is not derivable from chip and RAM, so the enclosure is
    recorded: a fanless Air and a Mac Studio with the same chip sustain different
    clocks over a minutes-long decode."""
    info = machine_info()
    assert "model_id" in info
    if info["model_id"] is not None:
        assert re.match(r"^[A-Za-z]+\d+,\d+$", info["model_id"]), info["model_id"]
