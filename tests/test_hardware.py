"""Tests for default resolution.

The properties worth protecting: batch is sized from memory on every machine (so a
RAM size, not a chip name, decides it), the memory model stays above every peak
actually measured, and the result never lands in the batch-size valley. A
well-meaning refactor could break any of these silently, and the cost would be an
out-of-memory failure or a large throughput regression.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_asr.hardware import (
    FAST_BATCHES,
    batch_budget_gb,
    chip_family,
    derive_batch,
    machine_info,
    predicted_peak_gb,
    resolve_profile,
    snap_batch,
)

M4 = {"chip": "Apple M4", "ram_gb": 16, "gpu_cores": 10,
      "gpu_working_set_gb": 12.7}
ULTRA = {"chip": "Apple M2 Ultra", "ram_gb": 128, "gpu_cores": 60,
         "gpu_working_set_gb": 115.4}
UNTESTED = {"chip": "Apple M3 Pro", "ram_gb": 36, "gpu_cores": 18,
            "gpu_working_set_gb": 27.0}


def test_measured_machines_keep_their_chunk_length_from_the_table():
    for info, chunk in ((M4, 60), (ULTRA, 30)):
        p = resolve_profile(info)
        assert p["matched"] == "profile", f"{info['chip']} lost its measured profile"
        assert p["chunk_seconds"] == chunk


def test_batch_follows_memory_on_the_benchmarked_machines():
    """The two batch sweeps: B32 is the largest that ran reliably on the M4 16GB at
    60s, and the Ultra 128GB kept speeding up to B128 at 30s."""
    assert resolve_profile(M4)["batch"] == 32
    assert resolve_profile(ULTRA)["batch"] == 128


def test_batch_depends_on_memory_not_chip_name():
    """Same chip and profile, different GPU working set: the batch must follow the
    memory. Both machines keep their own chunk length while doing so."""
    assert resolve_profile({**ULTRA, "gpu_working_set_gb": 12.7})["batch"] < 128
    assert resolve_profile({**M4, "gpu_working_set_gb": 115.4})["batch"] == 128


@pytest.mark.parametrize("batch,chunk,measured", [
    (16, 30, 6.56), (32, 30, 7.23), (48, 30, 8.14), (64, 30, 9.17), (128, 30, 11.33),
    (16, 60, 6.60), (24, 60, 7.34), (32, 60, 8.08),
])
def test_memory_model_is_at_or_above_every_measured_peak(batch, chunk, measured):
    """Voxtral 4-bit (2.5GB weights) peaks over the 20-file corpus, M2 Ultra at 30s
    and M4 at 60s (docs/benchmarks/chunking.md). Under-predicting is the dangerous
    direction: it would hand a machine a batch it cannot hold."""
    assert predicted_peak_gb(batch, 2.5, chunk) >= measured


def test_overlap_counts_toward_the_decoded_chunk_length():
    """Each chunk after the first decodes its warm-up overlap as well, so 60s chunks
    with 30s of overlap must be sized as 90s rows (B32 there is predicted at 11.1GB,
    over the M4's 9.9GB budget)."""
    assert resolve_profile(M4)["batch"] == 32
    assert resolve_profile(M4, overlap_seconds=30)["batch"] == 24
    assert resolve_profile(UNTESTED, overlap_seconds=30)["batch"] <= \
        resolve_profile(UNTESTED)["batch"]


def test_nominal_sizing_covers_the_measured_padded_worst_case():
    """A batch pads to its longest row, and the last chunk can reach 1.5x the target.
    On the M4 a 32-row batch padded to an 89.4s last chunk (99% of the worst case)
    peaked at 8.31GB; sizing on the nominal 60s must still predict at least that, and
    the M4 must still resolve to the batch that was measured."""
    assert predicted_peak_gb(32, 2.5, 60) >= 8.31
    assert resolve_profile(M4)["batch"] == 32


def test_m4_budget_admits_b32_and_excludes_b48():
    budget = batch_budget_gb(12.7)
    assert predicted_peak_gb(32, 2.5, 60) <= budget
    assert predicted_peak_gb(48, 2.5, 60) > budget


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
                b = derive_batch(gpu, 2.5, chunk)
                assert not (2 <= b <= 11), (gpu, cores, chunk, b)


def test_small_memory_machines_fall_back_to_batch_one():
    """Below a useful batch, B=1 beats anything in the valley."""
    assert derive_batch(5.3, 2.5, 60) == 1
    assert derive_batch(4.0, 3.1, 60) == 1


def test_larger_model_gets_a_smaller_batch():
    """A 3.1GB model must not be given the same batch as a 0.5GB one on a
    memory-bound machine."""
    small = derive_batch(12.7, 0.5, 60)
    large = derive_batch(12.7, 5.0, 60)
    assert large <= small


def test_longer_chunks_get_a_smaller_batch():
    """KV cost scales with chunk length, so batch must fall as chunks grow."""
    b30 = derive_batch(12.7, 2.5, 30)
    b120 = derive_batch(12.7, 2.5, 120)
    assert b120 <= b30


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
