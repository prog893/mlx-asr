"""The memory-bound variant selector (mlx_asr/variants.py).

Working sets are the two benchmarked machines' measured values (M4 16GB 12.7GB, M2
Ultra 128GB 115.4GB) plus two that are not measured, to exercise the fallbacks.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_asr.models import CONVERT_SEP, REGISTRY, UnknownQuantization, UnknownSize  # noqa: E402
from mlx_asr.variants import PRIORITY, check_registry, select  # noqa: E402

M4, ULTRA = 12.7, 115.4      # source: m4-synthetic-decode.json#synth_nvfp4_r1; ultra-synthetic-decode.json#synth_ultra_4bit_r1
MID, TINY = 24.0, 5.3        # not-measured: hypothetical working sets


def pick(family, gpu, size=None, quant=None):
    c = select(family, size, quant, gpu)
    return (c.size, c.quant, c.fits)


def test_every_priority_entry_is_a_published_variant():
    assert check_registry() == []


@pytest.mark.parametrize("family,gpu,want", [
    ("voxtral", ULTRA, ("", "fp16", True)),
    ("voxtral", M4, ("", "8bit", True)),
    ("voxtral", TINY, ("", "4bit", True)),          # batch 1, the existing fallback
    ("voxtral-v1", ULTRA, ("24B", "bf16", True)),
    ("voxtral-v1", MID, ("24B", "4bit", True)),
    ("voxtral-v1", M4, ("3B", "8bit", True)),
    ("whisper", ULTRA, ("large-v3", "", True)),
    ("whisper", M4, ("large-v3", "", True)),
    ("whisper", TINY, ("base", "", True)),
    ("qwen3-asr", ULTRA, ("1.7B", "8bit", True)),
    ("qwen3-asr", M4, ("1.7B", "8bit", True)),
])
def test_the_first_entry_that_fits_is_chosen(family, gpu, want):
    """Accuracy order first, the larger variant first within a tie, then memory."""
    assert pick(family, gpu) == want


def test_a_pinned_size_keeps_only_its_entries():
    assert pick("voxtral-v1", ULTRA, size="3B") == ("3B", "bf16", True)
    assert pick("voxtral-v1", M4, size="3b") == ("3B", "8bit", True)
    assert pick("qwen3-asr", ULTRA, size="0.6B") == ("0.6B", "bf16", True)


def test_a_pinned_precision_keeps_only_its_entries():
    assert pick("voxtral-v1", ULTRA, quant="4bit") == ("24B", "4bit", True)
    assert pick("voxtral-v1", M4, quant="4bit") == ("3B", "4bit", True)
    assert pick("voxtral", ULTRA, quant="4bit") == ("", "4bit", True)
    assert pick("voxtral", ULTRA, quant="none") == ("", "fp16", True)


def test_both_pinned_is_used_as_given_and_warns_when_it_cannot_fit():
    c = select("voxtral-v1", "24B", "bf16", M4)
    assert (c.size, c.quant, c.fits) == ("24B", "bf16", False)
    assert "may swap or run out of memory" in c.reason


def test_a_pinned_size_that_cannot_fit_takes_its_smallest_build():
    """Nothing pinned fits: the smallest remaining entry has the best chance to run."""
    assert pick("voxtral-v1", M4, size="24B") == ("24B", "4bit", False)


def test_unpublished_pins_are_errors_not_fallbacks():
    with pytest.raises(UnknownSize):
        select("voxtral-v1", "7B", None, ULTRA)
    with pytest.raises(UnknownQuantization):
        select("voxtral", None, "6bit", ULTRA)
    with pytest.raises(UnknownQuantization):
        select("whisper", None, "4bit", ULTRA)


def test_the_chosen_build_carries_its_own_weight_size():
    """derive_batch subtracts weights_gb, so fp16 must not be sized as 4bit."""
    fp16 = select("voxtral", None, None, ULTRA).model
    assert fp16.repo.endswith("-fp16") and fp16.weights_gb == 8.9  # not-measured: registry weight constant
    q8 = select("voxtral", None, None, M4).model
    assert q8.repo == REGISTRY["voxtral"].quant_repos["fp16"] + CONVERT_SEP + "8bit"


def test_voxtral_8bit_converts_with_the_recipe_of_its_saved_runs():
    """The saved 8bit runs used `mlx_audio.convert --quantize --q-mode affine --q-bits 8
    --q-group-size 64` on the fp16 hub build; first use must convert the same way."""
    from mlx_asr.convert import REALTIME_RECIPES, realtime_weights

    assert REALTIME_RECIPES["8bit"] == {"q_mode": "affine", "q_bits": 8, "q_group_size": 64}
    assert realtime_weights("mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit") == \
        "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"


def test_ties_put_the_larger_variant_first():
    """Spot checks of the order the saved runs justify (see variants.PRIORITY)."""
    assert PRIORITY["whisper"][0][0] == "large-v3" and PRIORITY["whisper"][1][0] == "turbo"
    assert [e[1] for e in PRIORITY["voxtral"]] == ["fp16", "8bit", "4bit"]
    assert PRIORITY["voxtral-v1"][0][:2] == ("24B", "bf16")


def test_the_docs_priority_column_matches_the_code():
    """docs/MODELS.md prints each family's priority list (gen_model_matrix.py); a
    drifted order would describe a selection the CLI does not make."""
    import re

    doc = (Path(__file__).resolve().parents[1] / "docs" / "MODELS.md").read_text()
    for fam, entries in PRIORITY.items():
        block = doc.split(f"### `--model {fam}`")[1].split("\n### ")[0]
        ranked = {}
        for line in block.splitlines():
            cells = [c.strip(" `*") for c in line.strip("|").split("|")]
            if line.startswith("| ") and cells[0].isdigit():
                size = cells[1] if len(cells) == 6 else ""
                quant = cells[2] if len(cells) == 6 else cells[1]
                ranked[int(cells[0])] = (size, "" if quant == "-" else quant)
        assert [ranked[i] for i in sorted(ranked)] == [e[:2] for e in entries], fam
        assert not re.search(r"\*\*default\*\*", block), fam
