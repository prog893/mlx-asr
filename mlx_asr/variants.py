"""Which size and precision of a family to load: the first priority entry that fits.

Each multi-variant family has a priority list, ordered by measured accuracy on the
20-file corpus. Variants that tie on accuracy are ordered larger first: a tie on 20
files says nothing about material the corpus lacks, and the larger build is where that
risk is smallest. Accuracy decides over speed, so a slower variant that ties stays ahead.

At run time the first entry that fits this machine's memory is loaded, using the same
budget as the batch sizing (``hardware.batch_budget_gb``). A Mac with less memory can
therefore get a smaller variant from the same tie group, or a lower rung, than a Mac
with more: memory decides between variants the measurements cannot separate.

A pinned ``--size`` or ``--quantization`` drops every entry that does not match it, and
the rest is chosen the same way. If nothing left fits, the smallest remaining entry is
used with a warning; a variant the user named is never swapped for an unnamed one.
See docs/DEFAULTS.md, "How a variant is chosen".
"""

from dataclasses import dataclass, replace

from .hardware import batch_budget_gb, derive_batch
from .models import Model, UnknownQuantization, families, resolve_family

# family -> [(size, precision, peak GB)], best first. size "" is a single-size family,
# precision "" one that publishes a single build. Peak is the largest per-run peak over
# the 20-file corpus on the M2 Ultra; Voxtral has None, because its peak depends on the
# batch, so it fits when the memory model still allows a useful batch (12 or more), and
# its last entry always fits, since the batch sizing falls back to 1 for it.
PRIORITY: dict[str, list[tuple[str, str, float | None]]] = {
    # fp16 ties 8bit (+0.11, CI [-0.01, +0.25]) and both beat 4bit (-0.60 and -0.71)
    # at 30s / B128. source: ultra-voxtral-precision-default.json#vox_default_c30b128_kv8,vox_w8bit_c30b128_kv8,vox_wfp16_c30b128_kv8
    "voxtral": [("", "fp16", None), ("", "8bit", None), ("", "4bit", None)],
    # 24B beats 3B by 8.97 points; within a size bf16 ties 8bit; 24B 4bit is worse on
    # English (+1.03, CI [+0.06, +2.87]), 3B 4bit on Japanese (+8.18).
    # source: ultra-voxtral-v1.json#3b_4bit_c30,3b_8bit_c30,3b_bf16_c30,24b_4bit_c30,24b_8bit_c30,24b_bf16_c30
    "voxtral-v1": [("24B", "bf16", 50.08), ("24B", "8bit", 27.92), ("24B", "4bit", 16.27),
                   ("3B", "bf16", 10.91), ("3B", "8bit", 7.28), ("3B", "4bit", 5.25)],
    # large-v3 ties turbo (6 runs each); then by accuracy, medium tying small.
    # source: ultra-whisper-sizes.json#whisper_tiny_libdef,whisper_base_libdef,whisper_small_nocond,whisper_medium_nocond,whisper_large-v2_nocond,whisper_large-v3_nocond,whisper_turbo_nocond
    "whisper": [("large-v3", "", 7.00), ("turbo", "", 5.53), ("large-v2", "", 6.97),
                ("medium", "", 5.43), ("small", "", 4.37), ("base", "", 4.07),
                ("tiny", "", 3.98)],
    # At 15s 8bit is best on both sizes; 1.7B 5bit ties it, 1.7B bf16 ties 6bit; the
    # 0.6B is behind every 1.7B rung; 0.6B bf16 ties 8bit.
    # source: ultra-qwen3-precision-c15.json#qwen3_0.6B_4bit_c15,qwen3_0.6B_5bit_c15,qwen3_0.6B_6bit_c15,qwen3_0.6B_8bit_c15,qwen3_0.6B_bf16_c15,qwen3_1.7B_4bit_c15,qwen3_1.7B_5bit_c15,qwen3_1.7B_6bit_c15,qwen3_1.7B_8bit_c15,qwen3_1.7B_bf16_c15
    "qwen3-asr": [("1.7B", "8bit", 3.74), ("1.7B", "5bit", 3.09), ("1.7B", "bf16", 5.35),
                  ("1.7B", "6bit", 3.31), ("1.7B", "4bit", 2.88), ("0.6B", "bf16", 2.63),
                  ("0.6B", "8bit", 2.07), ("0.6B", "6bit", 1.92), ("0.6B", "5bit", 1.85),
                  ("0.6B", "4bit", 1.77)],
}

USEFUL_BATCH = 12   # below this derive_batch returns 1, which is not worth a larger build


@dataclass(frozen=True)
class Choice:
    model: Model
    size: str
    quant: str
    fits: bool
    reason: str


def _default_quant(m: Model) -> str:
    return next((q for q, r in m.quant_repos.items() if r == m.repo), "")


def _build(family: str, size: str, quant: str) -> Model:
    m = resolve_family(family, size or None)
    if not quant or quant == _default_quant(m):
        return m
    repo = m.repo_for(quant)
    return replace(m, repo=repo, weights_gb=m.weights_gb_for(quant),
                   label=f"{m.label.rsplit(' (', 1)[0]} ({quant})")


def _name(size: str, quant: str) -> str:
    return " ".join(x for x in (size, quant) if x) or "default"


def select(family: str, size: str | None = None, quant: str | None = None,
           gpu_gb: float | None = None, chunk_seconds: float = 30.0) -> Choice:
    """The variant to load for ``family`` on a GPU with ``gpu_gb`` of working set."""
    gpu_gb = gpu_gb or 10.0      # not-measured: the same unknown-GPU fallback as resolve_profile
    budget = batch_budget_gb(gpu_gb)
    entries = PRIORITY.get(family)
    if not entries:              # single-variant families keep the plain lookup
        m = resolve_family(family, size)
        if quant:
            m = _build(family, m.size, quant) if m.quant_repos else _refuse(m, quant)
        return Choice(m, m.size, quant or _default_quant(m), True, "")

    pinned = entries
    if size:
        want = resolve_family(family, size).size     # raises UnknownSize
        pinned = [e for e in pinned if e[0] == want]
    if quant:
        probe = resolve_family(family, pinned[0][0] or None)
        key = probe._quant_key(quant)
        if key not in probe.quant_repos:
            _refuse(probe, quant)
        pinned = [e for e in pinned if e[1] == key]

    def fits(e) -> bool:
        if e[2] is None:
            if e == entries[-1]:
                return True      # the smallest Voxtral build runs at batch 1 if it must
            m = _build(family, e[0], e[1])
            return derive_batch(gpu_gb, m.weights_gb, chunk_seconds) >= USEFUL_BATCH
        return e[2] <= budget

    skipped = []
    for e in pinned:
        if fits(e):
            reason = (f"{_name(e[0], e[1])}: the first of its priority list that fits "
                      f"this machine's {budget:.1f}GB budget")
            if skipped:
                reason += f" (does not fit: {', '.join(skipped)})"
            return Choice(_build(family, e[0], e[1]), e[0], e[1], True, reason)
        skipped.append(_name(e[0], e[1]))
    # Nothing fits: the smallest remaining entry has the best chance of running.
    # (Voxtral has no peak per entry; its list ends with the smallest build.)
    e = pinned[-1] if pinned[-1][2] is None else min(pinned, key=lambda x: x[2])
    return Choice(_build(family, e[0], e[1]), e[0], e[1], False,
                  f"no {family} variant{' matching the pinned flags' if (size or quant) else ''} "
                  f"is predicted to fit this machine's {budget:.1f}GB budget; using "
                  f"{_name(e[0], e[1])}, the smallest, which may swap or run out of memory")


def _refuse(m: Model, quant: str):
    m.repo_for(quant)            # raises UnknownQuantization with the accepted set
    raise UnknownQuantization(quant, m.quant_repos, m.family)


def priority_names(family: str) -> list[str]:
    return [_name(s, q) for s, q, _ in PRIORITY.get(family, [])]


def check_registry() -> list[str]:
    """Every priority entry must name a size and precision the registry publishes."""
    bad = []
    for fam, entries in PRIORITY.items():
        if fam not in families():
            bad.append(f"{fam}: not a family")
            continue
        for s, q, _ in entries:
            try:
                _build(fam, s, q)
            except Exception as exc:     # surfaced by tests/test_variants.py
                bad.append(f"{fam} {_name(s, q)}: {exc}")
    return bad
