"""Sweep charts for docs/benchmarks/engines/kotoba.md, copied from its tables.

The page's other two charts (engines-kotoba-window, engines-kotoba-authors-chunk) are
defined in sweeps/engines.py.
"""

SWEEPS = {
    "kotoba-sequential": {
        "doc": "docs/benchmarks/engines/kotoba.md",
        "title": "kotoba under mlx-whisper's sequential driver",
        "basis": "5 JP files (7-file subset) and the narration clip, M2 Ultra, means of 3 runs (own conversion with conditioning on the corpus: 2; a third hung). No default: "
                 "every config scores above 44%, so the CLI uses the chunked driver.",
        "xlabel": "weights and conditioning",
        "ylabel": "CER % (lower is better)",
        "scale": "category", "connect": False,
        "series": ["corpus coverage CER (5 files)", "narration CER"],
        "rows": [
            ("npz port", "| third-party npz port |", ("91.31%", "90.58%")),
            ("npz port\nno-condition", "| same, `--no-condition` | 53.26% |",
             ("53.26%", None)),
            ("own conversion", "| official weights, own MLX conversion |",
             ("93.83%", "86.69%")),
            ("own conversion\nno-condition", "| same, `--no-condition` | 54.32% |",
             ("54.32%", "44.31%")),
        ],
        "source": "ultra-kotoba-sequential.json#kseq_npz_cond_r1,kseq_npz_cond_r2,kseq_npz_cond_r3,kseq_npz_nocond_r1,kseq_npz_nocond_r2,kseq_npz_nocond_r3,kseq_off_cond_r1,kseq_off_cond_r2,kseq_off_nocond_r1,kseq_off_nocond_r2,kseq_off_nocond_r3,kseqclip_npz_cond_r1,kseqclip_npz_cond_r2,kseqclip_npz_cond_r3,kseqclip_off_cond_r1,kseqclip_off_cond_r2,kseqclip_off_cond_r3,kseqclip_off_nocond_r1,kseqclip_off_nocond_r2,kseqclip_off_nocond_r3",
        "chosen": None,
        "width": 7.6,
        "zero": True,
    },
}
