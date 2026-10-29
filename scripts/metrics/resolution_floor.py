"""The smallest corpus-level difference the per-file spread lets a paired comparison resolve.

    .venv/bin/python scripts/metrics/resolution_floor.py

Several pages say how large a difference must be before n files can resolve it. This is the
one computation behind that sentence: the per-file paired differences between two saved
configs (repeat runs averaged per file first), their standard deviation, and the half-width
of a 95% interval on the mean of n such files, 1.96 * SD / sqrt(n), for the corpus sizes the
docs use. The pair is the shipped Voxtral default against Whisper turbo without
conditioning (6 runs), the comparison on which this corpus has the most repeat data.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import archive_query as q  # noqa: E402

PAIR = (("ultra-voxtral-headline",
         ["vox_default_c30b128_kv8", "vox_default_c30b128_kv8_r2", "vox_default_c30b128_kv8_r3"]),
        ("ultra-whisper-sizes",
         ["whisper_turbo_nocond"] + [f"whisper_turbo_nocond_r{i}" for i in range(2, 7)]))
SIZES = (5, 7, 17, 20)


def per_file_mean(group: str, runs: list[str], unit: str) -> dict:
    acc = {}
    for r in q.select(q.load_group(group), runs):
        for f, row in q.valid_rows(r, unit).items():
            acc.setdefault(f, []).append(row["coverage_cer"])
    return {f: float(np.mean(v)) for f, v in acc.items()}


def main():
    (ga, ra), (gb, rb) = PAIR
    for unit in ("char", "word"):
        a, b = per_file_mean(ga, ra, unit), per_file_mean(gb, rb, unit)
        d = np.array([(a[f] - b[f]) * 100 for f in sorted(set(a) & set(b))])
        if len(d) < 2:
            continue
        sd = float(d.std(ddof=1))
        floors = ", ".join(f"n={n}: {1.96 * sd / np.sqrt(n):.2f}" for n in SIZES)
        print(f"{unit}: {len(d)} files, mean paired difference {d.mean():+.2f} points, "
              f"per-file SD {sd:.2f}; 95% half-width {floors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
