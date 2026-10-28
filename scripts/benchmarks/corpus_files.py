"""`--files` for the corpus runners: rerun an saved config on a few named files.

Every runner scores files independently (batches form inside a file), so a run on a
canary and a new file can be merged into an saved run (archive_results.py
--supplement) without rerunning the rest of the corpus.
"""

import sys


def add_files_option(parser):
    parser.add_argument("--files", nargs="+", metavar="STEM",
                        help="only these corpus stems (to add one file to an saved run)")


def only_files(pairs, stems):
    """The (audio, reference) pairs whose audio stem is in `stems`; all of them if empty."""
    if not stems:
        return pairs
    unknown = set(stems) - {audio.stem for audio, _ in pairs}
    if unknown:
        # the stems are private corpus names, so give a count, not the names
        print(f"--files: {len(unknown)} stem(s) not in the corpus", file=sys.stderr)
        raise SystemExit(2)
    return [(audio, ref) for audio, ref in pairs if audio.stem in stems]
