# Homebrew tap

`brew install prog893/tap/mlx-asr` installs from
[prog893/homebrew-tap](https://github.com/prog893/homebrew-tap). This directory
holds the formula source and the generator that keeps its pinned wheels in sync
with `pyproject.toml`.

## Why a tap and not homebrew-core

homebrew-core would reject this formula, for a reason that is not fixable here:
core builds Python dependencies from source (`pip --no-binary=:all:`), and this
dependency tree cannot be built that way.

- `mlx` and `mlx-metal` publish **no sdist at all**, only wheels.
- `numba` and `llvmlite` would each need a full LLVM toolchain, and `tokenizers`
  and `hf-xet` need Rust.

A personal tap can install wheels, so it does. The trade is that the formula
pins 65 exact wheel URLs and has to be regenerated when a dependency moves.

## Regenerating the resource list

```bash
uv run --with uv python packaging/gen_formula.py > /tmp/resources.rb
```

Then replace the `resource` blocks in `mlx-asr.rb`. The generator resolves the
real dependency set with `uv pip compile` rather than reading `requires-dist`, so
it reflects what actually gets installed.

## Findings that the formula encodes

These were each established by trying the obvious thing and watching it fail, so
they are worth keeping rather than rediscovering.

| thing | why the formula does it the other way |
|---|---|
| `venv.pip_install` | `std_pip_args` hard-codes `--no-binary=:all:`, so it tries to compile every wheel. The formula calls pip directly instead. |
| `virtualenv_create(..., without_pip: false)` | raises `ArgumentError` on Python 3.12+. The formula drives the brewed interpreter's pip with `--python=` at the venv, which is what Homebrew's own `do_install` does. |
| installing from `r.cached_download` | Homebrew caches downloads as `<sha256>--name-1.2.3-cp313-....whl`, and pip rejects that filename with "Invalid wheel filename (wrong number of parts)". Each wheel is copied to its original name first. |
| `depends_on "numpy"` | Homebrew ships numpy 2.5.1; `numba` caps numpy at `<2.5`. |
| `depends_on "protobuf"` / `"sentencepiece"` | both formulae install C++ libraries with no python3.13 bindings, so `import google.protobuf` fails. |
| `pip install <project>` | needs `hatchling`, which has no Homebrew formula and cannot be fetched inside the build sandbox. The package is pure Python, so it is copied into site-packages with hand-written entry points. |
| shipping `torch` | `mlx-whisper` declares it, but nothing this CLI calls imports it. Verified: every `whisper-*` alias transcribes with no torch installed. Dropping it removes ~2.5GB and the whole CUDA subtree. |
| shipping `miniaudio` | PyAV decodes in-process and reads strictly more formats. `load_audio_16k` prefers it, with the ffmpeg CLI as a fallback. |
| `tag: "v#{version}"` | `brew style --fix` sorts `url` above `version`, at which point `version` is still empty, so the tag interpolates to a bare `"v"` and the clone fails with `Remote branch v not found in upstream origin`. The tag is written out literally. |
| a `version` line at all | redundant once the tag carries it: `brew audit --strict` flags `version 0.1.0 is redundant with version scanned from URL`. |

`brew audit --strict` still reports one finding, and it is a false positive:
`Add depends_on lines above for "libyaml"` on the `pyyaml` resource. That rule
assumes a source build; the wheel statically bundles libyaml, verified with
`otool -L` on its `_yaml` .so, which links only libSystem.

## Cutting a release

The formula takes its source from `url ... tag:` (a git URL, not a release
tarball) so that there is no tarball checksum to update, and so that it keeps
working while the source repo is private: Homebrew shells out to `git`, which uses
the machine's credentials, whereas `codeload.github.com` would 404 without a
token.

```bash
# 1. bump `version` in the formula, then tag the source repo to match
git tag v0.1.0 && git push origin v0.1.0

# 2. copy the formula into the tap and push
cp packaging/mlx-asr.rb ../homebrew-tap/Formula/mlx-asr.rb

# 3. verify against the real Homebrew, not just by reading it
brew install prog893/tap/mlx-asr
brew test mlx-asr
brew audit --strict --online prog893/tap/mlx-asr   # expect wheel-related warnings
```

`brew audit` will complain about the wheel URLs and the direct pip calls. Those
are the deliberate deviations described above, not oversights.

## Verified

`brew install prog893/tap/mlx-asr` from the published tap, on an M4 running
macOS 26: builds in about 2 minutes to 914MB across 22,430 files, and `brew test`
passes. Exercised through the installed binary, not a source checkout:

- `voxtral` and `whisper-turbo` (the full 1.6GB model) transcribing real audio
- all four output formats, `-f all` writing a 4-file set under a path stem
- `--vad`, confirming onnxruntime is bundled and that its cuts differ from the
  energy-based ones rather than silently falling back
- `--fast`, `--compact-silence` (timestamps land back on the original timeline),
  `--prompt`, `--stats-json`, `--quiet` (stdout byte-empty; the remaining progress
  bars are huggingface_hub writing to stderr)
- an `.m4a` input, which is the ffmpeg fallback path, since the formula ships no
  miniaudio
- `mlx-asr-bench`, and the error paths for a missing file, a non-audio file, no
  arguments, and an unknown repo id

Voxtral's SRT is byte-identical to a pip install of the same commit, which is the
determinism claim holding across two installation methods.

Confirmed absent from the installed tree: `torch`, any `nvidia-*`, `triton`, and
`miniaudio`.

Not verified: any machine other than this one.
