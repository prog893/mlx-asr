"""Generate the Homebrew resource stanzas for the tap formula.

Homebrew wants one `resource` block per Python dependency, each pinned to a URL
and a sha256. Doing that by hand for ~60 packages is how a tap drifts out of sync
with `pyproject.toml`, so this reads the resolved set and prints the blocks.

Two deliberate differences from `brew update-python-resources`, both forced by
what this project depends on:

* **Wheels, not sdists.** Homebrew's own tooling prefers sdists (`--no-binary
  =:all:`), which cannot work here: `mlx` and `mlx-metal` ship no sdist at all,
  and `numba`/`llvmlite`/`tokenizers` would each need a full LLVM or Rust
  toolchain to build. A personal tap can install wheels; homebrew-core could not
  accept this formula for exactly this reason.
* **Only what is imported.** `mlx-whisper` declares `torch`, which nothing this
  CLI calls actually imports (see the note on the `whisper` extra in
  `pyproject.toml`). Installing it would add ~2.5GB and a CUDA dependency tree
  for dead code, so `SKIP` drops it, along with the nvidia-* wheels it drags in.

    uv run python packaging/gen_formula.py > /tmp/resources.rb
"""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Deliberately empty: everything comes from a pinned wheel.
#
# The obvious candidates were all checked and all rejected, so this is a finding
# rather than laziness:
#   numpy         a formula exists, but Homebrew ships 2.5.1 and numba caps numpy
#                 at <2.5, so the brewed one is an unsatisfiable combination.
#   protobuf      the formula installs C++ libraries only (libprotobuf.dylib and
#                 friends); it has no python3.13 site-packages, so `import
#                 google.protobuf` fails.
#   sentencepiece same: no Python bindings in the formula at all.
#   certifi       does ship bindings, but pulling one package from Homebrew and 61
#                 from wheels buys nothing and adds a second source of version
#                 drift.
BREWED: set[str] = set()

# Declared by a dependency but never imported on any code path this CLI runs.
# torch is the big one: mlx-whisper lists it, only its unused torch_whisper.py
# imports it, and dropping it removes the whole nvidia/triton/cuda subtree.
SKIP = {"torch", "triton", "setuptools", "sounddevice", "miniaudio"}
SKIP_PREFIXES = ("nvidia-", "cuda-")


def resolved_versions(extras):
    """Ask uv to resolve the real dependency set, rather than guessing from
    requires-dist. Runs in a temp project so the local package is not included."""
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    want = list(deps["dependencies"])
    for e in extras:
        want += deps["optional-dependencies"][e]
    out = subprocess.run(
        [sys.executable, "-m", "uv", "pip", "compile", "-", "--no-header",
         "--python-version", "3.13", "--quiet"],
        input="\n".join(want), capture_output=True, text=True, cwd=ROOT,
    )
    if out.returncode:
        sys.exit(f"uv pip compile failed:\n{out.stderr}")
    pins = {}
    for line in out.stdout.splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, _, ver = line.partition("==")
            pins[name.strip().lower()] = ver.strip().split()[0]
    return pins


def wheel_for(name, version, py="cp313"):
    """Pick the wheel a brewed CPython on arm64 macOS would install.

    Order: a pure-python wheel (one file serves every interpreter), then an arm64
    macOS build. For the native case both an interpreter-specific tag (`cp313`)
    and a stable-ABI one (`cp310-abi3`, which a newer CPython also loads) are
    accepted, because several packages here ship only the latter: tokenizers and
    safetensors are cp39/cp310-abi3, hf-xet is cp38-abi3, and mlx-metal is
    py3-none-macosx. Among candidates, take the highest macOS deployment target,
    since the formula requires a recent macOS anyway.
    """
    import urllib.request

    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(url) as r:
        data = json.load(r)
    pure, native = [], []
    for f in data["urls"]:
        if f["packagetype"] != "bdist_wheel":
            continue
        fn = f["filename"]
        if fn.endswith("-py3-none-any.whl") or "-py2.py3-none-any" in fn:
            pure.append(f)
            continue
        if "macosx" not in fn or "arm64" not in fn:
            continue
        # cp313 for this interpreter, abi3 for a stable-ABI wheel a newer
        # CPython still loads, py3-none-macosx for a platform-only wheel.
        if py in fn or "abi3" in fn or "py3-none-macosx" in fn:
            native.append(f)
    if pure:
        return pure[0]
    if native:
        def target(f):                      # macosx_15_0 -> 15
            part = f["filename"].split("macosx_")[1]
            return int(part.split("_")[0])
        return max(native, key=target)
    return None


def main():
    pins = resolved_versions(["whisper", "vad"])
    blocks, skipped, missing = [], [], []
    for name in sorted(pins):
        if name in BREWED or name in SKIP or name.startswith(SKIP_PREFIXES):
            skipped.append(name)
            continue
        f = wheel_for(name, pins[name])
        if f is None:
            missing.append(f"{name}=={pins[name]}")
            continue
        blocks.append(f'  resource "{name}" do\n'
                      f'    url "{f["url"]}"\n'
                      f'    sha256 "{f["digests"]["sha256"]}"\n'
                      f'  end\n')
    print("\n".join(blocks))
    print(f"# {len(blocks)} resources", file=sys.stderr)
    print(f"# brewed or skipped: {' '.join(skipped)}", file=sys.stderr)
    if missing:
        print(f"# NO ARM64 WHEEL (needs attention): {' '.join(missing)}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
