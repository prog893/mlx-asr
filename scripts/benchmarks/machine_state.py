"""Machine identity and load, recorded alongside any throughput number.

Exists because of one specific failure. The 20-file headline run was made on a host
that was generating video on the same GPU throughout, and measured ~2.6x realtime
where the same config reaches ~31x idle. Nothing in the output said so, so the JSON
was indistinguishable from a clean run and the whole session had to be voided once
the contention was remembered rather than read.

An x-realtime figure is a property of a machine in a state, not of a config. A result
file that carries the config but not the state is therefore under-specified in the one
dimension that silently ruins it, which is why this is recorded automatically rather
than left to the person running the benchmark to note down.

Accuracy is unaffected: decoding is greedy, so contention costs wall clock only. That
asymmetry is why `busy` is a warning here rather than a refusal. The call on whether a
loaded run is usable belongs to whoever reads it, and they can only make it if the
number is on the page.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Load average per CPU, above which the machine is not idle enough to trust a timing
# figure. Deliberately low: a benchmark should have the machine to itself, and the cost
# of a false warning is one glance at the log.
LOAD_PER_CORE_WARN = 0.35

# Swap is recorded but deliberately does NOT set `busy`. macOS does not reclaim swap
# once written, so "swap in use" reflects the machine's whole uptime rather than what
# is running now: a host that paged heavily days ago still reports gigabytes while
# completely idle. Gating on it would warn on nearly every run, and a warning that
# fires every time is one nobody reads. What would be diagnostic is swap *activity*
# during the run, which needs two samples over the decode rather than one before it;
# the figure is recorded so a suspicious timing can be checked against it by hand.


def _sh(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _load_average() -> float | None:
    """1-minute load average. `sysctl vm.loadavg` prints `{ 2.27 4.18 4.17 }`."""
    raw = _sh("sysctl -n vm.loadavg")
    try:
        return float(raw.strip("{} ").split()[0])
    except (IndexError, ValueError):
        return None


def _swap_used_mb() -> float | None:
    """Swap in use, MB. `total = 19456.00M used = 17742.88M free = 1713.12M`."""
    raw = _sh("sysctl -n vm.swapusage")
    try:
        used = raw.split("used =")[1].split()[0]
        return float(used.rstrip("M"))
    except (IndexError, ValueError):
        return None


def _gpu_in_use_gb() -> float | None:
    """GPU memory in use machine-wide, GB, from IOKit.

    This is the signal that would have caught the voided run. Load average sees CPU
    work, and a GPU-bound video generator can leave load low while owning the device
    this benchmark needs. Read before the model loads, so anything reported here
    belongs to some other process.
    """
    raw = _sh('ioreg -r -d 1 -w 0 -c AGXAccelerator '
              '| grep -o \'"In use system memory"=[0-9]*\' | head -1')
    try:
        return int(raw.split("=")[1]) / 1e9
    except (IndexError, ValueError):
        return None


# GPU memory another process may hold before its work is assumed to matter. Some
# resident use is normal (WindowServer and every open window), so this is well above
# idle desktop use and well below a model.
GPU_IN_USE_WARN_GB = 3.0


def machine_state() -> dict:
    """Identity plus current load, for embedding in a result file.

    Identity comes from `mlx_asr.hardware.machine_info` so a benchmark labels a
    machine exactly the way the CLI and the hardware profiles do, rather than
    inventing a second naming scheme that has to be reconciled later.
    """
    try:
        from mlx_asr.hardware import machine_info
        info = machine_info()
    except Exception:
        info = {}

    load = _load_average()
    swap_mb = _swap_used_mb()
    cores = info.get("gpu_cores") or 0
    ncpu = _sh("sysctl -n hw.ncpu")
    ncpu = int(ncpu) if ncpu.isdigit() else None

    gpu_gb = _gpu_in_use_gb()

    reasons = []
    if load is not None and ncpu and load / ncpu > LOAD_PER_CORE_WARN:
        reasons.append(f"load {load:.2f} over {ncpu} cpus")
    if gpu_gb is not None and gpu_gb > GPU_IN_USE_WARN_GB:
        reasons.append(f"{gpu_gb:.1f}GB GPU memory already in use")

    return {
        "label": machine_label(info),
        "chip": info.get("chip"),
        "model_id": info.get("model_id"),
        "ram_gb": info.get("ram_gb"),
        "gpu_cores": cores or None,
        "macos": info.get("macos"),
        "mlx": info.get("mlx"),
        "load_1min": load,
        "cpu_count": ncpu,
        "swap_used_mb": swap_mb,
        "gpu_in_use_gb": round(gpu_gb, 2) if gpu_gb is not None else None,
        # True means "do not trust the timings in this file". Accuracy is still
        # valid; greedy decoding does not care what else is running.
        "busy": bool(reasons),
        "busy_reasons": reasons,
    }


def machine_label(info: dict) -> str:
    """"Apple M2 Ultra 128GB (Mac14,14)". Chip, nameplate GiB, then enclosure.

    The enclosure is part of the label because cooling, not chip and memory, sets
    sustained throughput over a decode that runs for minutes.
    """
    chip = (info.get("chip") or "unknown chip").strip()
    ram = info.get("ram_gb")
    model = info.get("model_id")
    out = f"{chip} {ram}GB" if ram else chip
    return f"{out} ({model})" if model else out


def warn_if_busy(state: dict, log=print) -> bool:
    """Print a warning naming what else is running. Returns True if busy.

    Printed at the start of a run as well as recorded in the JSON, because the
    person who launched it can still stop and come back to an idle machine, which
    is cheaper than discovering the contention after the fact and voiding a session.
    """
    if not state.get("busy"):
        return False
    log(f"[machine] WARNING: {state['label']} is not idle "
        f"({'; '.join(state['busy_reasons'])}). Accuracy is unaffected (greedy "
        f"decode), but every x-realtime figure from this run is unreliable. "
        f"Consider stopping and re-running on an idle machine.")
    return True


if __name__ == "__main__":
    import json
    s = machine_state()
    warn_if_busy(s)
    print(json.dumps(s, indent=2))
