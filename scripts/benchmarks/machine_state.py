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

import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Load average per CPU, above which the machine is not idle enough to trust a timing
# figure. Deliberately low: a benchmark should have the machine to itself, and the cost
# of a false warning is one glance at the log.
LOAD_PER_CORE_WARN = 0.35

# Resident swap is recorded but deliberately does NOT set `busy`. macOS does not reclaim
# swap once written, so "swap in use" reflects the machine's whole uptime rather than what
# is running now: a host that paged heavily days ago still reports gigabytes while
# completely idle. Gating on it would warn on nearly every run, and a warning that fires
# every time is one nobody reads.
#
# Swap *activity* is different, and it is what this file previously missed. Sampling
# `Swapouts` twice catches a host that is thrashing right now, which is the condition that
# actually destroys a throughput figure. Measured on a 16GB laptop mid-session: 92,484
# swapouts in 5 seconds, about 1.4GB/s, while an idle 128GB host showed exactly 0. That
# machine had 10% memory free and recovered to 63% the moment the competing process died.
# Cheap to detect and unambiguous, so unlike resident swap this one does set `busy`.
SWAPOUT_PAGES_PER_S_WARN = 200.0

# Battery-powered and low-power-mode runs are not comparable to mains-powered ones. A
# laptop on battery reduces sustained clocks, and Low Power Mode does so deliberately, so
# a benchmark taken that way understates the machine by an amount nothing in the result
# file would reveal. Neither is fatal to accuracy, for the same reason contention is not,
# but both make a speed number describe a power policy rather than a config.


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


def _power_state() -> dict:
    """AC vs battery, battery percentage, and Low Power Mode.

    `pmset -g batt` prints "Now drawing from 'AC Power'" or "'Battery Power'" followed by
    a percentage line. `pmset -g` carries `lowpowermode 0|1`. Both are cheap shell reads,
    and neither exists on a Mac Studio in a meaningful sense, so a desktop simply reports
    ac_power True with no battery.
    """
    batt = _sh("pmset -g batt")
    on_ac = "AC Power" in batt
    pct = None
    m = re.search(r"(\d+)%", batt)
    if m:
        pct = int(m.group(1))
    lpm = _sh("pmset -g | awk '/lowpowermode/{print $2}'")
    return {"ac_power": on_ac if batt else None,
            "battery_percent": pct,
            "low_power_mode": (lpm == "1") if lpm else None}


def _swapout_rate(seconds: float = 2.0) -> float | None:
    """Pages swapped OUT per second, from two `vm_stat` samples.

    The resident swap total says what the host did since boot; this says what it is doing
    now. A benchmark host should be at zero. Costs `seconds` of wall clock before the model
    loads, which is worth it: this is the signal that would have caught a thrashing laptop.
    """
    def swapouts():
        raw = _sh("vm_stat | awk '/Swapouts/{print $NF}'")
        try:
            return int(raw.strip().rstrip("."))
        except ValueError:
            return None

    a = swapouts()
    if a is None:
        return None
    time.sleep(seconds)
    b = swapouts()
    if b is None or b < a:          # counter reset
        return None
    return (b - a) / seconds


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
    power = _power_state()
    swap_rate = _swapout_rate()

    reasons = []
    if load is not None and ncpu and load / ncpu > LOAD_PER_CORE_WARN:
        reasons.append(f"load {load:.2f} over {ncpu} cpus")
    if gpu_gb is not None and gpu_gb > GPU_IN_USE_WARN_GB:
        reasons.append(f"{gpu_gb:.1f}GB GPU memory already in use")
    if swap_rate is not None and swap_rate > SWAPOUT_PAGES_PER_S_WARN:
        reasons.append(f"swapping out {swap_rate * 16384 / 1e6:.0f}MB/s "
                       f"({swap_rate:.0f} pages/s)")
    # Power policy is reported as a busy reason because the effect on a timing figure is
    # the same as contention: the machine is not running as fast as it can.
    if power.get("ac_power") is False:
        pct = power.get("battery_percent")
        reasons.append(f"on BATTERY power{f' at {pct}%' if pct is not None else ''}, "
                       f"connect to mains before benchmarking")
    if power.get("low_power_mode"):
        reasons.append("Low Power Mode is ON, which caps clocks deliberately")

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
        # Rate, not total: the total is a property of uptime, the rate is a property of now.
        "swapout_pages_per_s": round(swap_rate, 1) if swap_rate is not None else None,
        "gpu_in_use_gb": round(gpu_gb, 2) if gpu_gb is not None else None,
        **power,
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
