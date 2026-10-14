"""Portable, read-only observations; GPU-ready never constitutes a reservation.

No torch/CUDA imports, host identifiers, GPU UUIDs or raw stderr enter reports.
A ready observation requires a single visible full idle H100, >=75,000 MiB
observed total and >=70 GiB observed free. These are conservative admission
thresholds, not promises that a future training allocation will succeed.
"""
from dataclasses import dataclass
import csv
import io
import os
import re
import shutil
import subprocess


@dataclass(frozen=True)
class GPUDevice:
    index: int
    name: str
    memory_total_mib: int | None
    memory_free_mib: int | None
    utilization_percent: int | None
    mig_enabled: bool | None
    has_compute_processes: bool | None
    visible: bool | None


@dataclass(frozen=True)
class HardwareReport:
    status: str
    nvidia_smi_status: str
    cpu_count: int | None
    host_memory_total_bytes: int | None
    host_memory_available_bytes: int | None
    disk_free_bytes: int | None
    allocation: str
    devices: tuple[GPUDevice, ...]
    notes: tuple[str, ...]


def _host_resources() -> tuple[int | None, int | None, int | None, int | None]:
    total = available = disk = None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and pages > 0:
            total = page_size * pages
    except (OSError, ValueError, AttributeError):
        pass
    # Linux MemAvailable accounts for reclaimable cache; a free-page count does
    # not represent usable RAM. Other OSes remain explicitly unknown.
    try:
        with open("/proc/meminfo", encoding="ascii") as source:
            for line in source:
                if line.startswith("MemAvailable:"):
                    value = int(line.split()[1])
                    if value >= 0:
                        available = value * 1024
                    break
    except (OSError, ValueError, IndexError):
        pass
    try:
        disk = shutil.disk_usage(os.getcwd()).free
    except OSError:
        pass
    return os.cpu_count(), total, available, disk


def _query(executable: str, field: str) -> str | None:
    try:
        result = subprocess.run(
            [executable, field, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False, shell=False,
        )
        if result.returncode != 0 or len(result.stdout) > 1_000_000:
            return None
        return result.stdout
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return None


def _integer(value: str) -> int | None:
    try:
        number = int(value.strip())
        return number if number >= 0 else None
    except ValueError:
        return None


def _gpu_uuid(value: str) -> str | None:
    """Validate and normalize internal identity; never serialize this value."""
    value = value.strip()
    if re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value):
        return value.lower()
    return None


def inspect_hardware() -> HardwareReport:
    """Observe host and GPUs without changing allocations or invoking a shell.

    GPU process and resource queries are non-atomic snapshots. CUDA ordinals can
    differ from nvidia-smi indices; numeric masks on multiple GPUs therefore stay
    unresolved. UUID/MIG masks are also conservatively unresolved and redacted.
    """
    cpu, total, available, disk = _host_resources()
    notes = ["GPU readiness is an observation, not a reservation."]
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    allocation = "all" if mask is None else "empty" if not mask.strip() else "unresolved"

    def report(status: str, query_status: str, devices: tuple[GPUDevice, ...] = ()) -> HardwareReport:
        return HardwareReport(status, query_status, cpu, total, available, disk,
                              allocation, devices, tuple(notes))

    executable = shutil.which("nvidia-smi")
    if executable is None:
        notes.append("nvidia-smi is absent; CUDA availability is unobserved.")
        return report("cpu_only", "absent")
    raw = _query(executable, "--query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu,mig.mode.current")
    if raw is None:
        notes.append("GPU query failed or timed out; details redacted.")
        return report("unknown", "query_failed")
    process_raw = _query(executable, "--query-compute-apps=gpu_uuid")
    processes = None
    if process_raw is not None:
        rows = [line.strip() for line in process_raw.splitlines() if line.strip()]
        identities = [_gpu_uuid(line) for line in rows]
        if all(identity is not None for identity in identities):
            processes = set(identities)
    devices = []
    parse_failed = process_raw is not None and processes is None
    try:
        rows = list(csv.reader(io.StringIO(raw), skipinitialspace=True, strict=True))
    except csv.Error:
        rows = []
        parse_failed = True
    rows = [row for row in rows if row]
    for row in rows:
        if len(row) != 7:
            parse_failed = True
            continue
        index, total_mib, free_mib, util = (_integer(row[i]) for i in (0, 3, 4, 5))
        if index is None:
            parse_failed = True
            continue
        if (total_mib is not None and total_mib == 0
                or total_mib is not None and free_mib is not None and free_mib > total_mib):
            total_mib = free_mib = None
        if util is not None and util > 100:
            util = None
        mig = {"enabled": True, "disabled": False}.get(row[6].strip().lower())
        # Preserve only product class, never arbitrary driver text/identifiers.
        name = "NVIDIA H100" if "H100" in row[1] else "other GPU"
        if "MIG" in row[1]:
            mig = True
        identity = _gpu_uuid(row[2])
        if identity is None:
            parse_failed = True
        busy = None if processes is None or identity is None else identity in processes
        visible = True if mask is None else False if not mask.strip() else None
        if len(rows) == 1 and mask is not None and mask.strip() == "0" and index == 0:
            allocation = "numeric_single"
            visible = True
        devices.append(GPUDevice(index, name, total_mib, free_mib, util, mig, busy, visible))
    if len({device.index for device in devices}) != len(devices):
        parse_failed = True
    if parse_failed:
        notes.append("Malformed GPU query output; availability unknown.")
        return report("unknown", "malformed", tuple(devices))
    if process_raw is None:
        notes.append("GPU process query failed or timed out; details redacted.")
        return report("unknown", "query_failed", tuple(devices))
    if not devices:
        return report("cpu_only", "ok")
    if allocation == "empty":
        return report("unavailable", "ok", tuple(devices))
    if allocation == "unresolved" or len(devices) != 1:
        notes.append("Single-GPU visibility/allocation could not be established.")
        return report("unknown", "ok", tuple(devices))
    device = devices[0]
    if (device.name != "NVIDIA H100" or device.mig_enabled is True
            or device.has_compute_processes is True
            or device.utilization_percent is not None and device.utilization_percent > 0
            or device.memory_total_mib is not None and device.memory_total_mib < 75_000
            or device.memory_free_mib is not None and device.memory_free_mib < 70 * 1024):
        notes.append("Device does not meet the observed full idle H100 memory envelope.")
        return report("unavailable", "ok", tuple(devices))
    if any(value is None for value in (device.memory_total_mib, device.memory_free_mib,
                                       device.utilization_percent, device.mig_enabled,
                                       device.has_compute_processes, device.visible)):
        notes.append("Required GPU observations are unavailable.")
        return report("unknown", "ok", tuple(devices))
    return report("gpu_ready", "ok", tuple(devices))
