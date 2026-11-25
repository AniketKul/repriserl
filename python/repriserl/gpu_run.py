"""Observe compute-process occupancy throughout one single-device experiment."""
from contextlib import contextmanager
from dataclasses import dataclass
import os
import subprocess
from threading import Event, Thread
from typing import Iterator

from .hardware import inspect_hardware


def _other_processes() -> bool | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        pids = [int(row.strip()) for row in output.splitlines() if row.strip()]
        if any(pid <= 0 for pid in pids):
            return None
        return any(pid != os.getpid() for pid in pids)
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


def check_device(allow_shared: bool) -> None:
    report = inspect_hardware()
    if report.nvidia_smi_status != "ok" or len(report.devices) != 1:
        raise RuntimeError("cannot verify a single H100")
    device, = report.devices
    if (device.name != "NVIDIA H100" or device.visible is not True or device.mig_enabled is not False
            or (device.memory_total_mib or 0) < 75_000 or (device.memory_free_mib or 0) < 70*1024
            or device.utilization_percent is None or device.has_compute_processes is None):
        raise RuntimeError("device does not meet the full H100 memory and visibility requirements")
    other = _other_processes()
    if other is None or (not allow_shared and (other or device.utilization_percent != 0)):
        raise RuntimeError("device occupancy prevents an exclusive H100 run")


@dataclass
class Occupancy:
    samples: int = 0
    shared: bool = False
    unknown: bool = False

    def sample(self) -> None:
        value = _other_processes()
        self.samples += 1
        self.shared |= value is True
        self.unknown |= value is None

    @property
    def label(self) -> str:
        if self.shared:
            return "shared_h100"
        if self.unknown or not self.samples:
            return "unverified"
        return "sampled_uncontended_h100"


@contextmanager
def observe_run(allow_shared: bool) -> Iterator[Occupancy]:
    check_device(allow_shared)
    occupancy, stop = Occupancy(), Event()
    occupancy.sample()

    def poll():
        while not stop.wait(1):
            occupancy.sample()

    monitor = Thread(target=poll, name="gpu-occupancy", daemon=True)
    monitor.start()
    try:
        yield occupancy
    finally:
        stop.set()
        monitor.join()
        occupancy.sample()
