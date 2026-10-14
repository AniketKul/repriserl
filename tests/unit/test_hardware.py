"""Only executable discovery/subprocess boundaries are mocked."""
import json
import subprocess
from dataclasses import asdict

import pytest
from repriserl import hardware


def probe(monkeypatch, gpu_rows, process_rows="", mask=None):
    monkeypatch.setattr(hardware.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)

    def run(args, **kwargs):
        assert isinstance(args, list)
        assert kwargs.get("shell", False) is False
        assert 0 < kwargs["timeout"] <= 5
        output = process_rows if any("query-compute-apps" in a for a in args) else gpu_rows
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(hardware.subprocess, "run", run)
    return hardware.inspect_hardware()


def test_no_cuda_is_reported(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda _: None)
    report = hardware.inspect_hardware()
    assert report.status == "cpu_only"
    assert report.cpu_count is None or report.cpu_count > 0
    assert report.devices == ()
    assert report.nvidia_smi_status == "absent"
    json.dumps(asdict(report))


def test_full_idle_h100_is_observed_ready(monkeypatch):
    report = probe(monkeypatch, "0, NVIDIA H100 80GB HBM3, GPU-11111111-1111-1111-1111-111111111111, 81559, 80000, 0, Disabled\n")
    assert report.status == "gpu_ready"
    assert report.devices[0].memory_total_mib == 81559
    assert "reservation" in " ".join(report.notes)
    assert "GPU-11111111-1111-1111-1111-111111111111" not in json.dumps(asdict(report))


@pytest.mark.parametrize("row,apps", [
    ("0, NVIDIA H100 MIG, GPU-11111111-1111-1111-1111-111111111111, 10000, 9999, 0, Enabled", ""),
    ("0, NVIDIA H100, GPU-11111111-1111-1111-1111-111111111111, 81559, 80000, 40, Disabled", ""),
    ("0, NVIDIA H100, GPU-11111111-1111-1111-1111-111111111111, 81559, 80000, 0, Disabled", "GPU-11111111-1111-1111-1111-111111111111\n"),
    ("0, NVIDIA H100, GPU-11111111-1111-1111-1111-111111111111, 81559, 20000, 0, Disabled", ""),
    ("0, NVIDIA A100, GPU-11111111-1111-1111-1111-111111111111, 81559, 80000, 0, Disabled", ""),
])
def test_mig_and_busy_device_are_not_full_free_h100(monkeypatch, row, apps):
    assert probe(monkeypatch, row, apps).status == "unavailable"


@pytest.mark.parametrize("row", [
    "garbage", "0, NVIDIA H100, GPU-11111111-1111-1111-1111-111111111111, N/A, N/A, N/A, N/A",
    "0, NVIDIA H100, GPU-11111111-1111-1111-1111-111111111111, 81559, 80000, 0, N/A",
    "0, NVIDIA H100, GPU-11111111-1111-1111-1111-111111111111, 81559, 90000, 0, Disabled",
    "0, NVIDIA H100, GPU-11111111-1111-1111-1111-111111111111, 81559, 80000, .nan, Disabled",
])
def test_unknown_observations_never_become_ready(monkeypatch, row):
    assert probe(monkeypatch, row).status == "unknown"


@pytest.mark.parametrize("mask,status", [("", "unavailable"), ("1", "unknown"), ("GPU-11111111-1111-1111-1111-111111111111", "unknown"), ("0", "gpu_ready")])
def test_visibility_mask_limits_readiness(monkeypatch, mask, status):
    report = probe(monkeypatch, "0, NVIDIA H100, GPU-11111111-1111-1111-1111-111111111111, 81559, 80000, 0, Disabled", mask=mask)
    assert report.status == status
    assert "GPU-11111111-1111-1111-1111-111111111111" not in json.dumps(asdict(report))


@pytest.mark.parametrize("failure", [
    subprocess.TimeoutExpired("nvidia-smi", 5, stderr="secret-host"),
    subprocess.CalledProcessError(1, "nvidia-smi", stderr="secret-host"),
    OSError("secret-host"),
])
def test_query_failure_is_sanitized(monkeypatch, failure):
    monkeypatch.setattr(hardware.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    def run(*args, **kwargs):
        raise failure
    monkeypatch.setattr(hardware.subprocess, "run", run)
    report = hardware.inspect_hardware()
    assert report.status == "unknown"
    assert report.nvidia_smi_status == "query_failed"
    assert "secret-host" not in json.dumps(asdict(report))


def test_process_query_failure_blocks_readiness(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    def run(args, **kwargs):
        if any("query-compute-apps" in a for a in args):
            return subprocess.CompletedProcess(args, 1, "", "private-host")
        return subprocess.CompletedProcess(args, 0, "0, NVIDIA H100, GPU-11111111-1111-1111-1111-111111111111, 81559, 80000, 0, Disabled", "")
    monkeypatch.setattr(hardware.subprocess, "run", run)
    assert hardware.inspect_hardware().status == "unknown"


@pytest.mark.parametrize("mask", [None, "0", "1", "0,1"])
def test_multiple_devices_are_not_a_single_gpu_allocation(monkeypatch, mask):
    rows = "0, NVIDIA H100, GPU-11111111-1111-1111-1111-111111111111, 81559, 80000, 0, Disabled\n1, NVIDIA H100, GPU-22222222-2222-2222-2222-222222222222, 81559, 80000, 0, Disabled"
    assert probe(monkeypatch, rows, mask=mask).status == "unknown"


def test_unclosed_csv_quote_is_malformed(monkeypatch):
    assert probe(monkeypatch, '0, NVIDIA H100, GPU-11111111-1111-1111-1111-111111111111, 81559, 80000, 0, "Disabled').nvidia_smi_status == "malformed"


def test_no_gpu_rows_do_not_invent_a_device(monkeypatch):
    report = probe(monkeypatch, "")
    assert report.status == "cpu_only"
    assert report.devices == ()


def test_host_resource_failures_preserve_unknowns(monkeypatch):
    def unavailable(*args):
        raise OSError("private-host")
    monkeypatch.setattr(hardware.os, "sysconf", unavailable)
    monkeypatch.setattr(hardware.shutil, "disk_usage", unavailable)
    monkeypatch.setattr(hardware.shutil, "which", lambda _: None)
    report = hardware.inspect_hardware()
    assert report.host_memory_total_bytes is None
    assert report.disk_free_bytes is None
    assert "private-host" not in json.dumps(asdict(report))


@pytest.mark.parametrize("device_uuid,process_uuid", [
    ("GPU-", "GPU-nonsense"),
    ("GPU-nonsense", ""),
    ("GPU-11111111-1111-1111-1111-111111111111", "GPU-nonsense"),
    ("GPU-11111111-1111-1111-1111-11111111111z", ""),
])
def test_malformed_uuid_records_are_unknown(monkeypatch, device_uuid, process_uuid):
    report = probe(monkeypatch, f"0, NVIDIA H100, {device_uuid}, 81559, 80000, 0, Disabled", process_uuid)
    assert report.status == "unknown"
    assert device_uuid not in json.dumps(asdict(report))
    if process_uuid:
        assert process_uuid not in json.dumps(asdict(report))


def test_uuid_hex_case_cannot_evade_busy_detection(monkeypatch):
    report = probe(monkeypatch,
                   "0, NVIDIA H100, GPU-abcdefab-abcd-abcd-abcd-abcdefabcdef, 81559, 80000, 0, Disabled",
                   "GPU-ABCDEFAB-ABCD-ABCD-ABCD-ABCDEFABCDEF")
    assert report.status == "unavailable"
    assert report.devices[0].has_compute_processes is True
    assert "abcdefab" not in json.dumps(asdict(report)).lower()


@pytest.mark.parametrize("failed_query", ["gpu", "compute-apps"])
def test_query_decode_failure_is_sanitized(monkeypatch, failed_query):
    monkeypatch.setattr(hardware.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def run(args, **kwargs):
        if any(f"--query-{failed_query}=" in arg for arg in args):
            raise UnicodeDecodeError("utf-8", b"\xffprivate-host", 0, 1, "invalid byte")
        output = "0, NVIDIA H100, GPU-11111111-1111-1111-1111-111111111111, 81559, 80000, 0, Disabled"
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(hardware.subprocess, "run", run)
    report = hardware.inspect_hardware()
    assert report.status == "unknown"
    assert "private-host" not in json.dumps(asdict(report))
