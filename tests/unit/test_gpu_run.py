from types import SimpleNamespace
import pytest
import repriserl.gpu_run as gpu


def test_shared_run_requires_opt_in_and_enough_memory(monkeypatch):
    device = SimpleNamespace(name="NVIDIA H100", visible=True, mig_enabled=False,
                             memory_total_mib=81_000, memory_free_mib=78_000,
                             utilization_percent=0, has_compute_processes=True)
    monkeypatch.setattr(gpu, "inspect_hardware", lambda: SimpleNamespace(
        nvidia_smi_status="ok", devices=[device]))
    monkeypatch.setattr(gpu, "_other_processes", lambda: True)
    with pytest.raises(RuntimeError, match="exclusive"):
        gpu.check_device(False)
    with gpu.observe_run(True) as observed:
        pass
    assert observed.label == "shared_h100" and observed.samples >= 2
    device.memory_free_mib = 40_000
    with pytest.raises(RuntimeError, match="memory"):
        gpu.check_device(True)


def test_one_unreadable_sample_prevents_uncontended_claim(monkeypatch):
    values = iter((False, None, False))
    monkeypatch.setattr(gpu, "_other_processes", lambda: next(values))
    obs = gpu.Occupancy()
    obs.sample()
    assert obs.label == "sampled_uncontended_h100"
    obs.sample()
    obs.sample()
    assert obs.label == "unverified"
