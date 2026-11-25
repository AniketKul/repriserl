import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
from repriserl.adapters.trl.replay import run_fixed_replay
from repriserl.config import RunConfig


def test_tiny_cpu_optimizer_steps_include_real_reuse(tmp_path):
    config = RunConfig(max_steps=9, prompt_token_cap=64, completion_token_cap=3,
                       group_size=2, batch_groups=2, microbatch_size=2, std_convention="sample",
                       eos_handling="include", truncation_handling="include", normalization="optimizer_batch")
    result = run_fixed_replay(config, tmp_path / "run", tiny=True, sizes=(16, 2, 0), eval_every=9)
    assert result["status"] == "completed" and result["steps"] == 9
    replay = result["replay_metrics"]
    steps = replay["steps"]
    assert len(steps) == 9
    assert all(s["accepted"] == [] and len(s["fresh"]) == 2 for s in steps[:8])
    assert len(steps[8]["accepted"]) == len(steps[8]["fresh"]) == 1
    assert steps[8]["accepted"][0] in {gid for step in steps[:8] for gid in step["fresh"]}
    assert all(value >= 0 for value in replay["phase_seconds"].values())
