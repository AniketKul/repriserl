import json
import pytest
from repriserl.report import compare_runs


def save(path, arm, seconds, **overrides):
    path.mkdir()
    event = dict(step=1, elapsed_seconds=seconds, correct=2, total=4, split="development")
    s = dict(schema_version=1, run_id=arm, arm=arm, seed=1, fixture=True, steps=1,
             elapsed_seconds=seconds+1, device="cpu", split_digest="s", verifier_digest="v",
             status="completed", development=[event, event], eval_every=1, device_sharing="cpu",
             config={"max_steps":1})
    (path/"summary.json").write_text(json.dumps(s | overrides))
    return path


def test_compare_matching_interrupted_and_duplicate_inputs(tmp_path):
    a = save(tmp_path/"a", "baseline", 2)
    b = save(tmp_path/"b", "fixed_replay", 3)
    result = compare_runs([a, b, a], 0.5)
    assert len(result["runs"]) == 2
    assert result["pairs"][0]["time_to_target_savings_fraction"] == -0.5
    assert compare_runs([a,b], 0.75)["pairs"][0]["time_to_target_savings_fraction"] is None
    for name, extra in [("failed", {"status":"failed"}), ("budget", {"config":{"max_steps":2}})]:
        d = save(tmp_path/name, "fixed_replay", 1, **extra)
        assert compare_runs([a,d], 0.5)["pairs"][0]["time_to_target_savings_fraction"] is None


def test_invalid_evidence_cannot_win(tmp_path):
    for name,extra in [("missing", {"elapsed_seconds":None}), ("sealed", {"development":[
        dict(step=1, elapsed_seconds=1, correct=4, total=4, split="test")]})]:
        with pytest.raises(ValueError):
            compare_runs([save(tmp_path/name, "baseline", 1, **extra)], .5)
