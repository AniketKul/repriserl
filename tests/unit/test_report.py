import json

import pytest

from repriserl.report import compare_runs


def save(path, arm, seconds, **overrides):
    path.mkdir()
    event = dict(step=1, elapsed_seconds=seconds, correct=2, total=4, split="development")
    versions = dict.fromkeys(("torch", "transformers", "trl", "accelerate", "tokenizers", "datasets"), "1")
    s = dict(schema_version=1, run_id=arm, arm=arm, seed=1, fixture=True, steps=1,
             elapsed_seconds=seconds+1, device="cpu", split_digest="a"*64, verifier_digest="b"*64,
             status="completed", development=[event, event], eval_every=1, dev_size=4,
             device_sharing="cpu", backend="transformers", versions=versions,
             config={"max_steps": 1, "model": "tiny"})
    (path/"summary.json").write_text(json.dumps(s | overrides))
    return path


def test_compare_matching_interrupted_and_duplicate_inputs(tmp_path):
    a = save(tmp_path/"a", "baseline", 2)
    b = save(tmp_path/"b", "fixed_replay", 3)
    result = compare_runs([a, b, a], 0.5)
    assert len(result["runs"]) == 2
    assert len(result["runs"][0]["development"]) == 1
    assert result["pairs"][0]["time_to_target_savings_fraction"] == -0.5
    assert compare_runs([a,b], 0.75)["pairs"][0]["time_to_target_savings_fraction"] is None
    for name, extra in [("failed", {"status":"failed", "elapsed_seconds":None, "development":[]}),
                        ("budget", {"config":{"max_steps":1, "max_time_seconds":2, "model":"tiny"}})]:
        d = save(tmp_path/name, "fixed_replay", 1, **extra)
        assert compare_runs([a,d], 0.5)["pairs"][0]["time_to_target_savings_fraction"] is None


def test_invalid_evidence_cannot_win(tmp_path):
    event = dict(step=1, elapsed_seconds=1, correct=4, total=4, split="development")
    for name,extra in [("missing", {"elapsed_seconds":None}), ("sealed", {"development":[event | {"split":"test"}]}),
                       ("zero", {"development":[event | {"elapsed_seconds":0}]}),
                       ("subset", {"development":[event | {"correct":1, "total":1}]}),
                       ("off_cadence", {"eval_every":2, "steps":3, "config":{"max_steps":3},
                                        "development":[event, event | {"step":3, "elapsed_seconds":2}]}),
                       ("missing_eval", {"development":[]}), ("incomplete", {"config":{"max_steps":2}})]:
        with pytest.raises(ValueError):
            compare_runs([save(tmp_path/name, "baseline", 1, **extra)], .5)


@pytest.mark.parametrize("extra", [{"split_digest":None}, {"verifier_digest":""}, {"config":{}},
                                   {"backend":None}, {"versions":{}}, {"device_sharing":"unverified"}])
def test_identically_missing_or_unverified_provenance_cannot_match(tmp_path, extra):
    a = save(tmp_path/"a", "baseline", 2, **extra)
    b = save(tmp_path/"b", "fixed_replay", 1, **extra)
    pair, = compare_runs([a,b], .5)["pairs"]
    assert not pair["matched"]
    assert pair["time_to_target_savings_fraction"] is None


def test_final_off_cadence_evaluation_and_split_size_matching(tmp_path):
    e = dict(step=2, elapsed_seconds=1, correct=0, total=4, split="development")
    extra = {"steps":3, "config":{"max_steps":3, "model":"tiny"}, "eval_every":2}
    a = save(tmp_path/"a", "baseline", 2, **extra,
             development=[e, e | {"step":3, "elapsed_seconds":2, "correct":2}])
    b = save(tmp_path/"b", "fixed_replay", 2, **extra, dev_size=2,
             development=[e | {"total":2}, e | {"step":3, "elapsed_seconds":2, "correct":1, "total":2}])
    assert compare_runs([a,b], .5)["pairs"][0]["matched"] is False


@pytest.mark.parametrize("sharing,samples,matched", [
    ("sampled_uncontended_h100", 5, True), ("sampled_uncontended_h100", 0, False),
    ("shared_h100", 5, False), ("unverified", 5, False)])
def test_gpu_pairs_require_uncontended_observations(tmp_path, sharing, samples, matched):
    config = {"model":"model", "model_revision":"a"*40, "max_steps":1}
    fields = dict(config=config, fixture=False, device="cuda", device_sharing=sharing,
                  occupancy_samples=samples)
    dirs = [save(tmp_path/arm, arm, 2, **fields) for arm in ("baseline", "fixed_replay")]
    report = compare_runs(dirs, .5)
    assert report["pairs"][0]["matched"] is matched
    assert report["runs"][0]["device_sharing"] == sharing
