"""Only a synthetic offline policy and throwaway test split are evaluated here."""
import json

import pytest

pytest.importorskip("trl")
from repriserl.config import RunConfig


@pytest.mark.parametrize("checkpoint", ["target", "final"])
def test_sealed_checkpoint_restore_verify_and_refuse_rerun(tmp_path, monkeypatch, checkpoint):
    from repriserl.adapters.trl import baseline
    from repriserl.evaluate import evaluate_checkpoint

    # Supply the dev result so the target snapshot exists; sealed scoring uses real model output.
    monkeypatch.setattr(baseline, "_evaluate", lambda trainer, problems, cap: len(problems))
    config = RunConfig(max_steps=1, prompt_token_cap=64, completion_token_cap=2,
                       batch_groups=1, group_size=2, target_accuracy=1)
    summary = baseline.run_baseline(config, tmp_path, tiny=True, sizes=(3, 1, 2))
    path = tmp_path / "summary.json"
    def reject(edits, message):
        path.write_text(json.dumps({**summary, **edits}))
        with pytest.raises(ValueError, match=message):
            evaluate_checkpoint(tmp_path, checkpoint=checkpoint)
        assert not (tmp_path / "sealed-evaluation.json").exists()
    reject({"split_digest": "a" * 64}, "split")
    reject({"verifier_digest": "a" * 64}, "verifier")
    reject({"prompt_digest": "a" * 64}, "prompt")
    reject({"status": "failed"}, "completed")
    if checkpoint == "target":
        reject({"target_checkpoint": {**summary["target_checkpoint"], "model_path": "../model"}}, "path")
        reject({"target_checkpoint": {**summary["target_checkpoint"], "sha256": "b" * 64}}, "checksum")
    else:
        reject({"model_path": "../target-model"}, "path")
        reject({"model_sha256": "b" * 64}, "checksum")
    path.write_text(json.dumps(summary))
    result = evaluate_checkpoint(tmp_path, checkpoint=checkpoint)
    assert result["status"] == "completed" and result["split"] == "test"
    assert result["total"] == 2 and 0 <= result["correct"] <= 2
    assert result["training_run_id"] == summary["run_id"] and result["checkpoint"] == checkpoint
    assert json.loads((tmp_path / "sealed-evaluation.json").read_text()) == result
    with pytest.raises(FileExistsError):
        evaluate_checkpoint(tmp_path, checkpoint="target" if checkpoint == "final" else "final")
