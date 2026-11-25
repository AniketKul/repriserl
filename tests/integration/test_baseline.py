import json
from dataclasses import replace

import pytest

from repriserl.config import RunConfig

pytest.importorskip("trl")


def test_stock_cpu_training_and_development_evaluation(tmp_path):
    from repriserl.adapters.trl.baseline import run_baseline
    config = RunConfig(max_steps=1, prompt_token_cap=64, completion_token_cap=3,
                       group_size=2, batch_groups=1, microbatch_size=1, seed=1)
    summary = run_baseline(config, tmp_path, tiny=True, sizes=(3, 2, 0))
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert summary["status"] == "completed" and summary["steps"] == 1
    assert summary["fixture"] and summary["device"] == "cpu"
    assert summary["development"][-1]["total"] == 2
    assert summary["development"][-1]["split"] == "development"
    assert summary["phase_seconds"]["generation"] > 0
    assert summary["phase_seconds"]["backward"] > 0
    assert summary["phase_seconds"]["optimizer"] > 0
    assert {"grad_norm", "reward", "loss"} <= summary["training_metrics"][0].keys()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    saved_model = AutoModelForCausalLM.from_pretrained(tmp_path / summary["model_path"])
    saved_tokenizer = AutoTokenizer.from_pretrained(tmp_path / summary["model_path"])
    assert saved_model.config.vocab_size == len(saved_tokenizer)
    assert sum(summary["phase_seconds"].values()) <= summary["elapsed_seconds"]
    with pytest.raises(FileExistsError):
        run_baseline(config, tmp_path, tiny=True, sizes=(3, 2, 0))
    invalid = replace(config, std_convention="population")
    with pytest.raises(ValueError, match="std_convention"):
        run_baseline(invalid, tmp_path / "invalid", tiny=True, sizes=(3, 2, 0))
    failed = json.loads((tmp_path / "invalid/summary.json").read_text())
    assert failed["status"] == "failed" and failed["steps"] == 0

def test_expired_run_preserves_failure_receipt(tmp_path):
    from repriserl.adapters.trl.baseline import run_baseline
    config = RunConfig(max_steps=2, max_time_seconds=0.0001, batch_groups=1, group_size=2)
    with pytest.raises(TimeoutError, match="budget"):
        run_baseline(config, tmp_path, tiny=True, sizes=(8, 1, 0))
    receipt = json.loads((tmp_path / "summary.json").read_text())
    assert receipt["status"] == "failed" and receipt["stop_reason"] == "wall_time_budget"


def test_cpu_head_matches_model_logits_and_gradients():
    import torch
    from repriserl.adapters.trl.models import load_model_and_tokenizer, enable_tiny_cpu_scores
    model, _ = load_model_and_tokenizer(RunConfig(max_steps=1), tiny=True)
    tokens = torch.tensor([[5, 8, 1, 2]])
    labels = torch.tensor([[-100, -100, 1, 2]])
    scores = model(tokens).logits.log_softmax(-1)
    expected = torch.stack([scores[0, 1, 1], scores[0, 2, 2]])
    weights = model.get_input_embeddings().weight
    expected_gradient, = torch.autograd.grad(expected.sum(), weights)
    enable_tiny_cpu_scores(model)
    actual = model(tokens, fused_lm_head=True, labels=labels)
    assert actual.label_mask.tolist() == [[False, True, True]]
    torch.testing.assert_close(actual.log_probs[0, 1:], expected)
    gradient, = torch.autograd.grad(actual.log_probs.sum(), weights)
    torch.testing.assert_close(gradient, expected_gradient)
    padded = torch.cat((torch.zeros(1, 2, dtype=tokens.dtype), tokens), dim=1)
    attention = torch.tensor([[0, 0, 1, 1, 1, 1]])
    masked = torch.cat((torch.full((1, 2), -100), labels), dim=1)
    padded_scores = model(padded, attention_mask=attention, labels=masked, fused_lm_head=True)
    torch.testing.assert_close(padded_scores.log_probs[0, -2:], expected)


def test_cuda_preflight_records_denial_before_loading_a_model(monkeypatch, tmp_path):
    from repriserl import gpu_run
    from repriserl.adapters.trl.baseline import run_baseline
    def deny(allow_shared):
        assert not allow_shared
        raise RuntimeError("device unavailable")
    monkeypatch.setattr(gpu_run, "check_device", deny)
    with pytest.raises(RuntimeError, match="device unavailable"):
        run_baseline(RunConfig(max_steps=1), tmp_path, sizes=(8, 1, 0))
    record = json.loads((tmp_path / "summary.json").read_text())
    assert record["status"] == "failed" and record["steps"] == 0
    assert record["device_sharing"] == "unverified"


def test_instruct_prompt_uses_model_generation_template():
    from repriserl.adapters.trl.models import load_model_and_tokenizer, problem_prompt
    from repriserl.verifiers.number_expression import Problem
    _, tokenizer = load_model_and_tokenizer(RunConfig(max_steps=1), tiny=True)
    tokenizer.chat_template = (
        "{{ messages[0].role }}: {{ messages[0].content }}"
        "{% if add_generation_prompt %} assistant:{% endif %}"
    )
    problem = Problem((2, 2, 3), 7)
    assert problem_prompt(problem, tokenizer) == "user: " + problem_prompt(problem) + " assistant:"

def test_time_only_trainer_rejected_before_allocating(tmp_path):
    from repriserl.adapters.trl.baseline import run_baseline
    with pytest.raises(ValueError, match="max_steps"):
        run_baseline(RunConfig(max_time_seconds=15), tmp_path, tiny=True, sizes=(8, 1, 0))
    assert json.loads((tmp_path / "summary.json").read_text())["stop_reason"] == "error"


@pytest.mark.parametrize("expire_on_second_eval", [False, True])
def test_first_target_checkpoint_and_completed_progress(tmp_path, monkeypatch, expire_on_second_eval):
    import signal
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from repriserl.adapters.trl import baseline
    calls = []
    def evaluate(trainer, problems, cap):
        calls.append(len(problems))
        if expire_on_second_eval and len(calls) == 2:
            signal.raise_signal(signal.SIGALRM)
        return len(problems)
    monkeypatch.setattr(baseline, "_evaluate", evaluate)
    config = RunConfig(max_steps=2, max_time_seconds=120, prompt_token_cap=64, completion_token_cap=2,
                       batch_groups=1, group_size=2, target_accuracy=1)
    if expire_on_second_eval:
        with pytest.raises(TimeoutError):
            baseline.run_baseline(config, tmp_path, tiny=True, sizes=(3, 2, 0))
    else:
        baseline.run_baseline(config, tmp_path, tiny=True, sizes=(3, 2, 0))
    result = json.loads((tmp_path / "summary.json").read_text())
    hit = result["target_checkpoint"]
    assert result["steps"] == 2
    assert hit["step"] == 1 and hit["target_accuracy"] == hit["accuracy"] == 1
    assert hit["elapsed_seconds"] == result["development"][0]["elapsed_seconds"]
    assert 0 < result["phase_seconds"]["save_target_model"] <= hit["elapsed_seconds"]
    policy = tmp_path / hit["model_path"]
    assert AutoModelForCausalLM.from_pretrained(policy).config.vocab_size == len(AutoTokenizer.from_pretrained(policy))
    if expire_on_second_eval:
        assert result["status"] == "failed" and result["stop_reason"] == "wall_time_budget"
        assert len(result["development"]) == 1 and not (tmp_path / "model").exists()
    else:
        assert result["status"] == "completed" and (tmp_path / "model").is_dir()
        assert len(result["development"]) == 2
    (tmp_path / "summary.json").unlink()
    with pytest.raises(FileExistsError):
        baseline.run_baseline(config, tmp_path, tiny=True, sizes=(3, 2, 0))
