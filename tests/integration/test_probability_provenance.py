from dataclasses import replace
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
from repriserl.adapters.trl.compat import ReplayGRPOAdapter, drift_rejection, prepared_inputs
from repriserl.reference.types import PolicyIdentity, Rollout, RolloutGroup


def test_rescoring_never_rewrites_behavior_or_advantage():
    policy = PolicyIdentity("branch", 0, "model", "tok", "verifier", "sampler")
    row = Rollout((1, 2), (True, True), (-4., -2.), (-2., -2.), 1., 0.5)
    group = RolloutGroup("g", "p", (1,), policy, (row, row))
    batch = ReplayGRPOAdapter.prepare_batch((group,), replace(policy, step=3))
    inputs = prepared_inputs(batch, 0, "cpu")
    assert inputs["advantages"].tolist() == [[1.0, 0.5], [1.0, 0.5]]
    for delta, reason in [(0, None), (0.4, "drift"), (float("nan"), "invalid_ratio")]:
        assert drift_rejection(inputs["old_per_token_logps"] + delta,
                               inputs["old_per_token_logps"], inputs["tool_mask"]) == reason
    assert row.training_logp_b == (-2., -2.) and row.sampling_logp_b == (-4., -2.)
    for field in ("branch_id", "tokenizer_id", "verifier_id", "sampler_id"):
        with pytest.raises(ValueError):
            ReplayGRPOAdapter.prepare_batch((group,), replace(policy, **{field: "other"}))


def test_reject_unsupported_trainer(monkeypatch):
    import repriserl.adapters.trl.compat as compat
    monkeypatch.setattr(compat, "version", lambda name: "0.0")
    with pytest.raises(RuntimeError, match="1.15.0"):
        ReplayGRPOAdapter()


def test_actual_sample_and_frozen_denominators_survive_weight_updates(tmp_path):
    from repriserl.config import RunConfig
    from repriserl.data.problems import make_splits
    from repriserl.adapters.trl.baseline import make_trainer

    trainer = make_trainer(RunConfig(max_steps=1, group_size=2, batch_groups=1, microbatch_size=2,
                                    prompt_token_cap=64, completion_token_cap=4), tmp_path,
                           tiny=True, num_iterations=1, splits=make_splits(7, (8, 1, 0)),
                           trainer_class=ReplayGRPOAdapter)
    index = trainer.train_dataset[0]
    sample, = trainer._sample([index])
    policy = PolicyIdentity("branch", 0, "tiny", "wordlevel", "verifier", "sampler")
    group = trainer._freeze(sample, policy, "g0")
    for rollout in group.rollouts:
        torch.testing.assert_close(torch.tensor(rollout.training_logp_b), torch.tensor(rollout.sampling_logp_b),
                                   rtol=1e-5, atol=1e-5)
    batch = ReplayGRPOAdapter.prepare_batch([group], policy)
    inputs = prepared_inputs(batch, 0, "cpu")
    old = inputs["old_per_token_logps"].clone()
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.05)
    ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
    mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
    for _ in range(3):
        optimizer.zero_grad()
        scores, _, _ = trainer._get_per_token_logps_and_entropies(
            trainer.model, ids, mask, inputs["completion_ids"].shape[1])
        loss = -scores[inputs["tool_mask"]].mean()
        loss.backward()
        optimizer.step()
    assert not torch.allclose(trainer._teacher_score(inputs)[inputs["tool_mask"]], old[inputs["tool_mask"]])
    later_batch = ReplayGRPOAdapter.prepare_batch([group], replace(policy, step=3))
    torch.testing.assert_close(prepared_inputs(later_batch, 0, "cpu")["old_per_token_logps"], old)
