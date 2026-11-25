from dataclasses import FrozenInstanceError, replace

import pytest

from repriserl.reference.types import PolicyIdentity, ReplayBatch, ReserveRequest, Rollout, RolloutGroup


def test_snapshot_copies_all_mutable_payloads_and_request_maps():
    policy = PolicyIdentity("branch", 1, "model", "tokenizer", "verifier", "sampler")
    ids, mask, sample, train = [2, 3], [True, False], [-1.5, float("nan")], [-1.6, float("nan")]
    row = Rollout(ids, mask, sample, train, 0, -1)
    rows, prompt = [row, row], [19]
    group = RolloutGroup("g", "task", prompt, policy, rows)
    groups = [group]
    batch = ReplayBatch(groups, replace(policy, step=2))
    quotas = {"all": 1}
    req = ReserveRequest("b", policy, quotas, 7, 3)
    for seq in (ids, mask, sample, train, rows, prompt, groups, quotas):
        seq.clear()
    assert batch.groups[0].prompt_ids == (19,)
    assert batch.rows[0].completion_ids == (2, 3)
    assert batch.rows[0].sampling_logp_b == (-1.5, 0)
    assert batch.rows[0].training_logp_b == (-1.6, 0)
    assert batch.valid_tokens == 2
    assert req.quotas == {"all": 1}
    with pytest.raises(FrozenInstanceError):
        row.reward = 1


@pytest.mark.parametrize("field,bad", [
    ("completion_ids", [-1]), ("completion_ids", [True]), ("action_mask", [1]),
    ("training_logp_b", [float("inf")]), ("sampling_logp_b", [float("nan")]),
    ("training_logp_b", []), ("sampling_logp_b", [1]), ("reward", float("nan")),
    ("advantage", float("inf")), ("truncated", 1),
])
def test_invalid_generation_record(field, bad):
    row = Rollout((2,), (True,), (-2.,), (-1.,), 1, 1)
    with pytest.raises(ValueError):
        replace(row, **{field: bad})


def test_batch_rejects_duplicates_lineage_mismatch_future_version_or_empty_actions():
    row = Rollout((2,), (True,), (-1.,), (-1.,), 0., 0.)
    policy = PolicyIdentity("b", 2, "model", "tok", "rw", "settings")
    group = RolloutGroup("g", "t", (2, 1), policy, (row, row))
    for field in ("model_id", "branch_id", "tokenizer_id", "verifier_id", "sampler_id", "step"):
        with pytest.raises(ValueError):
            ReplayBatch((group,), replace(policy, **{field: 1 if field == "step" else "other"}))
    with pytest.raises(ValueError, match="duplicate"):
        ReplayBatch((group, group), policy)
    with pytest.raises(ValueError, match="sizes"):
        ReplayBatch((group, replace(group, group_id="other", rollouts=(row, row, row))), policy)
    masked = replace(row, action_mask=(False,))
    with pytest.raises(ValueError, match="no valid"):
        ReplayBatch((replace(group, rollouts=(masked, masked)),), policy)
