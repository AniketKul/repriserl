from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from random import Random

import pytest

from repriserl.reference.buffer import ReferenceReplayStore
from repriserl.reference.types import PolicyIdentity, ReserveRequest, Rollout, RolloutGroup


POLICY = PolicyIdentity("b", 0, "m", "t", "v", "s")
ROW = Rollout((1, 2), (True, True), (-2., -1.), (-2., -1.), 0., 0.)


def group(name, **kwargs):
    return RolloutGroup(name, "task", (1,), POLICY, (ROW,) * 4, **kwargs)


def request(name, step=1, epoch=0, seed=0, **quotas):
    return ReserveRequest(name, replace(POLICY, step=step), quotas or {"all": 1}, seed, epoch)


def test_put_idempotency_capacity_and_immutable_conflict():
    store = ReferenceReplayStore(capacity=1, max_batches=1)
    assert store.put_group(group("a")).inserted
    assert not store.put_group(group("a")).inserted
    with pytest.raises(ValueError, match="conflicting"):
        store.put_group(replace(group("a"), task_id="different"))
    with pytest.raises(BufferError):
        store.put_group(group("b"))
    with pytest.raises(ValueError):
        store.put_group(replace(group("a"), rollouts=(ROW,) * 2))
    store.reserve(request("r"))
    with pytest.raises(BufferError):
        store.reserve(request("r2"))
    assert store.get_group("a") == group("a")


def test_pending_uses_and_idempotent_subsets():
    store = ReferenceReplayStore()
    store.put_group(group("a"))
    r1 = request("one")
    r2 = request("two")
    assert store.reserve(r1) == store.reserve(r1)
    with pytest.raises(ValueError):
        store.reserve(replace(r1, seed=99))
    assert store.reserve(r2).group_ids == ("a",)
    assert store.reserve(request("three")).unmet == {"all": 1}
    with pytest.raises(ValueError):
        store.record_applied("one", 0, ("else",))
    store.record_applied("one", 0, ())
    store.record_applied("one", 0, ())
    assert store.use_count("a") == 1
    with pytest.raises(ValueError):
        store.record_applied("one", 0, ("a",))
    assert store.reserve(request("four")).group_ids == ("a",)
    store.record_applied("four", 0)
    store.record_applied("two", 0)
    store.record_applied("two", 0)
    assert store.use_count("a") == 3
    assert store.reserve(request("five")).group_ids == ()


def test_epoch_release_and_retry():
    store = ReferenceReplayStore(epoch=3)
    store.put_group(group("a"))
    request3 = request("one", epoch=3)
    with pytest.raises(ValueError):
        store.reserve(request("bad", epoch=2))
    store.reserve(request3)
    with pytest.raises(ValueError):
        store.release("one", 2)
    with pytest.raises(ValueError):
        store.record_applied("one", 2)
    store.release("one", 3)
    store.release("one", 3)
    with pytest.raises(ValueError):
        store.reserve(request3)
    with pytest.raises(ValueError):
        store.record_applied("one", 3)
    assert store.use_count("a") == 1


@pytest.mark.parametrize("step,eligible", [(0, False), (1, True), (8, True), (9, False)])
def test_age_window_and_lineage(step, eligible):
    store = ReferenceReplayStore()
    store.put_group(group("a"))
    assert bool(store.reserve(request("r", step=step)).group_ids) is eligible
    for field in ("branch_id", "model_id", "tokenizer_id", "verifier_id", "sampler_id"):
        bad = replace(request(field), current_policy=replace(POLICY, step=1, **{field: "other"}))
        assert store.reserve(bad).group_ids == ()


def test_seeded_uniform_stratum_selection_independent_of_insertion_order():
    request1 = request("r", short=2, long=5, seed=1)
    selected = []
    for order in (range(8), reversed(range(8))):
        store = ReferenceReplayStore()
        for i in order:
            store.put_group(group(str(i), stratum="short" if i < 6 else "long"))
        receipt = store.reserve(request1)
        assert receipt.unmet == {"short": 0, "long": 3}
        assert len(set(receipt.group_ids)) == 4
        assert {"6", "7"} <= set(receipt.group_ids)
        selected.append(receipt.group_ids)
    assert selected[0] == selected[1]


def test_concurrent_reservations_and_random_retries_respect_lifetime_budget():
    store = ReferenceReplayStore()
    for i in range(16):
        store.put_group(group(str(i)))
    with ThreadPoolExecutor(4) as pool:
        reserved = list(pool.map(lambda i: store.reserve(request(str(i), seed=i, all=4)), range(20)))
    rng = Random(43)
    rng.shuffle(reserved)
    expected = {str(i): 1 for i in range(16)}
    for r in reserved:
        accepted = tuple(g for g in r.group_ids if rng.random() < 0.6)
        for g in accepted:
            expected[g] += 1
        store.record_applied(r.batch_id, 0, accepted)
        store.record_applied(r.batch_id, 0, accepted)
    assert {g: store.use_count(g) for g in expected} == expected
    assert max(expected.values()) <= 3
