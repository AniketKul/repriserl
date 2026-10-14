"""Problem generation must preserve determinism and canonical split isolation."""
from dataclasses import FrozenInstanceError
import hashlib
import json
import random

import pytest

from repriserl.data.problems import (
    GENERATION_DIGEST, GENERATION_VERSION, ProblemSplits, make_splits,
)
from repriserl.verifiers.number_expression import Problem, verify_expression


def test_seeded_splits_are_repeatable_disjoint_and_do_not_mutate_global_rng():
    random.seed(123)
    state = random.getstate()
    first = make_splits(42, (20, 5, 7))
    second = make_splits(42, (20, 5, 7))
    assert random.getstate() == state
    assert first == second
    assert tuple(map(len, (first.train, first.development, first.test))) == (20, 5, 7)
    assert first.train_ids == tuple(p.id for p in first.train)
    assert first.development_ids == tuple(p.id for p in first.development)
    assert first.test_ids == tuple(p.id for p in first.test)
    all_ids = first.train_ids + first.development_ids + first.test_ids
    assert len(set(all_ids)) == 32
    assert first != make_splits(43, (20, 5, 7))
    assert first.generation_digest == GENERATION_DIGEST
    assert GENERATION_VERSION == "number-problems-v1"
    assert len(GENERATION_DIGEST) == 64


def test_split_digests_bind_canonical_membership():
    splits = make_splits(5, (3, 2, 1))
    for name in ("train", "development", "test"):
        ids = sorted(getattr(splits, name + "_ids"))
        expected = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        assert getattr(splits, name + "_digest") == expected
    changed = ProblemSplits(splits.development, splits.train, splits.test)
    assert changed.canonical_digest != splits.canonical_digest


def test_split_objects_copy_inputs_and_reject_canonical_overlap():
    p = Problem([2, 2, 3], 7)
    train = [p]
    splits = ProblemSplits(train, [], [])
    train.clear()
    assert splits.train == (p,)
    with pytest.raises(FrozenInstanceError):
        splits.train = ()
    with pytest.raises(ValueError):
        ProblemSplits([p], [Problem([3, 2, 2], 7)], [])
    with pytest.raises(ValueError):
        ProblemSplits([p, p], [], [])
    with pytest.raises(TypeError):
        ProblemSplits(["invalid"], [], [])


@pytest.mark.parametrize("seed,sizes", [
    (True, (1, 1, 1)), (1.0, (1, 1, 1)), (-1, (1, 1, 1)),
    (2**64, (1, 1, 1)), (1, [1, 1, 1]), (1, (1, 1)),
    (1, (-1, 1, 1)), (1, (True, 1, 1)), (1, (1.5, 1, 1)),
    (1, (50001, 0, 0)),
])
def test_generation_rejects_malformed_or_unsupported_requests(seed, sizes):
    with pytest.raises((ValueError, TypeError)):
        make_splits(seed, sizes)


def test_empty_splits_are_supported():
    splits = make_splits(0, (0, 0, 0))
    assert splits.train_ids == splits.development_ids == splits.test_ids == ()


def test_generation_stops_at_retry_budget_without_partial_results(monkeypatch):
    import repriserl.data.problems as generation

    monkeypatch.setattr(generation, "MIN_ATTEMPTS", 1)
    monkeypatch.setattr(generation, "ATTEMPTS_PER_PROBLEM", 0)
    with pytest.raises(RuntimeError, match="retry budget exhausted"):
        make_splits(7, (2, 0, 0))


def test_versioned_small_fixture_preserves_tasks_and_hash():
    splits = make_splits(7, (3, 1, 1))
    assert [(p.numbers, p.target) for p in splits.train + splits.development + splits.test] == [
        ((2, 5, 11, 13), 58), ((6, 10, 18, 19), 66), ((3, 4, 18, 19), 1),
        ((3, 6, 8, 8), 18), ((5, 11, 14, 16), 448),
    ]
    assert splits.canonical_digest == (
        "665689616c2c531cb58223557e5ab20036028125e386443ea86c5f09ddd43476"
    )


def test_generated_tasks_are_in_declared_domains_and_solvable():
    # Independent finite rational search finds witnesses; no generator solutions used.
    from fractions import Fraction

    def solutions(items):
        if len(items) == 1:
            yield items[0]
            return
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, a_text = items[i]
                b, b_text = items[j]
                rest = [x for k, x in enumerate(items) if k not in (i, j)]
                candidates = [
                    (a + b, f"({a_text}+{b_text})"), (a - b, f"({a_text}-{b_text})"),
                    (b - a, f"({b_text}-{a_text})"), (a * b, f"({a_text}*{b_text})"),
                ]
                if b:
                    candidates.append((a / b, f"({a_text}/{b_text})"))
                if a:
                    candidates.append((b / a, f"({b_text}/{a_text})"))
                for candidate in candidates:
                    yield from solutions(rest + [candidate])

    first = make_splits(7, (3, 1, 1))
    second = make_splits(7, (3, 1, 1))
    rewards = []
    for p in first.train + first.development + first.test:
        assert len(p.numbers) == 4
        assert all(1 <= n <= 20 for n in p.numbers)
        assert 1 <= p.target <= 1000
        witness = next(text for value, text in solutions(
            [(Fraction(n), str(n)) for n in p.numbers]) if value == p.target)
        rewards.append(verify_expression(p, witness).reward)
    assert rewards == [1.0] * 5
    assert first.canonical_digest == second.canonical_digest
