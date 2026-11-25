import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
from repriserl.adapters.trl.replay import run_fixed_replay
from repriserl.config import RunConfig


def test_tiny_cpu_optimizer_steps_include_real_reuse(tmp_path, monkeypatch):
    from itertools import islice
    from repriserl.adapters.trl.baseline import make_trainer
    from repriserl.adapters.trl.compat import ReplayGRPOAdapter
    from repriserl.data.problems import make_splits
    config = RunConfig(max_steps=9, prompt_token_cap=64, completion_token_cap=3,
                       group_size=2, batch_groups=2, microbatch_size=1, std_convention="sample",
                       eos_handling="include", truncation_handling="include", normalization="optimizer_batch")
    sizes = (24, 2, 0)
    stock = make_trainer(config, tmp_path / "stock", tiny=True, num_iterations=1,
                         splits=make_splits(config.seed, sizes))
    loader = iter(stock.get_train_dataloader())
    expected = []
    for _ in range(config.max_steps):
        batches = [[row["problem_id"] for row in batch]
                   for batch in islice(loader, config.group_size * config.batch_groups // config.microbatch_size)]
        assert all(batch == batches[0] for batch in batches)
        p, q = batches[0][::2]
        assert p != q and batches[0] == [p, p, q, q]
        expected.append([p, q])
    sampled = []
    original = ReplayGRPOAdapter._sample

    def observe(self, rows):
        sampled.append([r["problem_id"] for r in rows])
        return original(self, rows)

    monkeypatch.setattr(ReplayGRPOAdapter, "_sample", observe)
    result = run_fixed_replay(config, tmp_path / "run", tiny=True, sizes=sizes, eval_every=9)
    assert sampled[:8] == expected[:8]
    assert sampled[8] == expected[8][:1]
    assert result["status"] == "completed" and result["steps"] == 9
    replay = result["replay_metrics"]
    steps = replay["steps"]
    assert len(steps) == 9
    assert all(s["accepted"] == [] and len(s["fresh"]) == 2 for s in steps[:8])
    assert len(steps[8]["accepted"]) == len(steps[8]["fresh"]) == 1
    assert steps[8]["accepted"][0] in {gid for step in steps[:8] for gid in step["fresh"]}
    for step in steps:
        for source, ids in (("fresh", step["fresh"]), ("replay", step["accepted"])):
            counts = step["reward_counts"][source]
            assert counts["groups"] == len(ids)
            assert counts["completions"] == config.group_size * len(ids)
            assert counts["mixed_reward_groups"] == counts["nonzero_advantage_groups"]
    assert all(value >= 0 for value in replay["phase_seconds"].values())


def test_full_store_fails_before_generation_and_optimizer(tmp_path, monkeypatch):
    import json
    from repriserl.adapters.trl import compat
    from repriserl.reference.buffer import ReferenceReplayStore

    monkeypatch.setattr(compat, "ReferenceReplayStore",
                        lambda **kwargs: ReferenceReplayStore(capacity=1, **kwargs))
    def unexpected_sample(self, rows):
        pytest.fail("trainer generated a batch that could not be recorded")
    monkeypatch.setattr(compat.ReplayGRPOAdapter, "_sample", unexpected_sample)
    config = RunConfig(max_steps=1, group_size=2, batch_groups=2, microbatch_size=2,
                       prompt_token_cap=64, completion_token_cap=3)
    with pytest.raises(BufferError):
        run_fixed_replay(config, tmp_path, tiny=True, sizes=(8, 2, 0))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["steps"] == 0
    assert summary["replay_metrics"]["steps"] == []


def test_reward_counts_distinguish_all_correct_from_learning_signal():
    from repriserl.adapters.trl.compat import _reward_counts
    from repriserl.reference.types import PolicyIdentity, Rollout, RolloutGroup

    def row(reward, advantage, truncated=False):
        return Rollout((1,), (True,), (-1.0,), (-1.0,), reward, advantage, truncated)

    policy = PolicyIdentity("branch", 0, "model", "tok", "verifier", "sampler")
    rows = [(row(1, 0), row(1, 0)), (row(0, 0), row(0, 0, True)), (row(1, 1), row(0, -1, True))]
    groups = [RolloutGroup(str(i), "task", (1,), policy, group) for i, group in enumerate(rows)]
    assert _reward_counts(groups) == {
        "groups": 3, "completions": 6, "rewarded_completions": 3, "nonzero_reward_groups": 2,
        "mixed_reward_groups": 1, "nonzero_advantage_groups": 1, "truncated_completions": 2,
    }
