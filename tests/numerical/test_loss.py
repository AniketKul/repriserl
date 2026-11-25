from math import exp, log

import pytest

torch = pytest.importorskip("torch")
from repriserl.reference.loss import compute_advantages, replay_loss
from repriserl.reference.types import PolicyIdentity, ReplayBatch, Rollout, RolloutGroup


POLICY = PolicyIdentity("b", 0, "m", "t", "v", "s")


def example():
    a = Rollout([3, 4, 5], [True, False, True], [-3., 0., -1.], [-2., 0., -2.], 1., 1.5)
    b = Rollout([4], [True], [-1.], [-2.], 0., -0.5)
    return ReplayBatch((RolloutGroup("g", "task", [2], POLICY, [a, b]),), POLICY)


@pytest.mark.parametrize("correction,expected", [(0, 1.), (1, 2**0.5)])
def test_frozen_advantages(correction, expected):
    rewards = torch.tensor([0., 2.], dtype=torch.float64, requires_grad=True)
    actual = compute_advantages(rewards, correction)
    assert actual.tolist() == pytest.approx([-1 / (expected + 1e-4), 1 / (expected + 1e-4)])
    assert not actual.requires_grad
    assert compute_advantages(torch.ones(4), correction).tolist() == [0.] * 4


def test_gradient_and_loss_against_independent_scalar_reference():
    batch = example()
    data = [[-1.5, float("nan"), -2.1], [-1.95, float("nan"), float("nan")]]
    # Reference recomputes in Python scalars to avoid using the autograd expression.
    def scalar(x):
        total = 0.
        for values, row in zip(x, batch.rows):
            for cur, old, sample, valid in zip(values, row.training_logp_b, row.sampling_logp_b, row.action_mask):
                if valid:
                    ratio = exp(cur - old)
                    total += min(2., exp(old - sample)) * min(
                        ratio * row.advantage, max(.8, min(1.2, ratio)) * row.advantage)
        return -total / 3

    current = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    loss = replay_loss(current, batch)
    assert loss.item() == pytest.approx(scalar(data))
    loss.backward()
    for i, j in [(0, 0), (0, 2), (1, 0)]:
        step = 1e-6
        data[i][j] += step
        hi = scalar(data)
        data[i][j] -= 2 * step
        lo = scalar(data)
        data[i][j] += step
        assert current.grad[i, j].item() == pytest.approx((hi - lo) / (2 * step), abs=1e-8)
    assert current.grad[0, 1] == 0
    assert torch.isfinite(current.grad).all()

    # Unequal action counts require division by the full optimizer-batch denominator.
    partial = current.detach().clone().requires_grad_()
    for i in range(2):
        replay_loss(partial[i:i + 1], batch, row_start=i).backward()
    torch.testing.assert_close(partial.grad, current.grad)


def test_one_valid_token_masked_rows_and_clipping_negative_advantage():
    a = Rollout([2], [False], [float("nan")], [float("nan")], 0, -1)
    b = Rollout([3], [True], [-2.], [-2.], 0, -1)
    batch = ReplayBatch([RolloutGroup("g", "task", [1], POLICY, [a, b])], POLICY)
    current = torch.tensor([[float("nan")], [-2. + log(.5)]], requires_grad=True)
    assert replay_loss(current, batch).item() == pytest.approx(.8)
    replay_loss(current[:1], batch, row_start=0).backward()
    assert torch.equal(current.grad, torch.zeros_like(current))
    with pytest.raises(ValueError, match="row count"):
        replay_loss(current[:1], batch)


@pytest.mark.parametrize("field", ["nan", "inf", "-inf", "0.01"])
def test_invalid_current_action_probability_rejected(field):
    batch = example()
    current = torch.full((2, 3), -1.)
    current[0, 0] = float(field)
    with pytest.raises(ValueError, match="probabilities"):
        replay_loss(current, batch)


def test_numerical_overflow_fails_closed():
    # Difference between finite log likelihoods can overflow exp in optimizer dtype.
    b = Rollout((1,), (True,), (-1000.,), (-1000.,), 1, 1)
    batch = ReplayBatch((RolloutGroup("g", "task", (1,), POLICY, [b, b]),), POLICY)
    with pytest.raises(ValueError, match="importance weight"):
        replay_loss(torch.full((2, 1), -1.), batch)



def test_unrepresentable_frozen_advantage_cannot_enter_optimizer():
    b = Rollout((1,), (True,), (-1.,), (-1.,), 0., 1e300)
    batch = ReplayBatch([RolloutGroup("g", "task", [1], POLICY, [b, b])], POLICY)
    with pytest.raises(ValueError, match="non-finite replay loss"):
        replay_loss(torch.full((2, 1), -1.), batch)
