"""Real stock-TRL loss and model-gradient comparisons on supplied tokens."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
from trl import GRPOTrainer

from repriserl.adapters.trl.baseline import make_trainer
from repriserl.adapters.trl.compat import ReplayGRPOAdapter, prepared_inputs
from repriserl.config import RunConfig
from repriserl.data.problems import make_splits
from repriserl.reference.loss import replay_loss
from repriserl.reference.types import PolicyIdentity, ReplayBatch, Rollout, RolloutGroup


@pytest.mark.parametrize("sampler_shift", [0.0, 1.2])
def test_real_stock_loss_and_gradients_match_reference(tmp_path, sampler_shift):
    config = RunConfig(max_steps=1, group_size=2, batch_groups=1, microbatch_size=2,
                       prompt_token_cap=64, completion_token_cap=4, std_convention="sample",
                       truncation_handling="include", eos_handling="include", normalization="optimizer_batch")
    trainer = make_trainer(config, tmp_path, tiny=True, num_iterations=1,
                           splits=make_splits(42, (8, 2, 0)), trainer_class=ReplayGRPOAdapter)
    trainer.model.train()
    trainer.current_gradient_accumulation_steps = 1
    policy = PolicyIdentity("test", 0, "tiny", "wordlevel", "verifier", "sampler")
    ids = ((4, 8, 1), (6, 1))
    tokens = torch.tensor([[3, 5, 4, 8, 1], [3, 5, 6, 1, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    def logps():
        current, _, _ = trainer._get_per_token_logps_and_entropies(trainer.model, tokens, mask, 3)
        return current
    frozen = logps().detach().tolist()
    rows = []
    for i, completion in enumerate(ids):
        old = frozen[i][:len(completion)]
        rows.append(Rollout(completion, [True] * len(old), [x - sampler_shift for x in old],
                            old, float(i), (1, -1)[i]))
    group = RolloutGroup("group", "task", (3, 5), policy, rows)
    batch = ReplayBatch([group], policy)
    prepared = prepared_inputs(batch, 0, "cpu")
    weight = trainer.model.get_input_embeddings().weight
    ref = replay_loss(logps(), batch)
    ref_gradient, = torch.autograd.grad(ref, weight)
    actual = trainer._compute_loss(trainer.model, prepared)
    actual_gradient, = torch.autograd.grad(actual, weight)
    torch.testing.assert_close(actual, ref)
    torch.testing.assert_close(actual_gradient, ref_gradient, rtol=1e-4, atol=1e-6)
    assert actual_gradient.abs().sum() > 0
    if sampler_shift == 0:
        stock_inputs = dict(prepared, advantages=torch.tensor([1., -1.]))
        for field in ("old_per_token_logps", "sampling_per_token_logps", "tool_mask"):
            stock_inputs.pop(field)
        stock = GRPOTrainer._compute_loss(trainer, trainer.model, stock_inputs)
        torch.testing.assert_close(stock, actual)
        stock_gradient, = torch.autograd.grad(stock, weight)
        torch.testing.assert_close(stock_gradient, actual_gradient)
