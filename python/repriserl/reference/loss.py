"""Independent eager PyTorch reference for the globally token-normalized surrogate."""
import math

import torch

from .types import ReplayBatch


def compute_advantages(rewards: torch.Tensor, std_correction: int, epsilon: float = 1e-4) -> torch.Tensor:
    if not rewards.is_floating_point() or rewards.ndim != 1 or not torch.isfinite(rewards).all():
        raise ValueError("expected a vector of finite floating-point group rewards")
    if type(std_correction) is not int or std_correction not in (0, 1) or rewards.numel() <= std_correction:
        raise ValueError("invalid standard deviation correction for group size")
    if type(epsilon) not in (float, int) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    with torch.no_grad():
        values = rewards if rewards.dtype == torch.float64 else rewards.float()
        result = (values - values.mean()) / (values.std(correction=std_correction) + epsilon)
        if not torch.isfinite(result).all():
            raise ValueError("non-finite standardized group rewards")
        return result


def replay_loss(current_logp: torch.Tensor, batch: ReplayBatch, *, row_start: int | None = None) -> torch.Tensor:
    """Rows begin at row_start and may be right-padded; divide by all batch actions.

    For microbatch accumulation, backpropagate each contiguous row slice once.
    The loss is accumulated in at least FP32, matching the trainer's logp math.
    """
    if not current_logp.is_floating_point() or current_logp.ndim != 2:
        raise ValueError("current_logp must be a floating-point [rollout, token] tensor")
    nrows, width = current_logp.shape
    if row_start is None:
        if nrows != len(batch.rows):
            raise ValueError("full-batch row count differs; specify row_start for a microbatch")
        row_start = 0
    if type(row_start) is not int or row_start < 0 or nrows < 1 or row_start + nrows > len(batch.rows):
        raise ValueError("invalid microbatch row range")
    rows = batch.rows[row_start:row_start + nrows]
    if width < max(len(r.completion_ids) for r in rows):
        raise ValueError("insufficient token columns")
    dtype = torch.float64 if current_logp.dtype == torch.float64 else torch.float32
    device = current_logp.device
    masks = [r.action_mask + (False,) * (width - len(r.action_mask)) for r in rows]
    mask = torch.tensor(masks, device=device, dtype=torch.bool)
    current = current_logp[mask].to(dtype)
    if not torch.isfinite(current).all() or (current > 0).any():
        raise ValueError("invalid current action log probabilities")

    def field(name):
        return torch.tensor([x for r in rows for x, b in zip(getattr(r, name), r.action_mask) if b],
                            dtype=dtype, device=device)

    old = field("training_logp_b")
    sampled = field("sampling_logp_b")
    advantage = torch.tensor([r.advantage for r in rows for b in r.action_mask if b],
                             dtype=dtype, device=device)
    ratio = (current - old).exp()
    correction = (old - sampled).clamp(max=math.log(2.0)).exp()
    if not torch.isfinite(ratio).all() or not torch.isfinite(correction).all():
        raise ValueError("non-finite importance weight")
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
    result = -(correction * surrogate).sum() / batch.valid_tokens
    if not torch.isfinite(result):
        raise ValueError("non-finite replay loss")
    return result
