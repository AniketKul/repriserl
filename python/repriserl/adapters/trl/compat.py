"""Text-only replay integration for the pinned single-process GRPO trainer."""
from contextlib import contextmanager
from hashlib import sha256
from importlib.metadata import version
import math
from time import monotonic
from uuid import uuid4

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import TrainerCallback
from trl import GRPOTrainer
from trl.models import unwrap_model_for_generation
from trl.models.utils import disable_gradient_checkpointing

from repriserl.reference.buffer import ReferenceReplayStore
from repriserl.reference.loss import compute_advantages
from repriserl.reference.types import PolicyIdentity, ReplayBatch, ReserveRequest, Rollout, RolloutGroup
from repriserl.verifiers.number_expression import VERIFIER_DIGEST

from .models import reward_completions


def _pad(rows, value, device, left=False):
    return pad_sequence([torch.as_tensor(row, device=device) for row in rows],
                        batch_first=True, padding_value=value, padding_side="left" if left else "right")


def _tokens(prompts, completions, pad_id, device):
    return {
        "prompt_ids": _pad(prompts, pad_id, device, left=True),
        "prompt_mask": _pad([[1] * len(row) for row in prompts], 0, device, left=True),
        "completion_ids": _pad(completions, pad_id, device),
        "completion_mask": _pad([[1] * len(row) for row in completions], 0, device),
    }


def prepared_inputs(batch: ReplayBatch, pad_id: int, device) -> dict:
    rows = batch.rows
    tokens = _tokens([g.prompt_ids for g in batch.groups for _ in g.rollouts],
                     [r.completion_ids for r in rows], pad_id, device)
    action_mask = _pad([r.action_mask for r in rows], False, device)
    old = _pad([r.training_logp_b for r in rows], 0.0, device)
    sampled = _pad([r.sampling_logp_b for r in rows], 0.0, device)
    # A positive frozen c commutes with the minimum in stock TRL's clipped surrogate.
    correction = (old - sampled).clamp(max=math.log(2.0)).exp()
    return dict(tokens, old_per_token_logps=old, sampling_per_token_logps=sampled,
                advantages=torch.tensor([r.advantage for r in rows], device=device)[:, None] * correction,
                tool_mask=action_mask, num_items_in_batch=action_mask.sum())


def drift_rejection(current, old, mask):
    delta = (current - old)[mask.bool()].float()
    if not delta.numel() or not torch.isfinite(delta.exp()).all():
        return "invalid_ratio"
    outside = (delta < math.log(0.8)) | (delta > math.log(1.2))
    return "drift" if outside.sum().item() > 0.30 * delta.numel() else None


def _reward_counts(groups):
    rows = [r for group in groups for r in group.rollouts]
    return {
        "groups": len(groups),
        "completions": len(rows),
        "rewarded_completions": sum(r.reward != 0 for r in rows),
        "nonzero_reward_groups": sum(any(r.reward != 0 for r in g.rollouts) for g in groups),
        "mixed_reward_groups": sum(len({r.reward for r in g.rollouts}) > 1 for g in groups),
        "nonzero_advantage_groups": sum(any(r.advantage != 0 for r in g.rollouts) for g in groups),
        "truncated_completions": sum(r.truncated for r in rows),
    }


class _RecordUpdate(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.trainer._record_update()


class ReplayGRPOAdapter(GRPOTrainer):
    """Keep stock optimizer, gradient accumulation and DAPO loss; supply frozen groups."""

    def __init__(self, *args, **kwargs):
        if version("trl") != "1.15.0":
            raise RuntimeError("replay requires trl==1.15.0")
        super().__init__(*args, **kwargs)
        if (self.accelerator.num_processes != 1 or self.use_vllm or self._is_vlm or self.tools
                or self.loss_type != "dapo" or self.beta != 0 or self.num_iterations != 1
                or self.args.steps_per_generation != self.args.gradient_accumulation_steps
                or self.mask_truncated_completions or self.scale_rewards != "group"
                or self.epsilon_low != 0.2 or self.epsilon_high != 0.2):
            raise ValueError("unsupported replay trainer profile")
        gen = self.generation_config
        if (not gen.do_sample or gen.temperature != 1 or gen.top_p != 1 or gen.top_k not in (None, 0)
                or gen.min_p not in (0, None) or gen.repetition_penalty != 1 or gen.num_beams not in (1, None)):
            raise ValueError("unsupported replay sampling settings")
        self._branch = uuid4().hex
        self._model_config_digest = sha256(self.model.config.to_json_string(use_diff=False).encode()).hexdigest()
        self._tokenizer_digest = sha256(self._tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
        self._sampler_digest = sha256(gen.to_json_string(use_diff=False).encode()).hexdigest()
        self.replay_store = ReferenceReplayStore(group_size=self.num_generations)
        self._pending = None
        self.replay_metrics = {"steps": [], "phase_seconds": {}}
        self.add_callback(_RecordUpdate(self))

    @staticmethod
    def prepare_batch(groups, current_policy):
        return ReplayBatch(tuple(groups), current_policy)

    def _now(self):
        if self.accelerator.device.type == "cuda":
            torch.cuda.synchronize(self.accelerator.device)
        return monotonic()

    @contextmanager
    def _phase(self, name):
        start = self._now()
        try:
            yield
        finally:
            totals = self.replay_metrics["phase_seconds"]
            totals[name] = totals.get(name, 0.0) + self._now() - start

    @torch.no_grad()
    def _sample(self, rows):
        prompts, _, _ = self._tokenize_prompts([row["prompt"] for row in rows])
        n, device = self.num_generations, self.accelerator.device
        ids = _pad(prompts, self._tokenizer.pad_token_id, device, left=True).repeat_interleave(n, dim=0)
        mask = _pad([[1] * len(p) for p in prompts], 0, device, left=True).repeat_interleave(n, dim=0)
        if ids.shape[1] > self.repriserl_config.prompt_token_cap:
            raise ValueError("prompt exceeds configured token cap")
        with (unwrap_model_for_generation(self.model_wrapped, self.accelerator,
                                          generation_kwargs=self.generation_kwargs) as model,
              self.accelerator.autocast()):
            out = model.generate(input_ids=ids, attention_mask=mask, generation_config=self.generation_config,
                                 num_beams=1, return_dict_in_generate=True, output_scores=True)
        completion = out.sequences[:, ids.shape[1]:]
        if len(out.scores) != completion.shape[1] or not out.scores:
            raise ValueError("generator omitted per-token sampling scores")
        # Process one decoding position at a time to avoid a second vocabulary-sized stack.
        logps = torch.stack([score.float().log_softmax(-1).gather(1, completion[:, i:i+1]).squeeze(1)
                             for i, score in enumerate(out.scores)], dim=1).cpu().tolist()
        eos, sequences = set(self.eos_token_ids), completion.tolist()
        samples = []
        for i, (row, prompt) in enumerate(zip(rows, prompts, strict=True)):
            tokens, probabilities, truncated = [], [], []
            for j in range(i * n, (i + 1) * n):
                raw = sequences[j]
                end = next((k + 1 for k, tok in enumerate(raw) if tok in eos), len(raw))
                tokens.append(raw[:end])
                probabilities.append(logps[j][:end])
                truncated.append(raw[end - 1] not in eos)
            samples.append((row, prompt, tokens, probabilities, truncated))
        return samples

    @torch.no_grad()
    def _teacher_score(self, tokens):
        ids = torch.cat([tokens["prompt_ids"], tokens["completion_ids"]], 1)
        mask = torch.cat([tokens["prompt_mask"], tokens["completion_mask"]], 1)
        with disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            logps, _, _ = self._get_per_token_logps_and_entropies(
                self.model, ids, mask, tokens["completion_ids"].shape[1],
                batch_size=self.args.per_device_train_batch_size)
        return logps.detach().float()

    def _freeze(self, sample, policy, gid):
        row, prompt, ids, sampled, truncated = sample
        tensors = _tokens([prompt] * len(ids), ids, self._tokenizer.pad_token_id, self.accelerator.device)
        training = self._teacher_score(tensors).tolist()
        texts = self._tokenizer.batch_decode(ids, skip_special_tokens=True)
        rewards = reward_completions(completions=texts, numbers=[row["numbers"]] * len(ids),
                                     target=[row["target"]] * len(ids))
        advantages = compute_advantages(torch.tensor(rewards, dtype=torch.float32), std_correction=1).tolist()
        rows = tuple(Rollout(tokens, [True] * len(tokens), sampling, old[:len(tokens)], reward, advantage, trunc)
                     for tokens, sampling, old, reward, advantage, trunc in
                     zip(ids, sampled, training, rewards, advantages, truncated, strict=True))
        return RolloutGroup(gid, row["problem_id"], prompt, policy, rows)

    def _generate_and_score_completions(self, inputs):
        if not self.model.training or self._pending is not None:
            raise RuntimeError("replay preparation requires an idle training step boundary")
        config, step = self.repriserl_config, self.state.global_step
        model_id = (f"tiny:{self._model_config_digest}:seed={config.seed}" if self.args.use_cpu
                    else f"{config.model}@{config.model_revision}")
        policy = PolicyIdentity(self._branch, step, model_id,
                                self._tokenizer_digest, VERIFIER_DIGEST, self._sampler_digest)
        batch_id = f"{self._branch}:{step}"
        seed = config.seed + step
        with self._phase("plan"):
            n = self.num_generations
            groups = [inputs[i:i+n] for i in range(0, len(inputs), n)]
            if len(groups) != config.batch_groups or any(
                    len(g) != n or any(r["problem_id"] != g[0]["problem_id"] for r in g) for g in groups):
                raise ValueError("expected a full batch of consecutive repeated prompt groups")
            quota = len(groups) // 2 if step >= 8 else 0
            request = ReserveRequest(batch_id, policy, {"all": quota}, seed, 0)
            reservation = self.replay_store.reserve(request)
        try:
            with self._phase("rescore_replay"):
                accepted, rejected = [], {}
                for gid in reservation.group_ids:
                    group = self.replay_store.get_group(gid)
                    tensors = prepared_inputs(self.prepare_batch([group], policy), self._tokenizer.pad_token_id,
                                              self.accelerator.device)
                    reason = drift_rejection(self._teacher_score(tensors), tensors["old_per_token_logps"],
                                             tensors["tool_mask"])
                    if reason:
                        rejected[gid] = reason
                    else:
                        accepted.append(group)
            with self._phase("generate_fresh"):
                fresh_count = len(groups) - len(accepted)
                self.replay_store.ensure_capacity(fresh_count)
                samples = self._sample([g[0] for g in groups[:fresh_count]])
            with self._phase("verify_score_fresh"):
                fresh = [self._freeze(s, policy, f"{batch_id}:{i}") for i, s in enumerate(samples)]
                batch = self.prepare_batch(accepted + fresh, policy)
                result = prepared_inputs(batch, self._tokenizer.pad_token_id, self.accelerator.device)
            record = dict(step=step, batch_id=batch_id, selection_seed=seed, replay_quota=quota,
                          candidates=reservation.group_ids, accepted=[g.group_id for g in accepted],
                          rejected=rejected, fresh=[g.group_id for g in fresh], valid_tokens=batch.valid_tokens,
                          reward_counts={"fresh": _reward_counts(fresh), "replay": _reward_counts(accepted)})
            self._pending = record, fresh, self._now()
            return result
        except BaseException:
            self.replay_store.release(batch_id, 0)
            raise

    def _record_update(self):
        if self._pending is None:
            raise RuntimeError("optimizer step has no prepared replay batch")
        record, fresh, start = self._pending
        totals = self.replay_metrics["phase_seconds"]
        totals["train"] = totals.get("train", 0.) + self._now() - start
        with self._phase("record"):
            if self.accelerator.optimizer_step_was_skipped:
                self.replay_store.release(record["batch_id"], 0)
                raise RuntimeError("optimizer skipped replay update")
            self.replay_store.record_applied(record["batch_id"], 0, tuple(record["accepted"]))
            for group in fresh:
                self.replay_store.put_group(group)
            self.replay_metrics["steps"].append(record)
            self._pending = None

    def _compute_loss(self, model, inputs):
        loss = super()._compute_loss(model, inputs)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("nonfinite replay loss")
        return loss
