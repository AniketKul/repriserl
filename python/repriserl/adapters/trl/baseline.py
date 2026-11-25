"""Single-device stock GRPO training and shared evaluation for replay experiments."""
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from functools import wraps
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from uuid import uuid4

from repriserl.config import RunConfig
from repriserl.data.problems import ProblemSplits, make_splits
from repriserl.metrics import PhaseMeter
from repriserl.verifiers.number_expression import Problem, VERIFIER_DIGEST, verify_expression

from .models import (PROMPT_EXAMPLES, enable_tiny_cpu_scores, load_model_and_tokenizer,
                     problem_prompt, reward_completions)

_CONVENTIONS = {"std_convention": "sample", "eos_handling": "include",
                "truncation_handling": "include", "normalization": "optimizer_batch"}


def _checked_config(config: RunConfig) -> RunConfig:
    if config.max_steps is None:
        raise ValueError("training requires an explicit max_steps budget")
    for name, expected in _CONVENTIONS.items():
        if getattr(config, name) not in (None, expected):
            raise ValueError(f"stock TRL profile requires {name}={expected}")
    if config.beta != 0 or config.group_size < 2:
        raise ValueError("matched GRPO requires beta=0 and at least two completions per group")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("only single-process training is supported")
    return replace(config, **_CONVENTIONS)


class _WallTimeBudget(TimeoutError):
    pass


@contextmanager
def _deadline(seconds):
    """Interrupt CPU or single-host Unix training at the wall budget."""
    if seconds is None:
        yield
        return
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError("cannot replace an active process timer")
    def exhausted(signum, frame):
        raise _WallTimeBudget("training wall-time budget reached")
    old = signal.signal(signal.SIGALRM, exhausted)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def make_trainer(config: RunConfig, output_dir: Path, *, tiny: bool, num_iterations: int,
                 splits: ProblemSplits, trainer_class=None, callbacks=(), model_tokenizer=None):
    """Construct stock TRL or the replay subclass with identical training settings."""
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    config = _checked_config(config)
    if version("trl") != "1.15.0":
        raise RuntimeError("the validated trainer API requires trl==1.15.0")
    if type(num_iterations) is not int or num_iterations not in (1, 2, 4):
        raise ValueError("num_iterations must be 1, 2 or 4")
    if len(splits.train) < config.batch_groups or not splits.development:
        raise ValueError("at least batch_groups train problems and one development problem required")
    model, tokenizer = model_tokenizer or load_model_and_tokenizer(config, tiny=tiny)
    if tokenizer.chat_template:
        examples = {p.id for p, _ in PROMPT_EXAMPLES}
        if examples.intersection(splits.development_ids) or examples.intersection(splits.test_ids):
            raise ValueError("a held-out problem matches a prompt demonstration")
    prompts = [problem_prompt(problem, tokenizer) for problem in splits.train]
    # This TRL release does not apply a prompt length limit to plain text prompts.
    for group in (prompts, [problem_prompt(problem, tokenizer) for problem in splits.development]):
        for start in range(0, len(group), 256):
            ids = tokenizer(group[start:start + 256], return_attention_mask=False)["input_ids"]
            if any(len(row) > config.prompt_token_cap for row in ids):
                raise ValueError("an arithmetic prompt exceeds prompt_token_cap")
    dataset = Dataset.from_dict({
        "prompt": prompts, "numbers": [p.numbers for p in splits.train],
        "target": [p.target for p in splits.train], "problem_id": list(splits.train_ids),
    })
    batch_size = config.batch_groups * config.group_size
    args = GRPOConfig(
        output_dir=str(output_dir), use_cpu=tiny, bf16=not tiny, fp16=False,
        seed=config.seed, data_seed=config.seed, max_steps=config.max_steps,
        per_device_train_batch_size=config.microbatch_size, generation_batch_size=batch_size,
        gradient_accumulation_steps=batch_size // config.microbatch_size,
        num_generations=config.group_size, num_iterations=num_iterations,
        loss_type="dapo", scale_rewards="group", importance_sampling_level="token",
        beta=0.0, epsilon=0.2, epsilon_high=0.2, mask_truncated_completions=False,
        max_completion_length=config.completion_token_cap, temperature=1.0, top_p=1.0,
        top_k=0, repetition_penalty=1.0, disable_dropout=True, use_vllm=False,
        activation_checkpointing=not tiny, optim="adamw_torch", lr_scheduler_type="constant",
        learning_rate=config.learning_rate, weight_decay=0.0,
        report_to="none", save_strategy="no", logging_steps=1, disable_tqdm=True,
        dataloader_pin_memory=not tiny,
    )
    trainer = (trainer_class or GRPOTrainer)(
        model=model, args=args, train_dataset=dataset, processing_class=tokenizer,
        reward_funcs=reward_completions, callbacks=list(callbacks),
    )
    if tiny:
        enable_tiny_cpu_scores(trainer.model)
    trainer.repriserl_prompt_digest = prompt_digest(tokenizer)
    trainer.repriserl_config = config
    return trainer


def prompt_digest(tokenizer) -> str:
    fixture = problem_prompt(Problem((1, 3, 7, 11), 24), tokenizer)
    return sha256(fixture.encode("utf-8")).hexdigest()


def _instrument(trainer, meter, tiny):
    """Time disjoint stock calls, including device completion at phase boundaries."""
    import torch
    active = False
    def track(obj, method_name, label):
        original = getattr(obj, method_name)
        @wraps(original)
        def timed(*args, **kwargs):
            nonlocal active
            if active:
                return original(*args, **kwargs)
            active = True
            try:
                if not tiny:
                    torch.cuda.synchronize()
                with meter.phase(label):
                    result = original(*args, **kwargs)
                    if not tiny:
                        torch.cuda.synchronize()
                return result
            finally:
                active = False
        setattr(obj, method_name, timed)
    for method, phase in (("_generate_single_turn", "generation"), ("_calculate_rewards", "verify"),
                          ("_compute_loss", "loss_forward"),
                          ("_get_per_token_logps_and_entropies", "policy_scores")):
        track(trainer, method, phase)
    track(trainer.accelerator, "backward", "backward")
    return track


def _evaluate(trainer, problems, token_cap):
    return greedy_correct(trainer.model, trainer.processing_class, problems, token_cap,
                          trainer.eos_token_ids, trainer.accelerator.autocast(),
                          prompt_cap=trainer.repriserl_config.prompt_token_cap)


def greedy_correct(model, tokenizer, problems, token_cap, eos_ids, autocast, *, prompt_cap):
    import torch
    from transformers import GenerationConfig
    was_training = model.training
    model.eval()
    num_correct = 0
    greedy = GenerationConfig(max_new_tokens=token_cap, do_sample=False, use_cache=True,
                              eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id)
    try:
        with torch.inference_mode(), autocast:
            for problem in problems:
                encoded = tokenizer(problem_prompt(problem, tokenizer), return_tensors="pt",
                                    return_token_type_ids=False).to(model.device)
                if encoded.input_ids.shape[1] > prompt_cap:
                    raise ValueError("an evaluation prompt exceeds prompt_token_cap")
                tokens = model.generate(**encoded, generation_config=greedy)
                completion = tokenizer.decode(tokens[0, encoded.input_ids.shape[1]:],
                                              skip_special_tokens=True)
                num_correct += int(verify_expression(problem, completion).reward)
    finally:
        model.train(was_training)
    return num_correct


def checkpoint_digest(path: Path) -> str:
    """Bind checkpoint contents and names, including weight shards and tokenizer files."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError("checkpoint must be a local directory")
    manifest = []
    for file in sorted(path.rglob("*")):
        if file.is_symlink() or not (file.is_dir() or file.is_file()):
            raise ValueError("checkpoint contains a link or non-regular file")
        if file.is_file():
            with file.open("rb") as stream:
                manifest.append((file.relative_to(path).as_posix(), file_digest(stream, "sha256").hexdigest()))
    return sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()


def _save_policy(trainer, path: Path) -> str:
    if os.path.lexists(path):
        raise FileExistsError(f"checkpoint already exists: {path.name}")
    with TemporaryDirectory(prefix=".saving-policy-", dir=path.parent) as staging:
        trainer.model.save_pretrained(staging)
        trainer.processing_class.save_pretrained(staging)
        digest = checkpoint_digest(Path(staging))
        Path(staging).rename(path)
    return digest


def _write_summary(path: Path, summary: dict) -> None:
    tmp = path.with_name("." + summary["run_id"] + ".tmp")
    tmp.write_text(json.dumps(summary, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_training(config: RunConfig, output_dir: Path, *, tiny: bool, num_iterations: int,
                 sizes: tuple[int, int, int], arm: str = "baseline",
                 eval_every: int = 1, callbacks=()) -> dict:
    """Write one result for a bounded run and re-raise failures after recording them."""
    meter = PhaseMeter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    summary = dict(schema_version=1, run_id=uuid4().hex, arm=arm, fixture=tiny,
                   device="cpu" if tiny else "cuda", seed=config.seed, config=asdict(config),
                   steps=0, development=[], status="running", num_iterations=num_iterations,
                   backend="transformers", verifier_digest=VERIFIER_DIGEST, eval_every=eval_every)
    if any(os.path.lexists(output_dir / name) for name in ("model", "target-model")):
        raise FileExistsError("the run directory already contains a saved policy")
    with path.open("x") as started:
        json.dump(summary, started)
    summary["device_sharing"] = "cpu" if tiny else "unverified"
    summary["score_head"] = "dense_cpu_fixture" if tiny else "trl_fused"
    trainer = observation = None
    try:
        config = _checked_config(config)
        with _deadline(config.max_time_seconds), ExitStack() as resources:
            if not tiny:
                from repriserl.gpu_run import observe_run
                observation = resources.enter_context(observe_run(config.allow_shared_gpu))
            from transformers import TrainerCallback
            import torch
            summary["config"] = asdict(config)
            if type(tiny) is not bool:
                raise ValueError("tiny must be a boolean")
            if type(eval_every) is not int or eval_every < 1:
                raise ValueError("eval_every must be a positive integer")
            with meter.phase("setup"):
                splits = make_splits(config.seed, sizes)
                summary["split_sizes"] = list(sizes)
                summary["split_digest"] = splits.canonical_digest
                summary["dev_size"] = len(splits.development)
                summary["versions"] = {pkg: version(pkg) for pkg in (
                    "torch", "trl", "transformers", "accelerate", "datasets", "tokenizers")}
                trainer_class = None
                if arm == "fixed_replay":
                    from .compat import ReplayGRPOAdapter
                    trainer_class = ReplayGRPOAdapter
                trainer = make_trainer(config, output_dir, tiny=tiny, num_iterations=num_iterations,
                                       splits=splits, trainer_class=trainer_class, callbacks=callbacks)
            summary["generation_eos_ids"] = list(trainer.eos_token_ids)
            summary["prompt_digest"] = trainer.repriserl_prompt_digest
            track = _instrument(trainer, meter, tiny)
            class Evaluate(TrainerCallback):
                def on_train_begin(self, args, state, control, optimizer, **kwargs):
                    track(optimizer, "step", "optimizer")

                def on_step_end(self, args, state, control, **kwargs):
                    summary["steps"] = state.global_step
                    if state.global_step % eval_every == 0 or state.global_step == state.max_steps:
                        with meter.phase("development_evaluation"):
                            correct = _evaluate(trainer, splits.development, config.completion_token_cap)
                        accuracy = correct / len(splits.development)
                        hit = (config.target_accuracy is not None and accuracy >= config.target_accuracy
                               and "target_checkpoint" not in summary)
                        if hit:
                            with meter.phase("save_target_model"):
                                target_digest = _save_policy(trainer, output_dir / "target-model")
                        elapsed = meter.elapsed()
                        if hit:
                            summary["target_checkpoint"] = {
                                "model_path": "target-model", "sha256": target_digest, "step": state.global_step,
                                "accuracy": accuracy, "target_accuracy": config.target_accuracy,
                                "elapsed_seconds": elapsed,
                            }
                        summary["development"].append({
                            "step": state.global_step, "elapsed_seconds": elapsed, "correct": correct,
                            "total": len(splits.development), "split": "development",
                        })
                    summary["elapsed_seconds"] = meter.elapsed()
                    _write_summary(path, summary)
            trainer.add_callback(Evaluate())
            result = trainer.train()
            if not tiny:
                torch.cuda.synchronize()
                summary["cuda_allocator_peak_bytes"] = torch.cuda.max_memory_allocated()
            if not math.isfinite(result.training_loss):
                raise FloatingPointError("non-finite training loss")
            summary["optimizer_loss"] = result.training_loss
            if result.global_step != trainer.args.max_steps:
                raise RuntimeError("trainer exited before its requested update count")
            with meter.phase("save_model"):
                summary["model_sha256"] = _save_policy(trainer, output_dir / "model")
            summary["model_path"] = "model"
            summary["status"] = "completed"
    except BaseException as exc:
        reason = "wall_time_budget" if isinstance(exc, _WallTimeBudget) else "error"
        summary.update(status="failed", stop_reason=reason, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        timing = meter.finish()
        if observation is not None:
            summary.update(device_sharing=observation.label, occupancy_samples=observation.samples)
        if trainer is not None:
            summary["steps"] = trainer.state.global_step
            summary["training_metrics"] = [
                {key: record[key] for key in ("step", "loss", "grad_norm", "reward",
                                               "reward_std", "frac_reward_zero_std")
                 if key in record and math.isfinite(record[key])}
                for record in trainer.state.log_history if "loss" in record
            ]
            if hasattr(trainer, "replay_metrics"):
                summary["replay_metrics"] = trainer.replay_metrics
        summary.update(elapsed_seconds=timing["wall_seconds"], phase_seconds=timing["phase_seconds"])
        _write_summary(path, summary)
    return summary


def run_baseline(config: RunConfig, output_dir: Path, *, tiny: bool = False, num_iterations: int = 1,
                 sizes: tuple[int, int, int] = (20_000, 512, 2_048), eval_every: int = 1) -> dict:
    return run_training(config, output_dir, tiny=tiny, num_iterations=num_iterations,
                        sizes=sizes, eval_every=eval_every)
