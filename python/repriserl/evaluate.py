"""Score one selected local checkpoint on its original sealed split."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import time

from .adapters.trl.baseline import _checked_config, _deadline, checkpoint_digest, greedy_correct, prompt_digest
from .adapters.trl.models import PROMPT_EXAMPLES
from .config import RunConfig
from .data.problems import make_splits
from .gpu_run import observe_run
from .verifiers.number_expression import VERIFIER_DIGEST


def _select(summary, config, choice):
    if choice == "final":
        name, digest, step = summary.get("model_path"), summary.get("model_sha256"), summary["steps"]
    elif choice == "target":
        hit = summary.get("target_checkpoint")
        target = config.target_accuracy
        if not isinstance(hit, dict) or target is None or hit.get("target_accuracy") != target:
            raise ValueError("run lacks a checkpoint for its configured target")
        trajectory = summary.get("development", ())
        for event in trajectory:
            if (event.get("split") != "development" or event.get("total") != summary["dev_size"]
                    or not 0 <= event["correct"] <= event["total"]):
                raise ValueError("invalid development result in checkpoint selection")
            accuracy = event["correct"] / event["total"]
            if accuracy >= target:
                if (hit.get("step") != event["step"] or hit.get("accuracy") != accuracy
                        or hit.get("elapsed_seconds") != event.get("elapsed_seconds")):
                    raise ValueError("checkpoint differs from first qualifying development evaluation")
                break
        else:
            raise ValueError("no development evaluation reached the target")
        name, digest, step = hit.get("model_path"), hit.get("sha256"), hit["step"]
    else:
        raise ValueError("checkpoint must be target or final")
    if name != ("target-model" if choice == "target" else "model"):
        raise ValueError("unexpected checkpoint path")
    return name, digest, step


def _validate_run(summary):
    if summary.get("schema_version") != 1 or summary.get("status") != "completed":
        raise ValueError("a completed training summary is required")
    if (not isinstance(summary.get("run_id"), str) or not summary["run_id"]
            or summary.get("arm") not in {"baseline", "fixed_replay", "adaptive_replay"}
            or type(summary.get("steps")) is not int or type(summary.get("seed")) is not int):
        raise ValueError("invalid training run identity, arm, step count or seed")
    config = _checked_config(RunConfig(**summary["config"]))
    if asdict(config) != summary["config"] or config.seed != summary["seed"] or config.max_steps != summary["steps"]:
        raise ValueError("inconsistent training configuration, seed or completed step count")
    tiny = summary.get("fixture")
    if (type(tiny) is not bool or summary.get("device") != ("cpu" if tiny else "cuda")
            or summary.get("score_head") != ("dense_cpu_fixture" if tiny else "trl_fused")
            or summary.get("backend") != "transformers" or not tiny and config.model_revision is None):
        raise ValueError("unsupported training backend or model profile")
    if summary.get("verifier_digest") != VERIFIER_DIGEST:
        raise ValueError("verifier contract differs from training")
    for package in ("torch", "transformers", "trl", "accelerate", "datasets", "tokenizers"):
        if summary.get("versions", {}).get(package) != version(package):
            raise ValueError(f"installed {package} version differs from training")
    sizes = summary.get("split_sizes")
    if not isinstance(sizes, list) or len(sizes) != 3 or not sizes[2]:
        raise ValueError("a recorded nonempty sealed split is required")
    splits = make_splits(config.seed, tuple(sizes))
    if summary.get("split_digest") != splits.canonical_digest or len(splits.development) != summary.get("dev_size"):
        raise ValueError("regenerated split differs from training")
    return config, tiny, splits


def _validate_checkpoint(directory, digest):
    if checkpoint_digest(directory) != digest:
        raise ValueError("checkpoint checksum differs from training")
    if (directory / "adapter_config.json").exists():
        raise ValueError("only standalone local checkpoints are supported")
    index = directory / "model.safetensors.index.json"
    if index.exists():
        for shard in json.loads(index.read_text())["weight_map"].values():
            if (Path(shard).name != shard or not shard.endswith(".safetensors")
                    or not (directory / shard).is_file()):
                raise ValueError("invalid local checkpoint shard path")
    elif not (directory / "model.safetensors").is_file():
        raise ValueError("checkpoint requires safetensors weights")


def evaluate_checkpoint(run_dir: Path, *, checkpoint: str = "target", max_seconds: float = 600.0) -> dict:
    """Return aggregate sealed accuracy. Claim a single evaluation for the run."""
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    run_dir = Path(run_dir)
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("run directory must be a local directory")
    run_dir = run_dir.resolve()
    output = run_dir / "sealed-evaluation.json"
    if os.path.lexists(output):
        raise FileExistsError("a sealed evaluation already exists for this run")
    lock = run_dir / ".sealed-evaluation.lock"
    lock.mkdir()
    claimed = False
    observed = None
    start = time.monotonic()
    try:
        with _deadline(max_seconds):
            if os.path.lexists(output):
                raise FileExistsError("a sealed evaluation already exists for this run")
            summary_path = run_dir / "summary.json"
            if summary_path.is_symlink() or not summary_path.is_file():
                raise ValueError("training summary must be a regular local file")
            raw = summary_path.read_bytes()
            summary = json.loads(raw)
            config, tiny, splits = _validate_run(summary)
            name, digest, step = _select(summary, config, checkpoint)
            directory = run_dir / name
            _validate_checkpoint(directory, digest)
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
            if prompt_digest(tokenizer) != summary.get("prompt_digest"):
                raise ValueError("restored prompt differs from training")
            if tokenizer.chat_template and set(splits.test_ids).intersection(p.id for p, _ in PROMPT_EXAMPLES):
                raise ValueError("a sealed problem matches a prompt demonstration")
            with nullcontext() if tiny else observe_run(config.allow_shared_gpu) as observed:
                if not tiny and torch.cuda.device_count() != 1:
                    raise RuntimeError("one visible CUDA device is required")
                model = AutoModelForCausalLM.from_pretrained(directory, use_safetensors=True,
                    local_files_only=True, trust_remote_code=False, dtype=torch.float32).to(summary["device"])
                eos = model.generation_config.eos_token_id
                eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
                if tokenizer.eos_token_id not in eos_ids:
                    eos_ids.insert(0, tokenizer.eos_token_id)
                if eos_ids != summary.get("generation_eos_ids"):
                    raise ValueError("restored EOS contract differs from training")
                result = dict(schema_version=1, training_run_id=summary["run_id"],
                    training_summary_sha256=sha256(raw).hexdigest(), arm=summary["arm"],
                    checkpoint=checkpoint, target_accuracy=config.target_accuracy,
                    checkpoint_sha256=digest, step=step, split="test", split_digest=splits.test_digest,
                    dataset_digest=splits.canonical_digest, prompt_digest=summary["prompt_digest"],
                    verifier_digest=VERIFIER_DIGEST, fixture=tiny, device=summary["device"],
                    seed=config.seed, status="running")
                with output.open("x") as initial:
                    json.dump(result, initial)
                claimed = True
                correct = greedy_correct(
                    model, tokenizer, splits.test, config.completion_token_cap, eos_ids,
                    nullcontext() if tiny else torch.autocast("cuda", dtype=torch.bfloat16),
                    prompt_cap=config.prompt_token_cap,
                )
            result.update(status="completed", correct=correct, total=len(splits.test),
                          accuracy=correct / len(splits.test))
        return result
    except BaseException as exc:
        if claimed:
            result.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        try:
            if claimed:
                result["elapsed_seconds"] = time.monotonic() - start
                result["device_sharing"] = "cpu" if tiny else observed.label
                if observed is not None:
                    result["occupancy_samples"] = observed.samples
                tmp = lock / "result.json"
                try:
                    tmp.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
                    tmp.replace(output)
                finally:
                    tmp.unlink(missing_ok=True)
        finally:
            lock.rmdir()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", choices=("target", "final"), default="target")
    parser.add_argument("--max-seconds", type=float, default=600.0)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_checkpoint(args.run_dir, checkpoint=args.checkpoint,
                                         max_seconds=args.max_seconds), allow_nan=False))


if __name__ == "__main__":
    main()
