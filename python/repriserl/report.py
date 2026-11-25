"""Compare observed development trajectories from completed or interrupted runs."""
import argparse
import json
import math
import re
from pathlib import Path
from typing import Sequence


def _number(value, name, *, integer=False, positive=False):
    allowed = (int,) if integer else (int, float)
    if type(value) not in allowed or (type(value) is float and not math.isfinite(value)) or value < 0 or (positive and value == 0):
        raise ValueError(f"invalid {name}")
    return value


def _row(summary: dict, target: float) -> dict:
    if summary.get("schema_version") != 1 or not summary.get("run_id"):
        raise ValueError("missing run identity or unsupported summary schema")
    status = summary["status"]
    if status not in {"completed", "skipped", "failed", "running"}:
        raise ValueError("unknown run status")
    elapsed = summary.get("elapsed_seconds")
    if elapsed is not None or status == "completed":
        _number(elapsed, "elapsed_seconds", positive=status == "completed")
    steps = _number(summary["steps"], "steps", integer=True)
    _number(summary["seed"], "seed", integer=True)
    if status == "completed":
        cadence = _number(summary.get("eval_every"), "eval_every", integer=True, positive=True)
        dev_size = _number(summary.get("dev_size"), "dev_size", integer=True, positive=True)
        config = summary.get("config")
        if isinstance(config, dict) and config.get("max_steps") is not None and steps != config["max_steps"]:
            raise ValueError("completed run did not reach its configured step count")
        if steps < 1:
            raise ValueError("completed run has zero optimizer steps")
    trajectory = {}
    for event in summary["development"]:
        if event["split"] != "development":
            raise ValueError("target selection requires development evaluations")
        step = _number(event["step"], "evaluation step", integer=True, positive=True)
        seconds = _number(event["elapsed_seconds"], "evaluation time", positive=True)
        n = _number(event["total"], "evaluation total", integer=True, positive=True)
        correct = _number(event["correct"], "evaluation correct", integer=True)
        if correct > n or step > steps or (elapsed is not None and seconds > elapsed):
            raise ValueError("evaluation exceeds run bounds")
        if status == "completed" and (n != dev_size or (step != steps and step % cadence)):
            raise ValueError("evaluation differs from declared development size or cadence")
        point = (seconds, correct, n)
        if trajectory.setdefault(step, point) != point:
            raise ValueError("conflicting evaluation at same step")
    times = [v[0] for _, v in sorted(trajectory.items())]
    if status == "completed" and len(times) != (steps + cadence - 1) // cadence:
        raise ValueError("incomplete development evaluation schedule")
    if any(prev >= nxt for prev, nxt in zip(times, times[1:])):
        raise ValueError("evaluation times must increase with training steps")
    reached = next((t for _, (t, c, n) in sorted(trajectory.items()) if c / n >= target), None)
    last = trajectory[max(trajectory)] if trajectory else None
    return {"run_id": summary["run_id"], "arm": summary["arm"], "seed": summary["seed"],
            "status": status, "fixture": summary["fixture"], "device": summary["device"],
            "steps": steps, "device_sharing": summary.get("device_sharing"),
            "elapsed_seconds": elapsed, "time_to_target_seconds":
            reached if status == "completed" else None,
            "development": [{"step": step, "elapsed_seconds": t, "correct": c, "total": n}
                            for step, (t, c, n) in sorted(trajectory.items())],
            "phase_seconds": summary.get("phase_seconds"), "num_iterations": summary.get("num_iterations"),
            "last_development_accuracy": last[1] / last[2] if last else None}


def _has_provenance(summary: dict) -> bool:
    digests = ("split_digest", "verifier_digest")
    if any(not isinstance(summary.get(k), str) or not re.fullmatch("[a-f0-9]{64}", summary[k])
           for k in digests):
        return False
    versions = summary.get("versions")
    if not isinstance(versions, dict) or not all(isinstance(versions.get(k), str) and versions[k].strip()
            for k in ("torch", "transformers", "trl", "accelerate", "datasets", "tokenizers")):
        return False
    config = summary.get("config")
    if not isinstance(config, dict) or not config.get("model") or summary.get("backend") != "transformers":
        return False
    device, sharing, fixture = summary.get("device"), summary.get("device_sharing"), summary.get("fixture")
    if device == "cpu" and fixture is True:
        return sharing == "cpu"
    return (device == "cuda" and fixture is False and sharing == "sampled_uncontended_h100"
            and type(summary.get("occupancy_samples")) is int and summary["occupancy_samples"] >= 2
            and isinstance(config.get("model_revision"), str)
            and re.fullmatch("[0-9a-f]{40}", config["model_revision"]) is not None)


def compare_runs(run_dirs: Sequence[Path], target: float) -> dict:
    """List every run and compare matched pairs. Missing endpoints yield null savings."""
    if not run_dirs or not 0 < _number(target, "target") <= 1:
        raise ValueError("provide run directories and a target in (0, 1]")
    summaries, rows = {}, []
    for directory in run_dirs:
        s = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        row = _row(s, target)
        if row["run_id"] in summaries:
            if summaries[row["run_id"]] != s:
                raise ValueError("conflicting summaries for one run_id")
            continue
        summaries[row["run_id"]] = s
        rows.append(row)
    pairs = []
    for base in (r for r in rows if r["arm"] == "baseline"):
        a = summaries[base["run_id"]]
        for replay in (r for r in rows if r["arm"] == "fixed_replay" and r["seed"] == base["seed"]):
            b = summaries[replay["run_id"]]
            match_fields = ("config", "device", "fixture", "split_digest", "verifier_digest",
                            "eval_every", "dev_size", "device_sharing", "backend", "versions")
            matched = (_has_provenance(a) and _has_provenance(b)
                       and all(k in a and k in b and a[k] == b[k] for k in match_fields))
            ta, tb = base["time_to_target_seconds"], replay["time_to_target_seconds"]
            savings = 1 - tb / ta if matched and ta and tb is not None else None
            pairs.append({"baseline": base["run_id"], "fixed_replay": replay["run_id"],
                          "matched": matched, "time_to_target_savings_fraction": savings})
    return {"target_development_accuracy": target, "runs": rows, "pairs": pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps(compare_runs(args.run_dirs, args.target), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
