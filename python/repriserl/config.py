"""CPU-only run settings. Defaults describe a candidate, not a tuned baseline.

Numerical mask/std choices and model revision remain unset until integration.
The supplied example's 100-step limit is a bounded development budget.
"""
from dataclasses import dataclass, fields
import math
from pathlib import Path
import re
from typing import Any

import yaml


@dataclass(frozen=True)
class RunConfig:
    """Validated immutable configuration; at least one run budget is mandatory.

    If both budgets are set, a trainer must stop at whichever is reached first.
    Training consumers must require the unset conventions and immutable revision
    before launching matched experiments; this foundation does not launch runs.
    """

    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str | None = None
    requested_gpu_count: int = 1
    prompt_token_cap: int = 512
    completion_token_cap: int = 1024
    group_size: int = 4
    batch_groups: int = 8
    microbatch_size: int = 1
    learning_rate: float = 1e-6
    beta: float = 0.0
    seed: int = 0
    max_steps: int | None = None
    max_time_seconds: float | None = None
    std_convention: str | None = None
    eos_handling: str | None = None
    truncation_handling: str | None = None
    normalization: str | None = None

    def __post_init__(self) -> None:
        for name in ("requested_gpu_count", "prompt_token_cap", "completion_token_cap",
                     "group_size", "batch_groups", "microbatch_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.requested_gpu_count != 1:
            raise ValueError("requested_gpu_count must be exactly one")
        if self.microbatch_size > self.batch_groups * self.group_size:
            raise ValueError("microbatch_size exceeds the optimizer batch")
        if (self.batch_groups * self.group_size) % self.microbatch_size:
            raise ValueError("microbatch_size must divide the optimizer batch")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("learning_rate", "beta", "max_time_seconds"):
            value = getattr(self, name)
            if name == "max_time_seconds" and value is None:
                continue
            try:
                finite = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{name} must be finite and numeric")
            if value < 0 or (name != "beta" and value == 0):
                raise ValueError(f"{name} is outside its valid bounds")
        if self.max_steps is not None and (type(self.max_steps) is not int or self.max_steps <= 0):
            raise ValueError("max_steps must be a positive integer")
        if self.max_steps is None and self.max_time_seconds is None:
            raise ValueError("max_steps or max_time_seconds is required")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty string")
        if self.model_revision is not None and (
            not isinstance(self.model_revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", self.model_revision) is None
        ):
            raise ValueError("model_revision must be an immutable 40-character Hub commit SHA")
        # Optional choices are vocabulary only; no convention is silently chosen.
        choices = {
            "std_convention": {"population", "sample"},
            "eos_handling": {"include", "exclude"},
            "truncation_handling": {"include", "exclude"},
            "normalization": {"optimizer_batch"},
        }
        for name, allowed in choices.items():
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or value not in allowed):
                raise ValueError(f"{name} must be one of {sorted(allowed)} or unset")


class _UniqueSafeLoader(yaml.SafeLoader):
    """Reject duplicate keys, including duplicates introduced by YAML merges."""


def _mapping(loader: _UniqueSafeLoader, node: yaml.MappingNode) -> dict[str, Any]:
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            raise ValueError("configuration keys must be strings")
        if key in result:
            raise ValueError(f"duplicate configuration key: {key}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_config(path: Path) -> RunConfig:
    """Read a safe YAML mapping and reject every undeclared key.

    Filesystem errors propagate as OSError; invalid configuration raises ValueError.
    """
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueSafeLoader)
    except yaml.YAMLError as error:
        raise ValueError("invalid YAML configuration") from error
    if not isinstance(data, dict):
        raise ValueError("configuration must be a mapping")
    unknown = data.keys() - {field.name for field in fields(RunConfig)}
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    return RunConfig(**data)
