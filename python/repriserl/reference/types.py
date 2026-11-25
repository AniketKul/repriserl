"""Immutable generation records and the contracts used by the single-learner store."""
from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType


def _label(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} requires a nonempty string")


def _count(value: int, field: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")


def _tokens(values) -> tuple[int, ...]:
    ids = tuple(values)
    if not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("token IDs must be nonnegative integers in a nonempty sequence")
    return ids


def _number(value: float) -> bool:
    return type(value) in (float, int) and math.isfinite(value)


@dataclass(frozen=True, slots=True)
class PolicyIdentity:
    branch_id: str
    step: int
    model_id: str
    tokenizer_id: str
    verifier_id: str
    sampler_id: str

    def __post_init__(self):
        _count(self.step, "step")
        for label in self.lineage:
            _label(label, "policy identity")

    @property
    def lineage(self) -> tuple[str, ...]:
        return self.branch_id, self.model_id, self.tokenizer_id, self.verifier_id, self.sampler_id


@dataclass(frozen=True, slots=True)
class Rollout:
    completion_ids: tuple[int, ...]
    action_mask: tuple[bool, ...]
    sampling_logp_b: tuple[float, ...]
    training_logp_b: tuple[float, ...]
    reward: float
    advantage: float
    truncated: bool = False

    def __post_init__(self):
        ids = _tokens(self.completion_ids)
        mask = tuple(self.action_mask)
        if len(mask) != len(ids) or any(type(b) is not bool for b in mask):
            raise ValueError("action mask must have one boolean per completion token")
        if type(self.truncated) is not bool or not _number(self.reward) or not _number(self.advantage):
            raise ValueError("invalid reward, advantage or truncation flag")
        object.__setattr__(self, "completion_ids", ids)
        object.__setattr__(self, "action_mask", mask)
        for field in ("sampling_logp_b", "training_logp_b"):
            values = tuple(getattr(self, field))
            if len(values) != len(ids):
                raise ValueError(f"{field} length differs from completion")
            if any(not _number(x) or x > 0 for x, valid in zip(values, mask) if valid):
                raise ValueError(f"{field} must contain finite log probabilities on action tokens")
            # Padding carries no distribution. Canonicalize it for idempotent payload comparison.
            object.__setattr__(self, field, tuple(float(x) if b else 0.0 for x, b in zip(values, mask)))


@dataclass(frozen=True, slots=True)
class RolloutGroup:
    group_id: str
    task_id: str
    prompt_ids: tuple[int, ...]
    policy: PolicyIdentity
    rollouts: tuple[Rollout, ...]
    stratum: str = "all"

    def __post_init__(self):
        for field in ("group_id", "task_id", "stratum"):
            _label(getattr(self, field), field)
        if not isinstance(self.policy, PolicyIdentity):
            raise ValueError("a generating policy identity is required")
        rows = tuple(self.rollouts)
        if len(rows) < 2 or any(not isinstance(r, Rollout) for r in rows):
            raise ValueError("a group needs at least two rollout records")
        object.__setattr__(self, "rollouts", rows)
        object.__setattr__(self, "prompt_ids", _tokens(self.prompt_ids))


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    groups: tuple[RolloutGroup, ...]
    current_policy: PolicyIdentity

    def __post_init__(self):
        groups = tuple(self.groups)
        if not groups or any(not isinstance(g, RolloutGroup) for g in groups):
            raise ValueError("a batch needs complete groups")
        if len({g.group_id for g in groups}) != len(groups):
            raise ValueError("duplicate group ID")
        if len({len(g.rollouts) for g in groups}) != 1:
            raise ValueError("mismatched group sizes")
        if not isinstance(self.current_policy, PolicyIdentity) or any(
            g.policy.lineage != self.current_policy.lineage or g.policy.step > self.current_policy.step
            for g in groups
        ):
            raise ValueError("incompatible or future behavior policy")
        object.__setattr__(self, "groups", groups)
        if self.valid_tokens == 0:
            raise ValueError("batch has no valid action tokens")

    @property
    def rows(self) -> tuple[Rollout, ...]:
        return tuple(r for g in self.groups for r in g.rollouts)

    @property
    def valid_tokens(self) -> int:
        return sum(sum(r.action_mask) for g in self.groups for r in g.rollouts)


@dataclass(frozen=True, slots=True)
class ReserveRequest:
    batch_id: str
    current_policy: PolicyIdentity
    quotas: Mapping[str, int]
    seed: int
    epoch: int

    def __post_init__(self):
        _label(self.batch_id, "batch_id")
        _count(self.epoch, "epoch")
        _count(self.seed, "seed")
        if not isinstance(self.current_policy, PolicyIdentity) or not isinstance(self.quotas, Mapping):
            raise ValueError("a policy and stratum quotas are required")
        quotas = dict(self.quotas)
        for label, quota in quotas.items():
            _label(label, "stratum")
            _count(quota, "quota")
        object.__setattr__(self, "quotas", MappingProxyType(quotas))


@dataclass(frozen=True, slots=True)
class Reservation:
    batch_id: str
    group_ids: tuple[str, ...]
    unmet: Mapping[str, int]

    def __post_init__(self):
        object.__setattr__(self, "group_ids", tuple(self.group_ids))
        object.__setattr__(self, "unmet", MappingProxyType(dict(self.unmet)))


@dataclass(frozen=True, slots=True)
class PutResult:
    inserted: bool
