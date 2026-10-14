"""Versioned deterministic solvable arithmetic datasets.

Generation uses a private random.Random(seed), four numbers uniformly sampled
from 1..20, and a left fold of randomly chosen binary + - * / operations.
Retain only integral targets in 1..1000 and unique canonical problem IDs.
Maximum combined split size is 50,000; retries are bounded by
max(1,000, total * 100). A stalled generation raises RuntimeError and returns
no partial split. This is a reproducible task distribution, not a difficulty
calibration. No witness expression is returned.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import random

from repriserl.verifiers.number_expression import Problem

GENERATION_VERSION = "number-problems-v1"
NUMBER_COUNT = 4
NUMBER_MIN = 1
NUMBER_MAX = 20
TARGET_MIN = 1
TARGET_MAX = 1000
MAX_TOTAL_PROBLEMS = 50_000
MIN_ATTEMPTS = 1000
ATTEMPTS_PER_PROBLEM = 100
GENERATION_CONTRACT = json.dumps({
    "version": GENERATION_VERSION,
    "rng": "python-random.Random;private;randint-and-choice",
    "seed_range": [0, 2**64 - 1],
    "numbers": {"count": NUMBER_COUNT, "min": NUMBER_MIN, "max": NUMBER_MAX},
    "target": {"min": TARGET_MIN, "max": TARGET_MAX, "integral": True},
    "construction": "left-fold;ordered-draw;choice-from:+,-,*,/;skip-zero-divisor",
    "deduplication": "global-canonical-problem-id;first-occurrence",
    "allocation": "train-then-development-then-test",
    "limits": {"total": MAX_TOTAL_PROBLEMS, "min_attempts": MIN_ATTEMPTS,
               "attempts_per_problem": ATTEMPTS_PER_PROBLEM},
}, sort_keys=True, separators=(",", ":"))
GENERATION_DIGEST = hashlib.sha256(GENERATION_CONTRACT.encode("ascii")).hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class ProblemSplits:
    """Immutable disjoint splits; digests bind canonical membership."""
    train: tuple[Problem, ...]
    development: tuple[Problem, ...]
    test: tuple[Problem, ...]
    generation_digest: str = GENERATION_DIGEST

    def __post_init__(self):
        seen = set()
        for name in ("train", "development", "test"):
            problems = getattr(self, name)
            if not isinstance(problems, (list, tuple)):
                raise TypeError("splits must be lists or tuples")
            problems = tuple(problems)
            for problem in problems:
                if not isinstance(problem, Problem):
                    raise TypeError("split members must be Problems")
                if problem.id in seen:
                    raise ValueError("canonical problem overlap within or between splits")
                seen.add(problem.id)
            object.__setattr__(self, name, problems)
        if self.generation_digest != GENERATION_DIGEST:
            raise ValueError("unsupported generation digest")

    @property
    def train_ids(self) -> tuple[str, ...]:
        return tuple(p.id for p in self.train)

    @property
    def development_ids(self) -> tuple[str, ...]:
        return tuple(p.id for p in self.development)

    @property
    def test_ids(self) -> tuple[str, ...]:
        return tuple(p.id for p in self.test)

    @property
    def train_digest(self) -> str:
        return _digest(sorted(self.train_ids))

    @property
    def development_digest(self) -> str:
        return _digest(sorted(self.development_ids))

    @property
    def test_digest(self) -> str:
        return _digest(sorted(self.test_ids))

    @property
    def canonical_digest(self) -> str:
        return _digest({
            "generation_digest": self.generation_digest,
            "train": self.train_digest,
            "development": self.development_digest,
            "test": self.test_digest,
        })


def make_splits(seed: int, sizes: tuple[int, int, int]) -> ProblemSplits:
    """Generate disjoint tasks with bounded work and no global RNG mutation."""
    if type(seed) is not int:
        raise TypeError("seed must be an integer, excluding bool")
    if not 0 <= seed < 2**64:
        raise ValueError("seed outside supported unsigned 64-bit range")
    if not isinstance(sizes, tuple) or len(sizes) != 3:
        raise TypeError("sizes must be a tuple of three integers")
    if any(type(size) is not int for size in sizes):
        raise TypeError("sizes must be integers, excluding bool")
    if any(size < 0 for size in sizes) or sum(sizes) > MAX_TOTAL_PROBLEMS:
        raise ValueError("sizes outside supported range")
    total = sum(sizes)
    rng = random.Random(seed)
    problems = []
    seen = set()
    limit = max(MIN_ATTEMPTS, total * ATTEMPTS_PER_PROBLEM)
    for _ in range(limit if total else 0):
        numbers = tuple(rng.randint(NUMBER_MIN, NUMBER_MAX) for _ in range(NUMBER_COUNT))
        value = Fraction(numbers[0])
        for number in numbers[1:]:
            operator = rng.choice(("+", "-", "*", "/"))
            if operator == "+":
                value += number
            elif operator == "-":
                value -= number
            elif operator == "*":
                value *= number
            else:
                value /= number
        if value.denominator != 1 or not TARGET_MIN <= value <= TARGET_MAX:
            continue
        problem = Problem(numbers, value.numerator)
        if problem.id in seen:
            continue
        seen.add(problem.id)
        problems.append(problem)
        if len(problems) == total:
            break
    if len(problems) != total:
        raise RuntimeError("generation retry budget exhausted")
    train_end = sizes[0]
    development_end = train_end + sizes[1]
    return ProblemSplits(problems[:train_end], problems[train_end:development_end],
                         problems[development_end:])
