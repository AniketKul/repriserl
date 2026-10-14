"""Versioned arithmetic verifier with no execution of completion text.

A completion is trimmed and parsed in its entirety. Grammar: positive ASCII
decimal integer literals, binary + - * /, parentheses, and ASCII whitespace.
Every input number is used exactly once; calculations use exact fractions.
Limits apply to raw input, tokens, parentheses, literals and fraction growth.
Problems have 1..8 positive numbers <=1,000,000 and integer targets in
[-1,000,000,000, 1,000,000,000]. IDs hash sorted numbers and target as JSON.
"""
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re

VERIFIER_VERSION = "number-expression-v1"
ANSWER_EXTRACTION_RULE = "trim-outer-whitespace-then-parse-entire-expression"
MAX_INPUT_LENGTH = 4096
MAX_TOKENS = 256
MAX_NESTING = 32
MAX_LITERAL = 1_000_000
MAX_FRACTION_BITS = 256
MAX_PROBLEM_NUMBERS = 8
MAX_ABS_TARGET = 1_000_000_000

VERIFIER_CONTRACT = json.dumps({
    "version": VERIFIER_VERSION,
    "extraction": ANSWER_EXTRACTION_RULE,
    "grammar": "positive-ascii-decimal;binary:+,-,*,/;parentheses;ascii-whitespace",
    "leading_zeros": False,
    "arithmetic": "fractions.Fraction;normal-precedence;left-associative",
    "number_accounting": "exact-multiset",
    "reward": "1.0 iff valid and exact target;0.0 otherwise",
    "limits": {
        "raw_characters": MAX_INPUT_LENGTH, "tokens": MAX_TOKENS,
        "nesting": MAX_NESTING, "literal": MAX_LITERAL,
        "fraction_bits": MAX_FRACTION_BITS, "problem_numbers": MAX_PROBLEM_NUMBERS,
        "absolute_target": MAX_ABS_TARGET,
    },
}, sort_keys=True, separators=(",", ":"))
VERIFIER_DIGEST = hashlib.sha256(VERIFIER_CONTRACT.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Problem:
    """Immutable canonical number multiset and integer target."""
    numbers: tuple[int, ...]
    target: int

    def __post_init__(self):
        if not isinstance(self.numbers, (list, tuple)):
            raise TypeError("numbers must be a list or tuple")
        if not 1 <= len(self.numbers) <= MAX_PROBLEM_NUMBERS:
            raise ValueError("unsupported number count")
        if any(type(n) is not int for n in self.numbers):
            raise TypeError("numbers must be integers, excluding bool")
        if any(not 1 <= n <= MAX_LITERAL for n in self.numbers):
            raise ValueError("numbers outside supported domain")
        if type(self.target) is not int:
            raise TypeError("target must be an integer, excluding bool")
        if abs(self.target) > MAX_ABS_TARGET:
            raise ValueError("target outside supported domain")
        object.__setattr__(self, "numbers", tuple(sorted(self.numbers)))

    @property
    def canonical_digest(self) -> str:
        payload = json.dumps(
            {"numbers": self.numbers, "target": self.target},
            sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    @property
    def id(self) -> str:
        return "number-expression:" + self.canonical_digest


@dataclass(frozen=True)
class VerificationResult:
    """Valid means legal grammar and exact number use, regardless of target."""
    valid: bool
    reward: float
    reason: str
    value: Fraction | None = None
    verifier_digest: str = VERIFIER_DIGEST


class _Rejected(Exception):
    def __init__(self, reason: str):
        self.reason = reason


_TOKEN = re.compile(r"[0-9]+|[+*/()-]")


def _tokenize(text: str) -> list[str]:
    tokens = []
    position = 0
    while position < len(text):
        if text[position] in " \t\r\n\f\v":
            position += 1
            continue
        match = _TOKEN.match(text, position)
        if match is None:
            raise _Rejected("malformed_expression")
        token = match.group()
        if token[0].isdigit():
            if token[0] == "0":
                raise _Rejected("malformed_expression")
            if len(token) > len(str(MAX_LITERAL)) or int(token) > MAX_LITERAL:
                raise _Rejected("resource_limit")
        tokens.append(token)
        if len(tokens) > MAX_TOKENS:
            raise _Rejected("resource_limit")
        position = match.end()
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.position = 0
        self.numbers: list[int] = []

    def peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def parse(self) -> Fraction:
        value = self.expression(0)
        if self.peek() is not None:
            raise _Rejected("malformed_expression")
        return value

    def expression(self, depth: int) -> Fraction:
        value = self.term(depth)
        while self.peek() in ("+", "-"):
            operator = self.peek()
            self.position += 1
            right = self.term(depth)
            value = self.bounded(value + right if operator == "+" else value - right)
        return value

    def term(self, depth: int) -> Fraction:
        value = self.primary(depth)
        while self.peek() in ("*", "/"):
            operator = self.peek()
            self.position += 1
            right = self.primary(depth)
            if operator == "/" and right == 0:
                raise _Rejected("division_by_zero")
            value = self.bounded(value * right if operator == "*" else value / right)
        return value

    def primary(self, depth: int) -> Fraction:
        token = self.peek()
        if token is None:
            raise _Rejected("malformed_expression")
        self.position += 1
        if token == "(":
            if depth >= MAX_NESTING:
                raise _Rejected("resource_limit")
            value = self.expression(depth + 1)
            if self.peek() != ")":
                raise _Rejected("malformed_expression")
            self.position += 1
            return value
        if token[0] not in "123456789":
            raise _Rejected("malformed_expression")
        number = int(token)
        self.numbers.append(number)
        return Fraction(number)

    @staticmethod
    def bounded(value: Fraction) -> Fraction:
        if (value.numerator.bit_length() > MAX_FRACTION_BITS
                or value.denominator.bit_length() > MAX_FRACTION_BITS):
            raise _Rejected("resource_limit")
        return value


def verify_expression(problem: Problem, completion: str) -> VerificationResult:
    """Return defined zero reward for rejected text; reject bad API types."""
    if not isinstance(problem, Problem):
        raise TypeError("problem must be a Problem")
    if not isinstance(completion, str):
        raise TypeError("completion must be a string")
    try:
        if len(completion) > MAX_INPUT_LENGTH:
            raise _Rejected("resource_limit")
        parser = _Parser(_tokenize(completion.strip()))
        value = parser.parse()
        if Counter(parser.numbers) != Counter(problem.numbers):
            raise _Rejected("number_mismatch")
    except _Rejected as rejection:
        return VerificationResult(False, 0.0, rejection.reason)
    correct = value == problem.target
    return VerificationResult(True, float(correct), "correct" if correct else "wrong_target", value)
