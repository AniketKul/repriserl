"""Verifier behavior: exact arithmetic, multiset accounting and bounded grammar."""
from dataclasses import FrozenInstanceError
from fractions import Fraction
import hashlib

import pytest

from repriserl.verifiers.number_expression import (
    ANSWER_EXTRACTION_RULE, VERIFIER_DIGEST, VERIFIER_VERSION,
    Problem, verify_expression,
)


@pytest.mark.parametrize("numbers,target,expression", [
    ([3, 3, 8, 8], 24, "8 / (3 - 8 / 3)"),
    ([2, 5, 3], -1, "(2 - 5) / 3"),
    ([2, 2, 3], 7, " 2 + 2 + 3 \n"),
    ([1, 2, 3], 7, "1 + 2 * 3"),
])
def test_exact_arithmetic_and_repeated_numbers(numbers, target, expression):
    result = verify_expression(Problem(numbers, target), expression)
    assert result.valid
    assert result.reward == 1.0
    assert result.value == Fraction(target)
    assert result.verifier_digest == VERIFIER_DIGEST


@pytest.mark.parametrize("expression", ["2+3", "2+2+2+3", "2*3+1"])
def test_missing_extra_or_substituted_numbers(expression):
    result = verify_expression(Problem([2, 2, 3], 7), expression)
    assert result.reward == 0.0
    assert not result.valid
    assert result.reason == "number_mismatch"


def test_valid_expression_with_wrong_target_gets_zero():
    result = verify_expression(Problem([2, 3], 6), "2+3")
    assert result.valid
    assert result.value == Fraction(5)
    assert result.reward == 0.0
    assert result.reason == "wrong_target"


def test_division_by_zero_is_defined_failure():
    result = verify_expression(Problem([1, 2, 2], 4), "1/(2-2)")
    assert not result.valid
    assert result.reward == 0.0
    assert result.reason == "division_by_zero"


@pytest.mark.parametrize("expression", [
    "", "1+", "(1+2", "1 2", "1//2", "1**2", "-1+2", "+1+2",
    "1+-2", "0+1+2", "01+2", "1.0+2", "True+2", "１+2",
    "answer: 1+2", "<answer>1+2</answer>", "```1+2```",
    "1+2; print(1)", "__import__('os').system('echo unsafe')",
    "abs(1)+2", "[1,2]", "1e0+2", "1_0+2", "(1)(2)",
])
def test_rejects_malformed_and_code_like_payloads(expression):
    result = verify_expression(Problem([1, 2], 3), expression)
    assert not result.valid
    assert result.reward == 0.0
    assert result.value is None


@pytest.mark.parametrize("expression", [
    " " * 4096 + "1+2",
    "(" * 33 + "1+2" + ")" * 33,
    "+".join(["1"] * 129),
    "1000001+2",
    "*".join(["1000000"] * 20),
    "/".join(["1000000"] * 20),
])
def test_resource_limits_return_zero(expression):
    result = verify_expression(Problem([1, 2], 3), expression)
    assert not result.valid
    assert result.reward == 0.0
    assert result.reason == "resource_limit"


@pytest.mark.parametrize("numbers,target", [
    ([], 1), ([True], 1), ([0], 1), ([-1], 1), ([1.0], 1),
    ([1000001], 1), ([1] * 9, 1), ([1], True), ([1], 1.5),
    ([1], 1000000001), ("123", 1), (None, 1),
])
def test_problem_api_rejects_invalid_inputs(numbers, target):
    with pytest.raises((ValueError, TypeError)):
        Problem(numbers, target)


def test_problem_identity_is_canonical_and_defensively_copied():
    numbers = [3, 2, 2]
    problem = Problem(numbers, 7)
    numbers[0] = 99
    other = Problem([2, 3, 2], 7)
    assert problem.numbers == (2, 2, 3)
    assert problem.id == other.id
    assert problem.canonical_digest == other.canonical_digest
    expected = hashlib.sha256(b'{"numbers":[2,2,3],"target":7}').hexdigest()
    assert problem.canonical_digest == expected
    assert problem.id != Problem([2, 2, 3], 8).id
    with pytest.raises(FrozenInstanceError):
        problem.target = 8


def test_verifier_api_and_result_are_immutable():
    with pytest.raises(TypeError):
        verify_expression(None, "1+2")
    with pytest.raises(TypeError):
        verify_expression(Problem([1, 2], 3), None)
    result = verify_expression(Problem([1, 2], 3), "1+2")
    with pytest.raises(FrozenInstanceError):
        result.reward = 0


def test_contract_digest_binds_extraction_and_version():
    assert VERIFIER_VERSION == "number-expression-v1"
    assert ANSWER_EXTRACTION_RULE == "trim-outer-whitespace-then-parse-entire-expression"
    assert len(VERIFIER_DIGEST) == 64
    assert int(VERIFIER_DIGEST, 16) >= 0
