"""Boundary tests: relaxed validation or profile drift must fail these tests."""
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from repriserl.config import RunConfig, load_config


def load(tmp_path, text):
    path = tmp_path / "run.yaml"
    path.write_text(text)
    return load_config(path)


def test_default_profile():
    config = load_config(Path("configs/qwen15b_h100_fresh.yaml"))
    assert (config.prompt_token_cap, config.completion_token_cap) == (512, 1024)
    assert (config.group_size, config.batch_groups, config.microbatch_size) == (4, 8, 1)
    assert config.beta == 0
    assert config.learning_rate == 1e-6
    assert config.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.max_steps == 100
    assert config.std_convention is None
    assert config.eos_handling is None
    assert config.truncation_handling is None
    with pytest.raises(FrozenInstanceError):
        config.group_size = 2


@pytest.mark.parametrize("count", [0, 2, -1, True, 1.0, "1"])
def test_single_gpu_contract(count):
    with pytest.raises(ValueError, match="requested_gpu_count"):
        RunConfig(max_steps=1, requested_gpu_count=count)


@pytest.mark.parametrize("text", [
    "{}", "max_steps: 0", "max_steps: true", "max_steps: 1.5",
    "max_steps: 1\nlearning_rate: .nan", "max_steps: 1\nbeta: .inf",
    "max_steps: 1\ngroup_size: 0", "max_steps: 1\nprompt_token_cap: -1",
    "max_steps: 1\ncompletion_token_cap: false", "max_steps: 1\nbatch_groups: 0",
    "max_steps: 1\nmicrobatch_size: 3", "max_steps: 1\nlearning_rate: 0",
    "max_steps: 1\nbeta: -0.1", "max_time_seconds: -1",
    "max_steps: 1\nunknown: 1", "max_steps: 1\nmax_steps: 2",
    "[]", "null", "!!python/object/apply:os.system ['false']",
    "max_steps: 1\nmodel: ''", "max_steps: 1\nseed: true",
    "max_steps: 1\nstd_convention: invented", "max_steps: 1\nmodel_revision: main",
])
def test_invalid_inputs_are_rejected(tmp_path, text):
    with pytest.raises(ValueError):
        load(tmp_path, text)


def test_time_budget_and_explicit_choices(tmp_path):
    config = load(tmp_path, "max_time_seconds: 60\nstd_convention: population")
    assert config.max_steps is None
    assert config.max_time_seconds == 60
    assert config.std_convention == "population"


def test_direct_constructor_validates_bounds():
    with pytest.raises(ValueError):
        RunConfig(max_steps=1, learning_rate=float("nan"))


@pytest.mark.parametrize("kwargs", [
    {"learning_rate": 10 ** 1000}, {"max_time_seconds": True},
    {"max_time_seconds": float("inf")}, {"seed": -1},
    {"std_convention": []}, {"eos_handling": False},
    {"truncation_handling": "invented"}, {"normalization": "per_microbatch"},
])
def test_direct_constructor_rejects_wrong_types_and_nonfinite_values(kwargs):
    with pytest.raises(ValueError):
        RunConfig(max_steps=1, **kwargs)


def test_yaml_merge_duplicates_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load(tmp_path, "max_steps: 1\n<<: {max_steps: 2}")


def test_immutable_model_revision_is_preserved(tmp_path):
    config = load(tmp_path, "max_steps: 1\nmodel_revision: '1234567890abcdef1234567890abcdef12345678'")
    assert config.model_revision == "1234567890abcdef1234567890abcdef12345678"
