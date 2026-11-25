# RepriseRL

RepriseRL is a Python package for experiments with experience replay in language model training.

RepriseRL trains a stock GRPO baseline or reuses complete groups of recent
answers with fixed replay. The replay trainer preserves both generation-time
probability arrays, the original rewards, and within-group advantages. After
eight fresh updates it can fill up to half of each batch with eligible groups.
It rescores candidates before reuse and counts pending and committed uses.

## Run an offline training experiment

From the repository root, create a Python 3.12 environment and install the
training extra. The tiny model and arithmetic data are generated locally.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'
repriserl-train baseline --tiny --config configs/tiny_cpu.yaml --output outputs/stock --eval-every 9
repriserl-train fixed-replay --tiny --config configs/tiny_cpu.yaml --output outputs/replay --eval-every 9
repriserl-report --target 0.5 outputs/stock outputs/replay
```

Each run writes a `summary.json` with elapsed time, configuration, and
development evaluations. Give each invocation a new output directory.
The CPU fixture exercises generation, verification, optimization, and replay.
CPU scoring uses dense PyTorch operations. GPU scoring uses TRL's fused output head.

## Run with Qwen on one H100

Use a Linux Python 3.12 environment. Install the CUDA wheel from the PyTorch
index first, then install the recorded dependencies and the package.

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu126 'torch==2.9.1'
python -m pip install -r config/requirements-linux-cu126.lock
python -m pip install -e . --no-deps
repriserl-train baseline --config configs/qwen15b_h100_fresh.yaml --output outputs/qwen-stock --sizes 512 16 64 --eval-every 10
repriserl-train fixed-replay --config configs/qwen15b_h100_fresh.yaml --output outputs/qwen-replay --sizes 512 16 64 --eval-every 10
```

The supplied profile pins the model revision and bounds the run at 100 updates
or one hour. Stock GRPO supports `--num-iterations 1`, `2`, or `4`. The report
uses development accuracy and includes failed runs and runs that miss the
requested target. GPU runs sample other compute-process occupancy once per second.
The report compares only matched runs with uncontended observations throughout.
Short-lived activity between samples can go undetected.
The macOS ARM64 dependency snapshot is in `config/requirements-macos-arm64.lock`.

## Development

Use Python 3.11 or 3.12 in an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r config/requirements-dev.lock
python -m pip install --no-build-isolation -e .
python -m pytest -q
python -m build --no-isolation
```

The lock pins the CPU development dependencies. It is not a trainer-stack lock.
CI tests both supported development Python versions and checks the built wheel
outside the source checkout.

```python
from pathlib import Path
from repriserl.config import load_config
from repriserl.data.problems import make_splits
from repriserl.verifiers.number_expression import Problem, verify_expression

config = load_config(Path("configs/qwen15b_h100_fresh.yaml"))
result = verify_expression(Problem((1, 2, 3, 4), 24), "1 * 2 * 3 * 4")
assert result.reward == 1.0
splits = make_splits(seed=7, sizes=(8, 2, 2))
print(splits.canonical_digest)
```

The verifier accepts the entire expression, with each supplied number used
exactly once. Prose, code, unary operators, malformed or oversized expressions
receive zero reward. Arithmetic uses exact rational values. Verifier and generator
contracts expose versions and SHA-256 digests for later rollout provenance.

The initial generator draws four numbers from 1–20 and retains solvable integer
targets from 1–1000. This distribution has not been calibrated for difficulty.
Splits are deterministic and disjoint by canonical number multiset and target;
changing split sizes changes boundaries, so freeze a split artifact before use.

Use `repriserl.hardware.inspect_hardware()` for a read-only observation of
device availability immediately before running on the GPU.

## License

[Apache License 2.0](LICENSE). Third-party dependencies, models, datasets, and imported material retain their own licenses.
