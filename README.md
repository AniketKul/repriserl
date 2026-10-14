# RepriseRL

RepriseRL is a Python package for experiments with experience replay in language model training.

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

`repriserl.hardware.inspect_hardware()` reports observed availability without
allocating a GPU or importing a training framework. Availability is an observation,
not a reservation. The example configuration has a 100-step development limit;
model revision and numerical conventions must be pinned before training. No
training entry point is implemented yet.

## License

[Apache License 2.0](LICENSE). Third-party dependencies, models, datasets, and imported material retain their own licenses.
