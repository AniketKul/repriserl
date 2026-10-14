# RepriseRL

**Reuse experience. Account for every GPU-second.**

RepriseRL is a proposed policy-aware experience replay backend for LLM post-training. It aims to reuse eligible rollout groups while preserving their generating-policy identity, original probabilities, rewards, and consumption history.

**Current stage: M0 CPU foundations implemented.** The installable Python package includes validated configuration, read-only hardware reporting, a bounded exact-arithmetic verifier, and deterministic task splits. Trainer integration, replay, training results, and measured speedup remain future work.

## Objective

Test whether bounded replay reaches a declared model-quality target with less total H100 time than tuned existing GRPO reuse and fixed replay. Include generation, rescoring, rejected candidates, transfers, waits, training, checkpoints, and evaluation in the comparison. A 20% reduction is an experimental acceptance target, not a result.

## Initial scope

- One full H100, one learner, one compatible model lineage.
- Qwen2.5-1.5B-Instruct with a deterministic number-expression verifier; tiny models for numerical tests.
- Python/PyTorch/TRL for the learner adapter and reference backend.
- C++20/gRPC/RocksDB for the subsequent persistent replay service, `reprised`.
- Optional asynchronous ClickHouse analytics.
- Complete groups of four completions, including failures; immutable generation-time probabilities and advantages.
- Sequential GPU generation, candidate rescoring, and optimization, with asynchronous CPU services.

## Documentation

1. [Current status and next steps](docs/KICKOFF.md)
2. [Complete implementation proposal](docs/RepriseRL_Implementation_Proposal.md)
3. [M0/M1 implementation plan](docs/plans/2026-10-14-repriserl-m0-m1.md)
4. [Implementation brief](docs/IMPLEMENTATION_BRIEF.md)
5. [Engineering guidance](AGENTS.md)

## Milestones

| Phase | Deliverable | Status |
|---|---|---|
| M0 | Reproducible stock baseline, verifier, dataset splits, and resource measurements | CPU foundations complete; trainer and learning evidence pending |
| M1 | Numerically validated reference replay and measured fixed-replay comparison | Planned |
| M2 | Persistent C++ service and protocol parity | Planned |
| M3 | Joint learner/replay recovery and GPU phase control | Planned |
| M4 | Adaptive replay policy and analytics | Planned |
| M5 | Multi-seed evidence and reproducible release artifacts | Planned |

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

Retain numerical parity, immutable behavior probabilities, complete-group accounting, and explicit time/step budgets throughout implementation. CPU checks, GPU smoke runs, actual learning curves, and synthetic service tests must be reported separately.

## License

[Apache License 2.0](LICENSE). Third-party dependencies, models, datasets, and imported material retain their own licenses.
