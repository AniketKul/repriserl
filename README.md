# RepriseRL

**Reuse experience. Account for every GPU-second.**

RepriseRL is a proposed policy-aware experience replay backend for LLM post-training. It aims to reuse eligible rollout groups while preserving their generating-policy identity, original probabilities, rewards, and consumption history.

**Current stage: design and implementation planning.** This repository has no product code, installable package, training results, or measured speedup yet. The first implementation milestone is the reproducible baseline and loss-correct reference replay.

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
| M0 | Reproducible stock baseline, verifier, dataset splits, and resource measurements | Planning; initial hardware discovery complete |
| M1 | Numerically validated reference replay and measured fixed-replay comparison | Planned |
| M2 | Persistent C++ service and protocol parity | Planned |
| M3 | Joint learner/replay recovery and GPU phase control | Planned |
| M4 | Adaptive replay policy and analytics | Planned |
| M5 | Multi-seed evidence and reproducible release artifacts | Planned |

## Development

Start with M0/M1 Tasks 1 and 2: configuration and hardware reporting, then the bounded arithmetic verifier and deterministic task splits. Trainer integration follows a tested environment lock and loss-contract review. There are no build, installation, or training commands to run from this revision.

Retain numerical parity, immutable behavior probabilities, complete-group accounting, and explicit time/step budgets throughout implementation. CPU checks, GPU smoke runs, actual learning curves, and synthetic service tests must be reported separately.

## License

[Apache License 2.0](LICENSE). Third-party dependencies, models, datasets, and imported material retain their own licenses.
