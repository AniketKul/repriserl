# RepriseRL implementation brief

Read the [current status](KICKOFF.md), [engineering guidance](../AGENTS.md), [full proposal](RepriseRL_Implementation_Proposal.md), and [M0/M1 plan](plans/2026-10-14-repriserl-m0-m1.md) before changing the implementation.

## Accepted direction

Build policy-aware experience replay for LLM post-training. The research objective is less total measured H100 time to a declared capability target, with held-out quality preserved under a prespecified criterion. Begin with a reproducible baseline and a loss-correct reference backend; then implement the C++ persistence service.

The target is one full H100. CPU services may overlap, while GPU generation, candidate rescoring, and optimization remain mutually exclusive. Python/PyTorch/TRL handle training and numerical references; C++20/gRPC/RocksDB provide the durable service; ClickHouse is optional analytics.

## Initial algorithm contract

- Four completions per original prompt group; eight complete groups per optimizer batch.
- Preserve original sampling and training-backend log-probabilities, original advantages, policy lineage, masks, and verifier identity.
- Compute current-policy ratios against the original training probabilities; apply the detached, capped backend correction exactly once.
- Normalize over all valid action tokens in the optimizer batch, including across microbatches of unequal lengths.
- Start fixed replay with eight fresh warmup updates, 50% fresh groups, maximum policy age eight updates, and maximum three lifetime uses including the first use.
- Keep failures and zero-advantage groups visible in the diagnostics and cost accounting.
- Joint learner/replay checkpoints preserve consistent lineage and consumption accounting; they do not promise exactly-once gradient application.

The full proposal defines the equations and experimental defaults. Explicitly select baseline-compatible standard-deviation and EOS/truncation conventions before trainer integration. Replay remains an empirical off-policy surrogate.

## Evidence required

Compare tuned existing GRPO reuse (`num_iterations` 1/2/4), fixed replay, and adaptive replay with comparable tuning budgets. Keep the test set sealed. Count all measured end-to-end costs, including unsuccessful candidates and recovery/checkpoint overhead.

Do not infer a learning benefit from CPU numerical tests, tiny random-model runs, or synthetic service benchmarks. The proposed 20% improvement and test non-inferiority margin are acceptance targets requiring experiments.

## Execution order

Begin with the configuration/verifier/data foundations. Validate an isolated trainer environment, then implement the stock baseline, independent loss reference, bounded replay buffer, narrow TRL integration, and deduplicated reports. Establish numerical parity and measure fixed replay before investing in the service and adaptive controller.

Keep training runs bounded, preserve existing work, and publish only authorized public-safe artifacts. Report unresolved implementation choices and failed or skipped checks explicitly.
