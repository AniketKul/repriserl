# RepriseRL M0/M1 Implementation Plan

> Implement one bounded task at a time, with independent numerical references and focused verification. Checkboxes track work that has actually completed.

**Goal:** Establish a reproducible one-H100 baseline and a loss-correct fixed-replay reference before building the persistent C++ service.

**Architecture:** Reuse PyTorch/TRL for the learner and start with a local reference experience backend. One GPU-owning driver alternates generation, scoring, and training; CPU verification and reporting remain separate. M2-M5 in the full proposal follow only after these numerical and learning gates.

**Tech Stack:** Python, PyTorch, a tested/pinned TRL stack, deterministic CPU verification, local JSON/Parquet evidence; C++20/gRPC/RocksDB and ClickHouse are later deliverables, not M0 prerequisites.

**Spec:** [Implementation proposal](../RepriseRL_Implementation_Proposal.md). Current status: [Kickoff](../KICKOFF.md).

**Status (2026-10-14):** Tasks 1 and 2 are implemented and reviewed: configuration, hardware reporting, bounded exact-arithmetic verification and canonical task splits. The 147-test CPU suite and distribution builds passed on macOS and Linux; see the kickoff record for source-bound evidence. Tasks 3–7 remain planned. Trainer-dependent work requires a tested dependency lock and explicit algorithm conventions. No training or learning benchmark has run.

## Global Constraints

The following requirements are copied verbatim from the original proposal:

- One model lineage, one learner, one GPU, text-only single-turn verifiable tasks.
- Complete prompt groups with immutable generation-time probabilities and advantages.
- Fresh-only, fixed-replay, and adaptive-replay policies under the same objective.
- Do not advertise exactly-once gradient application.
- The learner must never query ClickHouse for a training decision.
- No log-probability threshold certifies learning safety.

Additional execution boundaries: use one authorized H100 and recheck allocation before use; preserve existing files and instructions; bound all runs by steps and time; verify dependency claims; publish only explicitly authorized, reviewed artifacts. Do not provision paid resources implicitly. CPU tests remain useful without GPU access.

## Review Focus

1. No GPU, a MIG slice, or a busy GPU: report actual availability and keep CPU work functional; do not use or terminate another workload. Owned by Task 1.
2. Adversarial or malformed expression text and equivalent train/test problems: reject safely and maintain split separation. Owned by Task 2.
3. All-equal rewards, empty masks, unequal sequence lengths, and non-finite probability fields: produce defined errors or correct loss/gradient behavior, never silent NaNs. Owned by Task 4.
4. Retries, invalid policy lineage, mutation through array aliases, and a fully ineligible buffer: preserve history and use counts; fall back to fresh generation rather than stall. Owned by Task 5.
5. Unavailable vLLM integration, incomplete measurement windows, and unsupported trainer updates: fail explicitly or use a declared comparable reference mode; never label skipped work as passed. Owned by Tasks 3, 6, and 7.

## Repository discovery before implementation

- [x] Inspect repository scope and guidance: this initial revision contains documentation and no product code, tests, build system, or dependency lock. Preserve unrelated changes.
- [x] Complete an initial read-only hardware inspection. Keep host identifiers and raw inventory outside the repository; recheck allocation before GPU execution.
- [ ] Read the pinned candidate trainer implementation and verify the proposal's probability-field assumptions. Choose and document exact std, action-mask, truncation, and loss-normalization conventions matching the compared baseline.
- [x] Review this repository-adjusted task/file map and retain the original numerical and learning gates. No learning algorithm change is introduced by publication.
- [ ] Before Tasks 3 and 6, complete a framework/API compatibility review and choose explicit baseline-matched conventions.

## Dependency preparation

The [TRL v1.15.0 declaration](https://github.com/huggingface/trl/blob/v1.15.0/pyproject.toml), checked on 2026-10-14, requires Python >=3.11, Accelerate >=1.4.0, Datasets >=4.7.0, and Transformers >=4.56.2. These minimum requirements are not a tested environment lock. Create an isolated Python >=3.11 environment, inspect the pinned trainer implementation, and resolve/test the complete stack before using it. Do not change system Python. The selected vLLM integration remains a separate validation gate.

For Tasks 1 and 2, keep configuration validation and exact-arithmetic verification independent of TRL and GPU initialization. Introduce packaging and CPU test dependencies in Task 1. Do not install dependencies or allocate a GPU just to publish these documents.

## File responsibilities

- `python/repriserl/config.py`: validated algorithm/training configuration, with no hidden library defaults.
- `python/repriserl/hardware.py`: read-only hardware inventory and proposed memory-envelope checks.
- `python/repriserl/verifiers/number_expression.py`: restricted parser and exact arithmetic verifier.
- `python/repriserl/data/problems.py`: deterministic, canonical, non-overlapping task splits.
- `python/repriserl/reference/types.py`: immutable rollout/group/batch identities and validation.
- `python/repriserl/reference/loss.py`: independent tensor loss and group-advantage reference.
- `python/repriserl/reference/buffer.py`: fixed-replay reference with reservations and accounting.
- `python/repriserl/adapters/trl/compat.py`: version-specific prepared-batch/probability handling.
- `python/repriserl/adapters/trl/driver.py`: one-GPU phase coordination and bounded execution.
- `python/repriserl/metrics.py`: monotonic phase timings and deduplicated run evidence.
- `benchmarks/learning/run.py`: baseline/fixed-replay entry point with explicit budgets.
- `benchmarks/learning/report.py`: fixed-budget and target-based comparisons without hiding failures.

Use existing equivalents if a repository is already present. Keep all version-specific overrides in `compat.py` rather than copying an entire trainer.

---

### Task 1: Validated configuration and honest hardware report

**Files:** Create the config/hardware modules above; `tests/unit/test_config.py`; `tests/unit/test_hardware.py`; `configs/qwen15b_h100_fresh.yaml`. Integrate packaging into this task, following existing manifests.

**Interfaces:** Produces `RunConfig`, `HardwareReport`, `load_config(path: Path) -> RunConfig`, and `inspect_hardware() -> HardwareReport`. The report contains observed devices/allocation/host resources and explicit unknowns; it does not grant permission to use the devices.

- [x] Write failing tests: `test_single_gpu_contract` rejects requested GPU counts other than one; `test_default_profile` checks 512/1024 token caps, group=4, batch_groups=8, microbatch=1, and beta=0; `test_no_cuda_is_reported` expects a usable CPU report; `test_mig_and_busy_device_are_not_full_free_h100` exercises mocked observations.
- [x] Run `python -m pytest tests/unit/test_config.py tests/unit/test_hardware.py -q`; confirm the failures are missing implementation, not broken test collection.
- [x] Implement the interfaces with explicit validation. Keep LR=1e-6 a declared candidate. Record a mandatory per-run maximum-step or maximum-time budget; do not default to an unlimited run.
- [x] Repeat the tests and save an actual read-only hardware report when available. An 80 GB assumption must never become an observed value without a measurement.
- [x] Review the diff and create a focused commit when repository policy permits it.

### Task 2: Deterministic verifier and separated task splits

**Files:** Create the verifier/data modules above; `tests/unit/test_number_expression.py`; `tests/unit/test_problem_splits.py`.

**Interfaces:** Produces `Problem`, `VerificationResult`, `verify_expression(problem: Problem, completion: str) -> VerificationResult`, and `make_splits(seed: int, sizes: tuple[int, int, int]) -> ProblemSplits`. `ProblemSplits` exposes train/development/test IDs and canonical digests.

- [x] Write failing tests for correct exact-arithmetic solutions, repeated-number accounting, missing/extra numbers, divide-by-zero, malformed syntax, overly large expressions, code-like payload rejection, and canonical split disjointness. Use no `eval` or `exec`.
- [x] Run `python -m pytest tests/unit/test_number_expression.py tests/unit/test_problem_splits.py -q` and verify expected failures.
- [x] Implement a restricted grammar with bounded parsing and rational arithmetic. Preserve a canonical multiset-and-target problem identity. Freeze the answer-extraction rule and verifier digest.
- [x] Repeat tests; generate a small deterministic fixture twice and verify identical IDs, rewards, and hashes. Full 20,000/512/2,048 datasets are proposed defaults, not prerequisites for unit tests.
- [x] Review and commit. Do not tune on sealed-test outputs.

### Task 3: Reproducible stock baseline and timing envelope

**Files:** Create `adapters/trl/driver.py`, `metrics.py`, `benchmarks/learning/run.py`; `tests/unit/test_phase_meter.py`; `tests/integration/test_baseline_smoke.py`; and `docs/environment.md`.

**Interfaces:** Consumes `RunConfig` and versioned problems. Produces `run_baseline(config: RunConfig, output_dir: Path) -> RunSummary` and `PhaseMeter`, which records non-overlapping GPU phase spans and a separate inclusive wall-clock span. `RunSummary` records completed/failed/skipped status and file paths, not fabricated metrics.

- [ ] Write failing tests for finite bounded runs, non-overlapping phase records, no silent missing metrics, and a CPU tiny-model baseline that terminates at its explicit update budget.
- [ ] Run unit/tiny tests first and observe the expected failures. Separate opt-in H100 integration markers so ordinary CI cannot silently initiate training.
- [ ] Implement a standard, unmodified baseline in a verified dependency environment. Start with one generation/training backend for parity; record any vLLM mode separately. Retain `num_iterations` choices 1/2/4 as baseline configurations.
- [ ] Repeat CPU tests. After authorization and confirmed access, run one budgeted H100 smoke/pilot, recording exclusive peak device memory, phase times, total wall time, and learning signals. If inaccessible, provide the command and label it NOT RUN.
- [ ] Review evidence and commit. M0 exits only with actual learning evidence and resource headroom; tiny random-model tests alone do not satisfy that gate.

### Task 4: Immutable rollout schema and independent loss reference

**Files:** Create `reference/types.py`, `reference/loss.py`; `tests/numerical/test_advantages.py`; `tests/numerical/test_loss.py`; `tests/unit/test_rollout_schema.py`.

**Interfaces:** Produces `Rollout`, `RolloutGroup`, `ReplayBatch`, `compute_advantages(rewards: Tensor, std_correction: int, epsilon: float = 1e-4) -> Tensor`, and `replay_loss(current_logp: Tensor, batch: ReplayBatch) -> Tensor`. Tensor shape and dtype contracts are explicit, and the batch stores both immutable old probability arrays.

- [ ] Write failing numeric fixtures matching the spec's c=min(2, exp(old_train-old_sample)), detached c, r=exp(current-old_train), clipped ratio [0.8,1.2], and whole-batch token normalization. Test all-equal rewards, one valid token, heterogeneous lengths, empty-mask rejection, non-finite values, and preservation against caller-array mutation.
- [ ] Run `python -m pytest tests/numerical tests/unit/test_rollout_schema.py -q` and confirm expected failures.
- [ ] Implement the reference without a trainer dependency. Require the std convention to be explicitly configured after baseline inspection. Protect valid-token calculations from padding NaNs; reject absent/non-finite valid-token probability fields.
- [ ] Repeat tests, compare finite-difference/autograd gradients on small float64 fixtures, and verify accumulated microbatches match the declared full optimizer batch.
- [ ] Review and commit. Treat corrections as an empirical surrogate, not proof of unbiased learning.

### Task 5: Uniform bounded replay reference

**Files:** Create `reference/buffer.py`; `tests/unit/test_reference_buffer.py`; `tests/property/test_buffer_histories.py`.

**Interfaces:** Produces `ReferenceReplayStore.put_group(group: RolloutGroup) -> PutResult`, `reserve(request: ReserveRequest) -> Reservation`, `record_applied(batch_id: str, epoch: int) -> None`, and `release(reservation_id: str, epoch: int) -> None`. Request/result dataclasses live in `reference/types.py`; reservations refer to whole-group immutable IDs.

- [ ] Write failing tests for idempotent identical puts, conflicting payload rejection, full-group selection, no duplicates in one batch, pending uses counted against the three-use cap, eight-update age cap, declared strata, fixed-seed reproducibility, stale epochs, and no eligible candidates.
- [ ] Run `python -m pytest tests/unit/test_reference_buffer.py tests/property/test_buffer_histories.py -q` and confirm expected failures.
- [ ] Implement in-memory bounded sampling and explicit reservation accounting; no persistence or ClickHouse dependency yet. Return unmet quotas for fresh generation rather than spinning or manufacturing data. Exact policy-age boundary semantics must be declared and tested.
- [ ] Repeat tests, including randomized histories and external-array mutation attempts. Retrying a reservation/batch identity must return the same selection and must not double-count use.
- [ ] Review and commit. This implementation becomes the service's behavioral oracle, not the final persistence backend.

### Task 6: Narrow TRL replay adapter and single-GPU fixed-replay path

**Files:** Create `adapters/trl/compat.py`; extend `driver.py`; add `tests/integration/test_fresh_parity.py`, `test_fixed_replay.py`, and `test_probability_provenance.py`.

**Interfaces:** Consumes the reference store, batch schema, loss reference, and baseline driver. Produces `ReplayGRPOAdapter.prepare_batch(groups: Sequence[RolloutGroup], current_policy: PolicyIdentity) -> ReplayBatch` and `run_fixed_replay(config: RunConfig, output_dir: Path) -> RunSummary`. Define `PolicyIdentity` in the shared types module.

- [ ] Write failing tests comparing fresh-only loss and gradients on identical supplied tokens; test that old denominators are unchanged after several optimizer updates and backend correction appears exactly once. Cover wrong reward/tokenizer/branch identity, unsupported trainer versions, and truncated completions.
- [ ] Run the integration tests against the independently validated same-backend reference and observe expected failures.
- [ ] Implement only the necessary prepared-batch integration, preserving original fields. Do not return old text to a generic rollout callback and assume correctness. Candidate rescoring is a derived no-update pass, never a behavior-field rewrite.
- [ ] Repeat numerical and tiny-model parity tests. Validate the declared phase order and prohibit simultaneous GPU phases. Add colocated vLLM only after actual probability/sleep/weight-sync compatibility is confirmed; unsupported modes fail clearly.
- [ ] After run-budget approval and GPU access, compare the baseline and fixed replay on a controlled short learning run, then record costs and outcomes without extrapolating a full research result. Review and commit.

### Task 7: Evidence report and M1 exit decision

**Files:** Create `benchmarks/learning/report.py`; `tests/unit/test_benchmark_report.py`; `reports/README.md`; update `docs/KICKOFF.md`.

**Interfaces:** Consumes `RunSummary` plus stable event records. Produces `compare_runs(run_dirs: Sequence[Path], protocol: EvaluationProtocol) -> ComparisonReport`. Define protocol/report types locally with declared target, evaluation cadence, seed identities, and completion status.

- [ ] Write failing fixtures for duplicate event IDs, missing end times, failed-to-target runs, unequal budgets, test/development split mixups, multiple seeds, and negative net savings. An incomplete run must never appear as a successful faster run.
- [ ] Run `python -m pytest tests/unit/test_benchmark_report.py -q` and confirm expected failures.
- [ ] Implement deduplicated reporting of inclusive H100 wall time, phase breakdowns, quality curves, completed/failed/skipped runs, and uncertainty limitations. Do not invent standard errors from one seed or treat completions as independent training runs.
- [ ] Repeat tests and produce a report only from existing run evidence. For synthetic report fixtures, visibly label the report as synthetic.
- [ ] Review M0/M1 exit gates, update progress and blockers, and commit. If fixed replay is harmful or costs more, investigate before M2. Do not build the daemon to conceal an unsuccessful algorithm.

## Deferred phases retained in the full proposal

M2 implements the C++20 daemon, gRPC contract, RocksDB/local payload durability, and reference-versus-service operation-history parity. M3 adds joint recovery, fencing, and robust GPU phase transitions. M4 implements the adaptive controller and optional ClickHouse with explicit deduplication. M5 runs the approved multi-seed/second-task evaluation and prepares reproducible public artifacts.

Each phase should receive its own repository-specific plan and acceptance review. The existing proposal preserves their full contracts; this document deliberately does not prematurely prescribe hundreds of lines of uninspected service code.

## Self-review of this planning artifact

- Scope covers M0 and M1 only; M2-M5 remain explicitly retained, not silently omitted.
- Every proposed module has an owner task; common types are defined before dependent interfaces.
- All five review-focus classes have assigned tests.
- Numerical correctness and baseline parity precede persistent/distributed infrastructure.
- Product tests and learning benchmarks remain unrun. Dependency declaration checks and hardware discovery are recorded separately from runtime compatibility and training results.
- Hardware assumptions and later research targets are not promoted to measured facts.
