# RepriseRL
## Policy-aware experience replay for LLM post-training

**Tagline:** Reuse experience. Account for every GPU-second.

**Status:** Implementation proposal, not implemented or benchmarked.
**Design date:** October 13, 2026. Public edition updated October 14, 2026.
**Hardware contract:** One full NVIDIA H100; assume 80 GB of device memory, not a smaller MIG allocation.
**Repository/package:** `AniketKul/repriserl` / `repriserl`; planned C++ daemon: `reprised`.
**License:** Apache-2.0; see the repository LICENSE file. Dependencies, models, datasets, and imported code retain their own terms.
**Naming:** Repository publication does not imply package-registry registration or trademark clearance.

---

## 1. Executive decision

Build an external replay service and a narrow training-framework adapter that make past LLM rollouts reusable without losing their generation-time policy identity, reward grouping, or consumption history. The first validated deployment is a single Linux host with one H100, asynchronous CPU services, and time-multiplexed GPU execution.

The research hypothesis is that bounded, policy-aware replay can lower total measured GPU time to a specified held-out development accuracy, with non-inferior performance on a sealed test set. The engineering hypothesis is that provenance, sampling, recovery, and diagnostics can be implemented as reusable infrastructure rather than a separate training-framework fork.

Neither hypothesis is a result. Experience replay itself is established, and replay for LLM post-training already has published precedents. Existing research documents both its promise and the bias/variance problems introduced by stale, repeatedly used data. Reverb already implements general-purpose C++/gRPC replay infrastructure. RepriseRL must earn its contribution through its LLM-specific contract, single-GPU usability, recovery semantics, and measured quality/compute trade-off. [S1, S2]

Do not begin with a large platform. First prove the loss semantics and a fixed-replay experiment in a reference backend; then put the validated contract behind the C++ service.

## 2. Scope and non-goals

### Required for the first supported release

- One model lineage, one learner, one GPU, text-only single-turn verifiable tasks.
- Complete prompt groups with immutable generation-time probabilities and advantages.
- Fresh-only, fixed-replay, and adaptive-replay policies under the same objective.
- A Python reference backend and a C++/gRPC persistent backend.
- Versioned sampling decisions, bounded queues, leases, process recovery, and joint learner/replay checkpoints.
- ClickHouse analytics and portable benchmark artifacts.
- Reproducible CPU correctness tests and opt-in H100 integration tests.

### Explicitly deferred

Multi-GPU actor/learner overlap, DDP/FSDP, tensor parallelism, distributed consensus, replicated storage, multi-tenant Internet hosting, cross-model replay, multimodal rollouts, tool agents, reward-model serving, priority replay driven by reward, and new CUDA kernels.

No prerequisite Kubernetes, Kafka, Redis, or PostgreSQL deployment is needed. A single metadata authority plus durable local storage is sufficient for the first release. Remote interfaces and injectable faults provide distributed-systems boundaries without making unsupported claims about multi-node scale.

## 3. Hardware and software profile

| Setting | Proposed initial choice |
|---|---|
| GPU | One full H100, nominally 80 GB |
| Main model | `Qwen/Qwen2.5-1.5B-Instruct`, exact Hub revision recorded |
| Smoke-test model | Qwen2.5-0.5B-Instruct or a tiny random model for numerical tests |
| Training | Full parameters, BF16 computation, numerically appropriate optimizer states |
| Optimizer | AdamW; initial LR candidate 1e-6, fixed after a bounded development sweep |
| Memory controls | Gradient checkpointing; bounded generation allocation; no retained GPU replay data |
| Prompt/completion cap | 512 / 1,024 tokens initially; optional 2,048-token total profile later |
| Group size | Four completions from the same prompt and generating policy |
| Optimizer batch | Eight complete groups, 32 sequences |
| Microbatch | One sequence; accumulation across the 32-sequence optimizer batch |
| Reward | Deterministic CPU verifier, no learned reward model |
| KL reference | beta = 0 for the first matched experiment; no separate reference model |
| Host target | 16 CPU cores, 64 GB RAM usable; 128 GB preferred |
| SSD target | 250-500 GB free; bounded payload and checkpoint retention |
| Memory acceptance target | Measured aggregate GPU peak below 70 GB, not just PyTorch allocator usage |

The model card reports 1.54B parameters. Conventional 18-byte mixed-precision Adam accounting gives approximately 27.7 GB for persistent training state before activations and generation resources. This is a planning estimate, not a measurement of the eventual implementation. Precision layouts and optimizer implementations change that accounting. [S3, S4]

Reduce sequence length and microbatch memory before changing the scientific question. LoRA is a fallback profile, not a silent replacement for full-parameter results; it requires its own matched baselines.

### Dependency policy

Evaluate the released TRL v1.15.0 tag as the initial compatibility candidate, rather than developing against an unpinned moving main branch. Record exact PyTorch, Transformers, Accelerate, vLLM, CUDA runtime, driver, C++ compiler, gRPC, and RocksDB versions in the successful environment lock and container digest. The v1.15.0 release is available and declares Python >=3.11. A complete compatible environment has not yet been validated for RepriseRL. [S5]

Use C++20/CMake for the replay daemon, Protobuf/gRPC for the protocol, RocksDB for local authoritative metadata, Python/PyTorch/TRL for training, a single ClickHouse instance for analytics, and Parquet/JSON for exports. No custom autograd implementation is needed.

## 4. Architecture

### Components

| Component | Owns | Does not own |
|---|---|---|
| `LearnerDriver` | Model, optimizer, GPU phase transitions, policy versions, training checkpoints | Replay metadata authority |
| `ReplayGRPOAdapter` | Batch preparation, frozen behavior fields, loss-contract integration | General training-framework replacement |
| `reprised` | Durable groups, eligibility index, reservations, use counts, batch identities, event outbox | GPU execution or gradient computation |
| Verifier workers | Versioned deterministic reward calculation, timeouts | Arbitrary execution of model-produced code |
| Controller | Fresh/replay fraction, bounded candidate selection, cost/drift feedback | Claims of certified safe off-policy reuse |
| Analytics exporter | Batched, retryable event delivery | Admission or optimizer-step commits |
| ClickHouse | Experiment history and analytical comparisons | Authoritative sampling decisions |

The first deployment uses independent CPU processes plus one GPU-owning Python process. The colocated generation backend is controlled by that learner process; there is not a second independently scheduled GPU server.

TRL documents sequential generation/training in standard GRPO, colocated vLLM sharing training devices, and separate GPUs for AsyncGRPOTrainer. Therefore the first integration targets standard GRPO rather than AsyncGRPOTrainer. [S6, S7]

### Single-GPU phase machine

`PLAN -> RESCORE_REPLAY -> GENERATE_FRESH -> VERIFY_AND_SCORE_FRESH -> TRAIN -> RECORD -> PLAN`

At an optimizer-step boundary:

1. Choose the fresh fraction and reserve complete replay candidate groups using CPU metadata.
2. With the inference engine asleep, teacher-force selected replay candidates through the current training model. Score only proposed candidates, never the whole buffer. No optimizer update occurs during this phase.
3. Reject ineligible candidates. Increase the fresh quota to preserve the intended eight-group batch.
4. Wake the colocated generation engine, synchronize the current weights, invalidate stale inference caches, and generate missing complete groups.
5. Sleep/release generation resources. CPU verifiers finish rewards, and the training backend records generation-time log-probabilities for fresh completions before any parameter update.
6. Assemble the batch, accumulate gradients, and perform one optimizer update.
7. Record the applied batch, use counts, metrics, and the new policy version. Periodically commit a joint checkpoint.

CPU verification, payload writes, metadata work, and analytics can overlap GPU work on independent batches. GPU generation, rescoring, and optimization remain mutually exclusive. Do not use MIG or MPS to pretend that one card is two independent training devices.

vLLM sleep mode can release inference allocations and has mechanisms intended for weight-update workflows. The implementation must verify wake/synchronize/cache invalidation behavior in its pinned version, and count the associated transfers and waits. [S8]

## 5. Replay's algorithm contract

### 5.1 Whole groups are the unit of reuse

A group contains one exact tokenized prompt and G=4 completions generated under one policy/sampler version. Compute and store its rewards and group advantages once:

\[
A_i = \frac{R_i - \overline{R}_g}{\operatorname{std}(R_g)+10^{-4}}.
\]

The standard-deviation convention is explicit in the algorithm manifest and identical in all arms. Never form a new comparison group by combining unrelated old completions. Never recompute an old advantage using today's batch reward statistics.

All-equal-reward groups have zero policy-gradient signal under this objective. Retain their diagnostics and count the generation work. Treat them consistently across arms; do not silently resample until a useful group appears. Truncation, EOS, and padding masks likewise follow one declared policy across all arms.

### 5.2 Keep two kinds of generation-time probability separate

For token t of completion i, store:

- `sampling_logp_b`: the probability reported for the token by the actual generation distribution.
- `training_logp_b`: the generation-time training-backend probability, with the same weights and token prefix, computed before updates.

For current parameters theta, use:

\[
r_{i,t}(\theta) = \exp(\ell_{\theta,i,t}-\ell^{train}_{b,i,t}),
\]

and a detached, frozen backend correction:

\[
c_{i,t}=\min(C,\exp(\ell^{train}_{b,i,t}-\ell^{sample}_{b,i,t})).
\]

Initial experimental C is 2. In the same-backend numerical reference, c is 1. If sampled probabilities are absent or invalid, the group is ineligible rather than being silently relabeled.

Use the same token-normalized clipped surrogate in every replay comparison:

\[
L(\theta) = -\frac{\sum_{i,t}m_{i,t}c_{i,t}\min(r_{i,t}A_i,\operatorname{clip}(r_{i,t},0.8,1.2)A_i)}{\sum_{i,t}m_{i,t}}.
\]

Here m excludes non-action and padding tokens and follows the declared truncation rule. Compute the normalization over the entire optimizer batch; microbatch gradient accumulation must not unintentionally reweight short sequences.

This separates policy aging from inference/training numerical mismatch. TRL has separate old-policy and sampling-probability paths; the adapter must deliberately provide both rather than rely on generic generation callbacks to infer them. [S6, S9]

This is a bounded off-policy surrogate, not an unbiased estimator of current-policy return. Token-level corrections, clipping, correlated replay, and content-dependent acceptance do not restore the full current trajectory/state distribution. Research on LLM replay explicitly discusses remaining bias even with importance corrections. No log-probability threshold certifies learning safety. [S1]

### 5.3 Sampling restrictions

Use temperature 1, no top-k/top-p truncation, no repetition penalty, and no generation-time dropout in the first profile. Explicitly set backend-specific equivalents and verify the returned probabilities. Record all settings even when they are defaults.

Keep a short replay horizon within the same model, tokenizer, reward, and task lineage. Cross-model and cross-tokenizer replay is unsupported. A policy version is an immutable branch/step/synchronization identity, not necessarily a full checkpoint retained for every update.

### 5.4 Start with fixed replay

The first candidate is uniform complete-group sampling with bounded age and reuse. Do not introduce success-only replay or learned priorities in the first experiment.

Proposed initial bounds, subject to development calibration:

| Parameter | Starting value |
|---|---:|
| Fresh warmup | Eight optimizer updates |
| Fixed fresh fraction | 0.5 |
| Maximum policy age | Eight optimizer updates |
| Maximum lifetime uses | Three, including the first use |
| Duplicate group in one optimizer batch | Prohibited |
| Candidate rescoring | Only proposed replay groups |

Every batch-selection record captures candidates, accepted groups, rejection reasons, sampling policy version, quotas, and RNG state/seed. Use complete groups and declared task-stratum quotas to reduce uncontrolled composition changes. Acceptance can still change the effective prompt distribution; expose that rather than calling the sampler unbiased.

### 5.5 Adaptive controller

After fixed replay works, add a transparent bounded controller. It chooses a fresh fraction from {1.0, 0.75, 0.5}; it does not require a second LLM.

At each control window of eight updates, examine observed policy-ratio drift, replay acceptance rate, diversity/coverage, and measured rescoring/service cost. Example initial rules:

- Start fresh-only. Reduce the fresh fraction by one level after two acceptable windows with positive estimated net generation-time savings.
- Treat a replay group as drift-ineligible when more than 30% of its valid action-token ratios lie outside [0.8, 1.2]. This is an experimental threshold, not a theorem.
- Increase the fresh fraction when rejection is high or the estimated rescoring/service cost exceeds avoided generation time.
- On non-finite diagnostics or broken provenance, reject the affected data and reset to fresh-only; do not continue corrupted training.
- Keep at least half the groups fresh and enforce age/use caps independently of the controller.

Compute avoided generation-time estimates from measured recent costs conditioned on length/task buckets. Estimates steer the controller; measured end-to-end results determine whether it helped.

A development sweep may change these thresholds. Freeze the chosen policy before final runs and give fixed replay an equal tuning budget. Compare cost-only and drift-aware variants as ablations. A learned utility predictor is a later experiment, not a dependency of the initial release.

## 6. Trainer integration and numerical gates

TRL's `rollout_func` is a useful experimental generation interface, but it is not a complete persistent replay API. The inspected implementation can recompute or omit old-policy probabilities based on generation/update scheduling. Returning old text through a hook therefore does not automatically preserve replay semantics. [S6, S9]

Use an external `ReplayGRPOAdapter` with a narrow, version-pinned batch-preparation extension. Its `ReplayBatch` contract contains prompt IDs, completion IDs, masks, original advantages, old training log-probabilities, actual sampling log-probabilities, frozen backend corrections, current policy metadata, and reservation identity.

Use existing model forward/backward and trainer utilities. Isolate any internal-method override in one compatibility module with tests. Do not silently fork or duplicate the whole trainer. After the interface is demonstrated, propose a small upstream prepared-batch/provenance extension rather than asking maintainers to host the daemon.

Mandatory tests before a research run:

- Fresh-only adapter loss and gradients match the pinned baseline on an identical supplied batch.
- A sampled token's generation-time denominator remains unchanged after several model updates.
- Frozen backend correction is applied exactly once.
- Group advantages, EOS/padding masks, and unequal-length accumulation match an independent small reference.
- Policy mismatch is distinguishable from a backend mismatch.
- Wrong tokenizer, reward version, branch, or tensor lengths produce explicit errors.

Fresh equivalence starts with one generation/training backend, then repeats with the chosen vLLM correction configuration. Calibrate floating-point tolerances by dtype/backend. Do not claim bitwise identity for arbitrary GPU generation.

## 7. Persistent data model

### Immutable group payload

`run_id`, `branch_id`, `group_id`, `task_id`, canonical prompt identity, exact prompt/completion token IDs, action masks, EOS/truncation status, reward values, frozen advantages, both generation-time log-probability arrays, policy version, sampler settings, tokenizer/template digest, reward/verifier digest, generation costs, schema version, and payload checksum.

### Mutable authoritative metadata

Use count, eligibility/retirement state, current lease, most recent assessment revision, compact drift diagnostics, sampling-policy version, and insertion/retirement sequence numbers.

Reassessment diagnostics are versioned derivatives. They cannot mutate the original behavior fields. Store payloads in RAM/SSD; no KV caches, activations, vocabulary-wide logits, or old weight copy per group is required.

Three four-byte arrays (token IDs plus two log-probability arrays) correspond to 12 bytes per recorded action token. Thus 100 million such tokens are approximately 1.2 GB before prompt storage, masks, metadata, indexes, replication, and checkpoints. This arithmetic illustrates payload scale only. Real allocation is measured by the profiler.

### Local storage layout

Use checksummed, length-prefixed Protobuf frames in append-only NVMe segments. Record locators and control metadata in RocksDB. A commit becomes acknowledged only after the payload is durable and the corresponding metadata/outbox transaction is synced. Recover incomplete segment tails and collect orphaned frames conservatively.

Keep write-ahead logging enabled. Serialize authoritative metadata mutations through one owner so a RocksDB batch can update the reservation, use bookkeeping, and outbox consistently. RocksDB provides the storage primitive, not a transaction spanning model memory and external files. [S10]

Budget payload storage by bytes and tokens, not only group count. Never evict data referenced by a live reservation or retained recovery checkpoint. Return explicit backpressure when capacity is exhausted.

## 8. Protocol and concurrency contract

These are proposed API names, not installed or implemented commands.

| Operation | Contract |
|---|---|
| `RegisterRun` | Fix algorithm/schema/model/reward identities and an epoch |
| `PutGroup` | Idempotent insert by group ID and payload digest; conflicting content fails |
| `ReserveCandidates` | Reserve whole groups against an eligibility snapshot and sampling policy |
| `RecordAssessment` | Attach diagnostics for a specified policy version; never overwrite behavior |
| `FinalizeBatch` | Persist an ordered set of accepted replay/fresh group IDs and the lease token |
| `RecordApplied` | Idempotently record that the live learner applied that batch; not by itself a recovery checkpoint |
| `PrepareCheckpoint` | Freeze/persist replay state through an applied-batch watermark |
| `RestoreEpoch` | Restore a joint checkpoint and issue a new fencing epoch |
| `StreamEvents` | Resume analytical export from a durable event sequence |

Leases reserve capacity and prevent accidental duplicate delivery. They are not the training-state commit mechanism. Retrying a batch ID returns the same immutable batch identity rather than drawing fresh samples. A late worker with an expired epoch cannot change use counts or checkpoint state.

Use bounded asynchronous gRPC operations with deadlines and cancellation. Count pending reservations against reuse caps. CPU verifier jobs are idempotent under group/verifier identity. A local daemon crash pauses the learner, instead of permitting untracked continuation.

## 9. Joint checkpoint and recovery semantics

Do not advertise exactly-once gradient application.

At a quiescent optimizer boundary, pause batch admission and finish the current update. Save model parameters, optimizer/scheduler state, random generators, data cursor, controller state, and ordered applied-batch history through step k. Ask `reprised` to prepare a matching durable replay snapshot.

Fsync both prepared artifacts. Publish one small manifest containing their identities/digests using atomic replace on the local filesystem, followed by directory synchronization. That manifest is the recovery commit point. Retain the previous valid manifest until the new one is verified.

After a crash, restore both sides from the latest complete manifest, issue a new branch/epoch, invalidate old leases, and roll back post-checkpoint consumption accounting. Quarantine rollouts produced on the abandoned post-checkpoint policy branch instead of treating them as ancestors of the restored policy. They may remain in analytical history, marked abandoned.

Updates since the last joint checkpoint can be recomputed. GPU nondeterminism can yield different future tokens, so the guarantee is consistent state lineage and accounting, not bit-for-bit identical learning.

Initial checkpoint interval: 32 completed optimizer updates, configurable after measuring checkpoint I/O. Keep the latest two recovery checkpoints plus deliberately selected evaluation artifacts. Include checkpoint time in primary end-to-end results.

Local fsync does not protect against disk or host destruction. Off-host backup and multi-node replication are separate later capabilities. Remote/object-storage publication needs a different commit protocol, not an assumed atomic filesystem rename.

## 10. ClickHouse design

Use a single CPU instance. It can be disabled for minimal development; an on-disk event outbox still preserves analytics. The learner must never query ClickHouse for a training decision.

| Logical table | Main fields / purpose |
|---|---|
| `runs` | Immutable configuration, code/container digests, hardware/software manifest |
| `rollout_events` | Group identity, generation policy, task stratum, cost, reward summary |
| `replay_decisions` | Candidate/accepted IDs, age, use count, drift summary, decision reason |
| `phase_metrics` | Exclusive GPU phases, wall durations, generated/trained tokens, memory |
| `evaluations` | Split identity, checkpoint, task-level correctness, length, truncation |
| `recovery_events` | Crash boundary, restored manifest, abandoned branch, recomputed work |

Batch exporter writes and retry with stable event IDs. A `ReplacingMergeTree` table can reduce duplicates eventually, but background merging does not guarantee immediate uniqueness. Final benchmark queries must deduplicate by event identity, e.g. FINAL on a correctly designed sorting key or explicit argMax/grouping, before aggregation. Do not sum duplicate-sensitive events through an incremental view and assume later merges will fix the sum. [S11]

Partition by bounded calendar windows and order by run plus stable event identity/sequence; do not create a partition per prompt or token. Keep sampled token diagnostics rather than inserting every token into an individual analytical row. Raw prompt text is off by default. Retain aggregate run results while expiring raw debug events under an explicit retention policy.

The dashboard should answer whether generation saved exceeds reassessment cost; where drift rejections occur; whether replay alters task coverage; whether reuse causes collapsed output diversity; and whether any claimed speedup comes from truncation or shorter answers.

A CPU-only load generator may create large synthetic event histories and multiple simulated clients. Label those results as systems tests, not observations from large real-model training runs.

## 11. Data, tasks, and evaluation

### First learning task

Use a procedural number-expression task with a deterministic grammar verifier. It should have a development-selected difficulty mix on which the starting model has both successes and failures. Verify exact arithmetic, allowed operators, and number usage; never execute arbitrary generated source via eval.

Generate separate canonical train/development/test problem sets; exclude equivalent number/target instances across splits. Starting sizes: 20,000 training problems, 512 development problems, and 2,048 sealed test problems. These sizes are experimental defaults, not a power analysis.

The controller uses training diagnostics, not the sealed test outcomes. Choose target accuracy, stopping budget, evaluation cadence, and hyperparameter sweep limits after a pilot and before confirmatory runs. Repeated development evaluation is reported as such.

### Second task

Before a broad learning-efficiency claim, repeat on a public reasoning task such as GSM8K using its documented split and a separately versioned answer verifier. Treat public-benchmark contamination as a limitation, not proof of general reasoning gains. Do not use official test answers to tune the controller. [S12]

### Baselines

| Arm | Purpose |
|---|---|
| B0: stock fresh data, one update per generation | Diagnostic baseline |
| B1: best tuned existing GRPO reuse, num_iterations in {1,2,4} | Strong non-buffer comparator |
| B2: uniform fixed replay | Isolate the benefit of adaptive control |
| B3: RepriseRL adaptive replay | Proposed method |

GRPO already exposes multiple updates per generated batch. Ignoring that feature would overstate this project's benefit. [S6]

Use identical precision, reward, masks, group size, optimizer-batch normalization, task distribution rules, backend correction, and initial checkpoint. Give B1/B2/B3 comparable tuning budgets. Count warmup, buffer population, teacher forcing, synchronization, and any unsuccessful candidates.

Run final configurations with at least three seeds, sequentially on the same H100. Add a prompt-matched fresh-regeneration control and a no-drift-controller ablation if the initial result looks positive. These help distinguish experience reuse from accidental curriculum effects.

### Metrics and claim threshold

Primary practical metric: exclusive H100 wall-seconds to a preregistered held-out development target, including generation, scoring, updates, transfers, service waits, checkpoints, and the declared evaluation schedule. Also report active GPU phase times, fixed-budget learning curves, final sealed-test accuracy, and a training-only breakdown. Never double-count overlapped CPU phases in the exclusive wall clock.

Select checkpoints using the declared development rule, then evaluate sealed test results. Runs that never reach target are censored/failed-to-target, not removed. Report seed variability and paired task-level confidence intervals separately rather than treating all completions across seeds as independent model runs.

Aspirational success criterion: at least 20% lower total measured time than B1, measurable improvement over B2, and a prespecified sealed-test non-inferiority margin (initial proposal: 2 percentage points, revisited through a power calculation before final runs). These are acceptance targets, not predicted results.

A cheap speedup that produces less accurate or prematurely truncated answers fails the main objective. If fixed replay does not show a useful quality/compute trade-off on the controlled task, investigate the algorithm before expanding the infrastructure.

## 12. Verification matrix

| Test class | Required evidence |
|---|---|
| Numerical CPU | Independent loss/gradient fixtures; frozen-denominator and backend-correction checks |
| Model integration | Fresh-only parity; finite gradients; masks and accumulation correct |
| Data integrity | Corrupt or conflicting groups rejected; original probability fields immutable |
| Sampling | Complete groups, bounded uses, deterministic reservation retries, no same-batch duplicates |
| Recovery | Crashes before/after payload sync, metadata sync, optimizer apply, snapshot preparation, manifest publication |
| Concurrency | Expired leases, delayed acknowledgments, verifier retries, daemon restart, analytics outage |
| Resources | Measured peak GPU/host RAM, disk-full handling, queue limits, slow storage/backpressure |
| Quality | Fixed-budget and target-based learning comparisons, multiple seeds, sealed evaluation |

CPU CI runs numerical tests, C++ sanitizers, schema compatibility, and randomized operation histories. A scheduled/manual self-hosted H100 workflow runs actual model smoke and integration tests. Large performance regressions require a pinned environment and repeated measurements, not one noisy shared CI run.

## 13. Implementation work packages

### M0 - Reproducible baseline and hardware envelope

Deliver the environment lock, hardware manifest, a deterministic verifier, stock GRPO learning curves, and separate generation/teacher-force/update/peak-memory measurements. Confirm the task is neither unsolvable nor saturated.

Exit: finite, repeatable learning with usable memory headroom on the one H100.

### M1 - Loss-correct reference replay

Implement the schema and Python reference buffer solely as an executable oracle. Integrate immutable behavior probabilities, group advantages, and a fixed replay policy. Add CPU loss fixtures and fresh-only parity tests before performance work.

Exit: all numerical invariants pass; fixed replay's learning curve and total cost are measured.

### M2 - C++ service and persistent protocol

Implement `reprised`, gRPC protocol, bounded sampling, durable metadata/payload writes, reservations, and event outbox. Replay the same seeded operation trace through the Python and C++ backends.

Exit: identical selected group identities/accounting for the trace; no silent mutation or duplicate use on retries.

### M3 - Joint recovery and single-GPU phase control

Add coordinated checkpoints, fencing, branch-aware recovery, and vLLM weight/sleep transitions. Inject crashes across all commit boundaries and enforce memory limits.

Exit: correct accounting and provenance after restoration; end-to-end benchmark includes checkpoint/phase overhead.

### M4 - Adaptive policy and analytics

Implement bounded drift/cost control, diagnostics, batched ClickHouse ingestion, and deduplicated reports. Add a controller-off ablation and capability to export run traces without a hosted service.

Exit: decisions are reproducible/explainable from recorded diagnostics; an analytics outage cannot corrupt training.

### M5 - Evidence and public release

Run the declared multi-seed comparisons and second task when feasible, publish performance/quality curves and limitations, package the adapter/daemon, and document one-H100 setup.

Exit: claims match measured evidence; users can reproduce the experiment without a second GPU or a paid backend.

These are ordered milestones with acceptance gates, not delivery-time promises. No implementation has been started by this document.

## 14. Proposed repository and user interface

```
repriserl/
  README.md
  LICENSE
  docs/
    design.md
    algorithm-contract.md
    single-h100.md
    recovery.md
    benchmark-protocol.md
  proto/
    replay.proto
  cpp/
    replayd/
    storage/
    sampling/
    tests/
  python/repriserl/
    reference/
    adapters/trl/
    controller/
    verifiers/
    cli/
  configs/
    qwen15b_h100_fresh.yaml
    qwen15b_h100_fixed.yaml
    qwen15b_h100_adaptive.yaml
  benchmarks/
    learning/
    recovery/
    service_load/
  analytics/
    schema/
    queries/
    dashboards/
  tests/
    numerical/
    integration/
    faults/
  examples/
  packaging/
  reports/
```

Planned command contracts: `doctor` checks hardware/software; `prepare` builds versioned datasets; `run` executes a named training profile; `compare` produces deduplicated quality/cost reports; `recover` restores a joint checkpoint; `export` writes portable traces/model metadata. These are interface requirements, not commands available today.

The minimal profile runs without ClickHouse and without C++ while M1 is under investigation. The supported service profile enables the C++ daemon; the analytical profile additionally launches ClickHouse. Avoid automatic background training or hidden external API calls.

## 15. Open-source contribution strategy

Release the algorithm contract and benchmark protocol alongside the implementation. Provide a narrow TRL adapter and discuss stable prepared-rollout hooks upstream only after a reference implementation exists. A connector to an existing framework is more useful than forcing users into a new trainer.

Use the Hub for permitted checkpoints, model cards, dataset manifests, and reproducible run artifacts. Publish only licensed, non-sensitive rollout content; keep per-token provenance and private task material opt-in. Code license does not relicense datasets or model outputs.

Evaluate Reverb as an alternative backend or comparator. Do not claim that gRPC transport, sampling, or checkpoint storage are inventions. The core contribution is the LLM-specific semantics and evidence that the complete method improves the real compute/quality trade-off. [S2]

Public artifacts must contain only material authorized for release, with compatible rights for code, models, and datasets. Keep credentials, private data, machine identifiers, and unpublished run material outside the repository.

## 16. Risks and stop conditions

| Risk | Response |
|---|---|
| Replay lowers accuracy or diversity | Fresh-data floor, short horizon, fixed-replay baseline, sealed evaluation; narrow or stop the claim |
| Rescoring costs more than regeneration | Account for every phase; candidate-only rescoring; controller falls back to fresh generation |
| Benefit is merely curriculum or extra num_iterations | Strong tuned comparator and prompt-matched ablation |
| Trainer upgrade changes probabilities or masks | Pin versions and run loss-contract parity tests before updating |
| CPU service underutilizes H100 | Batch RPC/storage, bound synchronization, measure no-service reference; optimize only a demonstrated bottleneck |
| Checkpoint cost dominates | Measure interval trade-off and include cost; never disable durability silently in only one arm |
| Single-node tests mistaken for distributed proof | Publish the deployment boundary; use CPU sharding simulations only as labeled tests |
| Synthetic reasoning does not generalize | Add a public second task before broad claims; state model/task limitations |

## 17. Final project definition

RepriseRL is a one-H100-first, policy-aware experience replay backend for LLM post-training. Its first meaningful deliverable is not a cluster or a dashboard: it is a numerically validated training integration with measured learning curves. The C++ service then makes the validated experience lifecycle durable, asynchronous, observable, and reusable.

The desired research statement is: under a declared model/task/algorithm setup, adaptive replay reaches the agreed capability with less total measured GPU time than tuned existing reuse and fixed replay. The desired engineering statement is: the service preserves group/probability/consumption semantics across retries and coordinated recovery. Both require tests and measurements.

---

## References

The design bibliography was assembled on October 13, 2026. The TRL release, experience-replay paper, and model card were rechecked on October 14, 2026. Implementation must validate APIs and record immutable versions.

- [S1] Arnal et al., Efficient RL Training for LLMs with Experience Replay: https://arxiv.org/html/2604.08706v1
- [S2] DeepMind Reverb: https://github.com/google-deepmind/reverb
- [S3] Qwen2.5-1.5B-Instruct model card: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- [S4] Transformers GPU memory accounting: https://huggingface.co/docs/transformers/model_memory_anatomy
- [S5] TRL releases: https://github.com/huggingface/trl/releases/tag/v1.15.0; dependency declaration: https://github.com/huggingface/trl/blob/v1.15.0/pyproject.toml
- [S6] TRL GRPO trainer: https://huggingface.co/docs/trl/en/grpo_trainer
- [S7] TRL asynchronous GRPO: https://huggingface.co/docs/trl/main/en/async_grpo_trainer
- [S8] vLLM sleep mode: https://docs.vllm.ai/en/latest/features/sleep_mode/
- [S9] Inspected TRL trainer source (moving branch; freeze a tested release at implementation): https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/grpo_trainer.py
- [S10] RocksDB atomicity/WAL notes: https://github.com/facebook/rocksdb/wiki/Atomic-flush
- [S11] ClickHouse ReplacingMergeTree semantics: https://clickhouse.com/docs/reference/engines/table-engines/mergetree-family/replacingmergetree
- [S12] GSM8K dataset card: https://huggingface.co/datasets/openai/gsm8k
