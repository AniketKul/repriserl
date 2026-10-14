# RepriseRL engineering guidance

This repository contains the project specification, implementation plan, and tested CPU foundations. Follow `docs/KICKOFF.md` for completed work and remaining trainer/replay gates. Run CPU tests with `python -m pytest -q` and build with `python -m build --no-isolation` in the isolated development environment described in `README.md`.

## Mission and fixed constraints

- Build policy-aware rollout reuse for LLM post-training with a measurable quality/compute benefit.
- Target one full H100, nominally 80 GB. Inspect device allocation before each run; prior access does not reserve a device or guarantee availability.
- CPU services are asynchronous; generation, candidate rescoring, and training time-share that one GPU. Do not require AsyncGRPOTrainer or separate GPU pools for v1.
- C++20 is the service language; Python/PyTorch/TRL remain appropriate for the trainer and numerical oracle. This is not a request to reimplement autograd in C++.
- The repository is `AniketKul/repriserl`; the planned package and daemon are `repriserl` and `reprised`. The repository uses Apache-2.0. Package-registry and model publication require explicit authorization.

## Sources of truth

Read `docs/RepriseRL_Implementation_Proposal.md`, `docs/IMPLEMENTATION_BRIEF.md`, and `docs/plans/2026-10-14-repriserl-m0-m1.md`. Update `docs/KICKOFF.md` with evidence. Proposed CLI commands are not installed capabilities.

## Learning correctness

- Reuse complete prompt groups, including failed completions. Do not use success-only replay in v1.
- Preserve original advantages, masks, generating policy lineage, sampling settings, verifier identity, and both old probability arrays.
- Distinguish current-policy drift from inference/training backend mismatch. Reassessment creates derived diagnostics; it never overwrites old behavior probabilities.
- Use the declared clipped surrogate; do not describe it as unbiased or theoretically safe. Reject missing or invalid provenance explicitly.
- Make std convention, EOS/truncation handling, normalization, correction clipping, sampler semantics, and reference-model settings explicit and identical across compared arms.
- Normalize across the whole optimizer batch. Test unequal lengths and gradient accumulation.
- Include zero-advantage groups and failed/rejected work in cost accounting consistently.
- Match a tuned baseline that already supports multiple updates per generated batch.

## Service correctness

Use bounded queues, batch identities, idempotent inserts, complete-group reservations, deadlines, fencing epochs, immutable payload checksums, and joint checkpoints. Never assert exactly-once gradient application. Local durability is not protection against host loss. ClickHouse is analytical, never an admission or optimizer transaction authority.

## Engineering process

Inspect before changing. Preserve unrelated/uncommitted work. Reconcile and review the phase plan before implementation. Use focused modules, test-first changes, version-pinned integrations, typed interfaces, and small reviewable commits when authorized. Numerical replay gates precede service optimization; fixed replay precedes the adaptive controller. Do not add a dashboard, cluster deployment, or new kernel to hide a failed learning experiment.

## Resource and security boundaries

Do not rent GPUs, install system-wide packages, discover or use unrelated credentials, invent remote hosts, stop other GPU jobs, upload private material, or launch an unbounded training sweep. Publish repository changes only within explicit user authorization. Inspect resource availability without exposing secrets. Respect explicit user run budgets and put step/time/checkpoint limits on approved GPU runs. If access is absent, implement CPU tests and report GPU checks as unexecuted. Use a restricted grammar verifier, never eval/exec on model text. Keep raw prompts out of default analytics.

## Verification and claims

Run the exact tests before calling them passed. Distinguish CPU numerical tests, tiny-model GPU smoke tests, actual learning curves, service-load simulations, and multi-node evidence. Count end-to-end H100 time including reassessment, transfers, waits, checkpoints, evaluation, and buffer population. Separate hypothetical examples and third-party paper results from measurements. The 20% reduction and 2-percentage-point margin are proposed targets, not promises.

At each checkpoint report changed files, commands, outcomes, limitations, resource use, current milestone, and next action. Write unresolved choices down rather than silently filling them with purported facts.

Keep each authored file below 500 lines. Keep secrets, private connection details, conversation exports, and unreviewed run artifacts outside tracked files. Use repository-local commit identity; do not add a co-author trailer unless explicitly configured by the project.
