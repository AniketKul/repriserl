# RepriseRL kickoff and status

Updated: 2026-10-14.

## Completed preparation

- Reviewed the final project specification and first-phase implementation plan.
- Prepared the initial documentation-only public repository for `AniketKul/repriserl`.
- Completed initial read-only hardware discovery; raw inventory and connection details are held outside the repository.
- Checked availability of the proposed TRL release and its Python/dependency requirements.
- Adapted the M0/M1 task map to the new repository without changing the learning objective or numerical acceptance gates.

## Implemented CPU foundations

The first implementation slice covers [M0/M1 Tasks 1 and 2](plans/2026-10-14-repriserl-m0-m1.md):

- `pyproject.toml` and `config/requirements-dev.lock`: installable Python package,
  pinned CPU development dependencies, wheel and source distribution builds.
- `python/repriserl/config.py` and `configs/qwen15b_h100_fresh.yaml`: immutable
  settings, explicit run budget, safe YAML parsing, duplicate/unknown key rejection,
  and unset model/numerical choices that training integration must resolve.
- `python/repriserl/hardware.py`: bounded read-only NVIDIA queries, CPU fallback,
  explicit unknown/unavailable states, and conservative full-H100 availability.
  Malformed identifiers and undecodable command output cannot imply readiness.
- `python/repriserl/verifiers/number_expression.py`: complete-expression grammar,
  exact rational arithmetic, exact repeated-number accounting, input/work bounds,
  canonical problem IDs, and a versioned verifier contract/digest.
- `python/repriserl/data/problems.py`: immutable deterministic train/development/test
  splits, global canonical deduplication, bounded generation, and generation/split
  digests. The initial four-number distribution has not been difficulty-calibrated.
- `tests/unit/`: configuration, hardware, adversarial expression, reproducibility,
  and split-separation coverage. `.github/workflows/ci.yml` runs CPU checks on
  Python 3.11 and 3.12, builds distributions, and imports the wheel outside source.

## Verification evidence

At implementation commit `261f11e40890350627d0d4f1e62b10f232111643`:

| Check | Command / method | Outcome |
|---|---|---|
| macOS, Python 3.12 | `python -m pytest -q` | 147 passed |
| Linux, Python 3.12 | `python -m pytest -q` in isolated environment | 147 passed |
| Both platforms | `python -m build --no-isolation` | Source distribution and wheel built |
| Linux installed package | Install wheel; use `python -I` outside source | Config, verifier, split digest and hardware imports passed |
| Reproducibility | Seed 7, sizes `(3, 1, 1)`, fixed golden fixture | Same digest on macOS and Linux |
| Actual hardware inspector | Read-only query on an authorized host | Full H100 observed available; no GPU allocation or model execution |

Raw hardware inventory, connection details and execution receipts stay outside
the public repository. Availability can change after inspection. Full proposed
20,000/512/2,048 splits have not been generated or used for model selection.

No model download, training, learning benchmark, replay backend, or measured
performance improvement has been produced. These CPU foundations do not satisfy
the M0 learning-evidence exit gate.

## Next implementation slice

Task 3: validate the pinned trainer environment and probability/loss conventions,
then implement the stock baseline driver and phase-time accounting. Tasks 4–7
cover numerical replay, bounded storage, the trainer adapter and reporting.

## Gates for subsequent work

- Create an isolated Python >=3.11 environment and validate the entire pinned trainer stack.
- Review the selected trainer's advantage standard deviation, EOS/truncation handling, action masks, probability fields, and full-batch normalization.
- Establish an explicit step/time budget and recheck exclusive device availability before the first H100 pilot.
- Validate fresh-only loss and gradient parity before fixed-replay learning comparisons.
- Use the pilot to freeze target accuracy, evaluation cadence, tuning limits, and the test non-inferiority protocol before confirmatory runs.

## Progress reporting

For each implemented task, record changed paths, exact commands and exit outcomes, CPU/GPU distinction, artifact locations, measured resource use, accepted decisions, remaining gates, and the next task. Include only public-safe evidence here; never copy raw credentials, personal paths, host inventories, or private task data.

The [full proposal](RepriseRL_Implementation_Proposal.md) retains the architecture and M0-M5 contracts. The [phase plan](plans/2026-10-14-repriserl-m0-m1.md) defines concrete module ownership and verification requirements.
