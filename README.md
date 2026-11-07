# Fauxtography — AI Image Detection

**Semester Group Project — MAI644 Computer Vision.**

Detect whether an image is a **real photograph** or **AI-generated**, by
comparing three approaches on a shared test set:

| # | Approach | Type | Training |
|---|----------|------|----------|
| 1 | Baseline — fine-tuned ResNet | CNN | fine-tune |
| 2 | CLIP zero-shot | foundation (vision-language) | none |
| 3 | DINOv2 + linear probe | foundation (self-supervised vision) | linear head only |

*MAI644 Semester Group Project — Computer Vision* — key deliverables:
[`docs/technical_plan.md`](docs/technical_plan.md), team split in
[`docs/work_division.md`](docs/work_division.md), setup in
[`docs/setup.md`](docs/setup.md).

**Install:** use either [Poetry](docs/setup.md#option-a--poetry-preferred-by-the-repo-author) or
[pip + requirements.txt](docs/setup.md#option-b--plain-pip-requirementstxt) —
see [`docs/setup.md`](docs/setup.md) for full instructions.

## Repo layout

```
docs/                     plan + work division + per-model analysis write-ups
notebooks/                analysis layer: results, error inspection, verdicts
src/fauxtography/         package (data, models, evaluate, utils, config, cli)
scripts/                  CLI wrappers + make_notebooks generator
configs/                  per-model YAML configs
tests/                    smoke tests
data/                     (git-ignored) raw + split dataset
experiments/              (git-ignored) checkpoints + cached features
results/                  committed metrics + figures for the report
```

## Status

Scaffolding complete: package layout, configs, shared data pipeline, CLI,
and per-model API stubs are in place. Model implementations are intentionally
stubbed (`NotImplementedError`) — filled in per owner as the milestones land.
The analysis layer lives in `notebooks/` (results, error inspection, verdicts),
run in Colab on a GPU; the pipeline itself runs headless via the `fauxtography`
CLI.

## Workflow

```bash
# --- Poetry ---
poetry install --all-extras

# --- or plain pip ---
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .

# CLI (installed by the package; run commands through it)
poetry run fauxtography --help             # list every subcommand

# prepare data (real + fake folders) -> data/train|val|test/{real,fake}
poetry run fauxtography prepare-data --source-real <real_dir> --source-fake <fake_dir>

# per-model experiments (implemented in their milestones)
poetry run fauxtography train-baseline            # shared baseline
poetry run fauxtography eval-clip                 # collaborator A
poetry run fauxtography extract-dinov2            # collaborator B
poetry run fauxtography train-linear-probe        # collaborator B

# compare & test
poetry run fauxtography compare
pytest
```

All subcommands accept `--help`, and `fauxtography env` prints the
accelerator/torch report (run it first on Colab to confirm a GPU is attached).
Each experiment command reads `configs/<model>.yaml` by default (override with
`--config <path>`); runs are reproducible via `seed` in each config. The legacy
`scripts/*.py` files remain as thin wrappers and behave identically. See
[`docs/setup.md`](docs/setup.md), which includes a full **Google Colab (GPU)**
section.

## Hardware

Devices resolve in one place (`utils/device.py`): CUDA → MPS (Apple Silicon) →
CPU. Verified on an Apple M4 (Python 3.14, torch 2.15); training and eval are
meant to run on a **Colab GPU** (T4/L4 is enough — no TPU needed). Set
`FAUXTOGRAPHY_DATA_ROOT` / `RUN_DIR` / `RESULTS_DIR` to redirect where data and
outputs live (e.g. a Colab Drive mount).

## Git workflow

- Personal branches from `dev` → PR into `dev` (integration buffer)
- `dev` → PR into `main` (final deliverable)