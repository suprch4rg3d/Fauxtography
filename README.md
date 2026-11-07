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
src/fauxtography/         package (data, models, evaluate, utils, config)
scripts/                  runnable pipeline entry points
configs/                  per-model YAML configs
tests/                    smoke tests
data/                     (git-ignored) raw + split dataset
experiments/              (git-ignored) checkpoints + cached features
results/                  committed metrics + figures for the report
```

## Status

Scaffolding complete: package layout, configs, shared data pipeline
(`prepare_data.py`), and per-model API stubs are in place. Model
implementations are intentionally stubbed (`NotImplementedError`) — they are
filled in per owner as the milestones land.

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

All subcommands accept `--help`. Each experiment command reads
`configs/<model>.yaml` by default (override with `--config <path>`); runs are
reproducible via `seed` in each config. The legacy `scripts/*.py` files remain
as thin wrappers (`python scripts/prepare_data.py ...`) and behave identically.
See [`docs/setup.md`](docs/setup.md) for detailed setup and troubleshooting.

## Hardware

Scripts auto-detect CUDA → MPS (Apple Silicon) → CPU. Verified working on an
Apple M4 (Python 3.14, torch 2.15).

## Git workflow

- Personal branches from `dev` → PR into `dev` (integration buffer)
- `dev` → PR into `main` (final deliverable)