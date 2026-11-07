# Fauxtography — AI Image Detection

Detect whether an image is a **real photograph** or **AI-generated**, by
comparing three approaches on a shared test set:

| # | Approach | Type | Training |
|---|----------|------|----------|
| 1 | Baseline — fine-tuned ResNet | CNN | fine-tune |
| 2 | CLIP zero-shot | foundation (vision-language) | none |
| 3 | DINOv2 + linear probe | foundation (self-supervised vision) | linear head only |

*MAI644 Computer Vision* final project — plan in
[`Docs/technical_plan.md`](Docs/technical_plan.md),
team split in [`Docs/work_division.md`](Docs/work_division.md).

## Repo layout

```
Docs/                     plan + work division + per-model analysis write-ups
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
# create venv & install
uv venv --python 3.15 .venv
uv pip install -e ".[dev]"        # or: pip install -e ".[dev]"

# prepare data (real + fake folders) -> data/train|val|test/{real,fake}
python scripts/prepare_data.py --source-real <real_dir> --source-fake <fake_dir>

# per-model experiments (implemented in their milestones)
python scripts/train_baseline.py            # shared baseline
python scripts/eval_clip.py                 # collaborator A
python scripts/extract_dinov2.py            # collaborator B
python scripts/train_linear_probe.py        # collaborator B

# compare & test
python scripts/compare_models.py
pytest
```

Each script reads `configs/<model>.yaml` (override with `--config <path>`).
Runs are reproducible via `seed` in each config.

## Hardware

Scripts auto-detect CUDA → MPS (Apple Silicon) → CPU. Verified working on an
Apple M4 (Python 3.15, torch 2.15).

## Git workflow

- Personal branches from `dev` → PR into `dev` (integration buffer)
- `dev` → PR into `main` (final deliverable)