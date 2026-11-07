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
[`Docs/technical_plan.md`](Docs/technical_plan.md), team split in
[`Docs/work_division.md`](Docs/work_division.md), setup in
[`Docs/setup.md`](Docs/setup.md).
team split in [`Docs/work_division.md`](Docs/work_division.md),
setup in [`Docs/setup.md`](Docs/setup.md).

**Install:** use either [Poetry](Docs/setup.md#option-a--poetry-preferred-by-the-repo-author) or
[pip + requirements.txt](Docs/setup.md#option-b--plain-pip-requirementstxt) —
see [`Docs/setup.md`](Docs/setup.md) for full instructions.

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
# --- Poetry ---
poetry install --all-extras

# --- or plain pip ---
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .

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
Runs are reproducible via `seed` in each config. See
[`Docs/setup.md`](Docs/setup.md) for detailed setup and troubleshooting.

## Hardware

Scripts auto-detect CUDA → MPS (Apple Silicon) → CPU. Verified working on an
Apple M4 (Python 3.14, torch 2.15).

## Git workflow

- Personal branches from `dev` → PR into `dev` (integration buffer)
- `dev` → PR into `main` (final deliverable)