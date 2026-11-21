# Fauxtography - Detecting AI-Generated Images

**Semester group project for MAI644 - Computer Vision, M.Sc. in Artificial
Intelligence, University of Cyprus, 2026.**

Distinguish whether an image is a **real photograph** or an **AI-generated image**,
by comparing three approaches on a shared test set:

| # | Approach | Type | Training |
|---|----------|------|----------|
| 1 | Baseline - fine-tuned ResNet | CNN | fine-tune |
| 2 | CLIP zero-shot | foundation (vision-language) | none |
| 3 | DINOv2 + linear probe | foundation (self-supervised vision) | linear head only |

**Read first:** the plan in [`docs/technical_plan.md`](docs/technical_plan.md)
(method + analysis strategy), the dataset choice in
[`docs/dataset_research.md`](docs/dataset_research.md), the team split in
[`docs/work_division.md`](docs/work_division.md),
[`docs/contributions.md`](docs/contributions.md) (novelty and claims),
and [`docs/setup.md`](docs/setup.md) (install + Colab).

---

## Install

Use either [Poetry](docs/setup.md#option-a--poetry-preferred-by-the-repo-author) or
[pip + requirements.txt](docs/setup.md#option-b--plain-pip-requirementstxt).
Both install the same pinned set on **Python 3.14**:

```bash
# Poetry
poetry install --all-extras

# or plain pip
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

The `fauxtography` console command comes from the package itself (so it needs
`pip install -e .` / `poetry install` to exist).

---

## Repo layout

**Pipeline (the code)** - run via the `fauxtography` CLI:

```
src/fauxtography/       the whole pipeline in one package
  ├── cli.py            argument parsing + command dispatch (the only CLI)
  ├── config.py         typed per-model configs + env-var path overrides
  ├── data/             dataset (train/val/test, real=0 / fake=1) + transforms
  ├── models/           baseline.py · clip.py · dinov2.py · linear_probe.py
  ├── evaluate/         metrics.py + visual.py (accuracy, confusion matrix)
  └── utils/            device.py (cuda→mps→cpu) · seed.py · logging.py
configs/                baseline.yaml · clip.yaml · dinov2.yaml
tests/                  smoke tests for the shared infrastructure
scripts/                notebook generator (make_notebooks.py)
```

**Analysis (results & verdicts)** - the presentation layer:

```
notebooks/              fauxtography-on-colab · 00_data_overview · analysis_clip ·
                        analysis_dinov2 · final_report
results/                committed metrics + figures (read by the notebooks)
docs/                   plan, work division, setup, per-model write-ups
```

**Git-ignored (local/Drive only):**

```
data/                   raw + split dataset
experiments/            checkpoints + cached DINOv2 features
```

The split is intentional: **code + configs** drive the pipeline;
**notebooks + results** tell the story. Notebooks are read-only consumers of
what the CLI writes to `results/`, so two people can edit them safely.

---

## How the pipeline flows

Each step is one `fauxtography` subcommand. Read the flow top-to-bottom:

| Step | Command | What happens | Writes to |
|------|---------|--------------|-----------|
| 0 | `fauxtography env` | confirm accelerator (run first in Colab) | - (prints) |
| 1 | `fauxtography prepare-data` | real/ + fake/ → stratified train/val/test (copies, reproducible via `--seed`) | `data/train\|val\|test/{real,fake}` |
| 2 | `fauxtography train-baseline` | fine-tune ResNet, save checkpoint | `experiments/runs/baseline/`, `results/baseline/` |
| 3 | `fauxtography eval-clip` | zero-shot CLIP on the test split (no training) | `results/clip/` |
| 4 | `fauxtography extract-dinov2` | cache [CLS] features for all splits | `experiments/runs/dinov2/features/` |
| 5 | `fauxtography train-linear-probe` | linear head on cached features, evaluate | `results/dinov2/` |
| 6 | `fauxtography compare` | aggregate all three → comparison table | `results/comparison.csv` |

Under the hood, one experiment flows like this:

```
configs/<model>.yaml ──load_config()──▶ typed config object
                                              │
data/<train,val,test> ──FauxtoDataset──▶ dataloader (labels 0=real / 1=fake)
                                              │
models/<model>.py ──model──▶ predictions on the test split
                                              │
evaluate/metrics.py ──y_true,y_pred──▶ metrics.json + confusion matrix
                                              │
results/<model>/ ◀────────written────────────┘
      │
      └──────────── notebooks/analysis_<model>.ipynb  (verdict)
```

Seeds are fixed in each config; the device resolves once via
`utils/device.py` (CUDA → MPS → CPU), so runs are reproducible and the same
code runs on your laptop and on Colab.

---

## Running it - local, then Colab

```bash
# local: checks that everything is wired up (data prep works on any machine)
poetry run fauxtography prepare-data --source-real <real> --source-fake <fake>

# Colab: real training happens here, on a GPU
```
1. Open **`notebooks/fauxtography-on-colab.ipynb`** in a **GPU** runtime and
   run it top-to-bottom: one notebook handles everything - it installs the
   Python 3.14 env with `uv`, installs the package, verifies the accelerator
   (`fauxtography env`), splits the data, and runs the model commands. Data + outputs go to your **Drive** via
   `FAUXTOGRAPHY_DATA_ROOT` / `RUN_DIR` / `RESULTS_DIR`, so they survive
   session disconnects (see [`docs/setup.md`](docs/setup.md) - Colab section).
2. Open `00_data_overview.ipynb`, then `analysis_clip.ipynb` /
   `analysis_dinov2.ipynb` to inspect results and write verdicts; finish with
   `final_report.ipynb` for the side-by-side.

**GPU, not TPU.** Only the baseline truly trains; a free-tier T4 handles all
three models at project scale. TPU would require `torch_xla` and a different
data path for no benefit here.

---

## Status

Scaffolding is complete: package, configs, shared data pipeline, CLI,
device/path infra, Colab setup notebook, and analysis notebooks are in place.
All three models (ResNet fine-tune, DINOv2 + linear probe, CLIP zero-shot)
are implemented and have committed results under `results/baseline/`,
`results/dinov2/`, and `results/clip/`. CLIP's canonical result is its best
prompt strategy ("balanced descriptions", heldout 0.7277); the full prompt
sweep lives under `results/clip/<strategy>/`.

A new model joins the same way: config + `models/<name>.py` + a CLI
command writing `results/<name>/`, then the analysis notebook (see
"Adding another model" below).

Notes:
- `notebooks/fauxtography-on-colab.ipynb`, `00_data_overview.ipynb` and
  `final_report.ipynb` are generated from `scripts/make_notebooks.py`; edit
  templates and re-run, don't hand-edit the JSON.
- `analysis_clip.ipynb` and `analysis_dinov2.ipynb` are hand-maintained:
  their content and figures live in the notebook files themselves, and the
  generators for them are commented out in `make_notebooks.py` so that
  regenerating never silently replaces the written-up analysis.

---

## Hardware & paths

- Device resolution: CUDA → MPS (Apple Silicon) → CPU, in `utils/device.py`.
  Verified on an Apple M4 (Python 3.14, torch 2.15); heavy work is meant for a
  Colab GPU.
- Path overrides via env vars (set once redirects every command):
  `FAUXTOGRAPHY_DATA_ROOT`, `FAUXTOGRAPHY_RUN_DIR`, `FAUXTOGRAPHY_RESULTS_DIR`.

<a name="contributing"></a>

## Adding another model (open to contributors)

The project's wider aim is a **reference point**: a repo where, for each
foundation model, anyone can reproduce the numbers, rerun the eval, and add
or swap models without touching the comparison machinery. The dataset
splits and seeds are fixed, results are committed, and the comparison table
is cheap to re-run, so both *rerunning the repo* and *dropping in a newer
model* are contributions that keep the table meaningful.

The repo is set up so a new foundation model drops in by following an
existing one - the comparison is a growing table, not a fixed one. The
pattern, one file at a time:

1. `configs/<model>.yaml` - model settings (copy `configs/dinov2.yaml`).
2. `src/fauxtography/models/<model>.py` - a wrapper exposing the model
   (copy `models/dinov2.py`; the stub contracts already fix the
   inputs/outputs: labels `0=real, 1=fake`).
3. A CLI command in `cli.py` that writes `results/<model>/` (metrics JSON
   + figures). `compare` and `analyze-fusion` read that layout
   automatically, so a new model joins the comparison table and the
   ensemble without touching them.
4. `notebooks/analysis_<model>.ipynb` - the verdict, written from the
   committed results (see `notebooks/analysis_dinov2.ipynb` as the
   template).

How a model and its analysis are owned: `docs/work_division.md`. What a
new model family adds to the comparison - and what is not a contribution:
`docs/contributions.md`.

## Git workflow

- Personal branches from `dev` → PR into `dev` (integration buffer)
- `dev` → PR into `main` (final deliverable)