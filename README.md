# Fauxtography — AI Image Detection

**Semester Group Project — MAI644 Computer Vision.**

Detect whether an image is a **real photograph** or an **AI-generated image**,
by comparing three approaches on a shared test set:

| # | Approach | Type | Training |
|---|----------|------|----------|
| 1 | Baseline — fine-tuned ResNet | CNN | fine-tune |
| 2 | CLIP zero-shot | foundation (vision-language) | none |
| 3 | DINOv2 + linear probe | foundation (self-supervised vision) | linear head only |

**Read first:** the plan in [`docs/technical_plan.md`](docs/technical_plan.md)
(method + analysis strategy), the team split in
[`docs/work_division.md`](docs/work_division.md), and
[`docs/setup.md`](docs/setup.md) (install + Colab).

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

**Pipeline (the code)** — run via the `fauxtography` CLI:

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

**Analysis (the story)** — what you read and present:

```
notebooks/              setup_colab · 00_data_overview · analysis_clip ·
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
| 0 | `fauxtography env` | confirm accelerator (run first in Colab) | — (prints) |
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

## Running it — local, then Colab

```bash
# local: checks that everything is wired up (data prep works on any machine)
poetry run fauxtography prepare-data --source-real <real> --source-fake <fake>

# Colab: real training happens here, on a GPU
```
1. Open **`notebooks/setup_colab.ipynb`** in a **GPU** runtime and run it
   top-to-bottom: it installs the Python 3.14 env with `uv`, installs the
   package, verifies the accelerator (`fauxtography env`), splits the data,
   and runs the model commands. Data + outputs go to your **Drive** via
   `FAUXTOGRAPHY_DATA_ROOT` / `RUN_DIR` / `RESULTS_DIR`, so they survive
   session disconnects (see [`docs/setup.md`](docs/setup.md) — Colab section).
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
The **model implementations are intentionally stubbed**
(`NotImplementedError`) — they are filled in per owner as the milestones land,
and the stub contracts already define their inputs/outputs (labels
`0=real, 1=fake`) so each half drops in without touching the other's code.

Notes:
- `notebooks/` are generated from templates via `scripts/make_notebooks.py`;
  edit templates and re-run, don't hand-edit the JSON.

---

## Hardware & paths

- Device resolution: CUDA → MPS (Apple Silicon) → CPU, in `utils/device.py`.
  Verified on an Apple M4 (Python 3.14, torch 2.15); heavy work is meant for a
  Colab GPU.
- Path overrides via env vars (set once redirects every command):
  `FAUXTOGRAPHY_DATA_ROOT`, `FAUXTOGRAPHY_RUN_DIR`, `FAUXTOGRAPHY_RESULTS_DIR`.

## Git workflow

- Personal branches from `dev` → PR into `dev` (integration buffer)
- `dev` → PR into `main` (final deliverable)