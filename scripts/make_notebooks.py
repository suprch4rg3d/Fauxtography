"""Generate the analysis notebooks as valid .ipynb files.

Run once (or re-run to refresh): `python scripts/make_notebooks.py`.
Notebooks import the installed fauxtography package, so run from the repo
root with the venv active or inside Colab after `pip install -e .`.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_HEADER = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.14"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build(cells: list[dict]) -> dict:
    nb = json.loads(json.dumps(NB_HEADER))  # deep copy
    nb["cells"] = cells
    return nb


def write(name: str, cells: list[dict]) -> None:
    out = Path("notebooks") / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(cells), indent=1))
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# 00_data_overview.ipynb
# --------------------------------------------------------------------------- #
def data_overview() -> list[dict]:
    return [
        md(
            """\
# 00 - Data Overview

**Purpose:** confirm the dataset is loaded as expected and inspect the class
balance / a few samples before any model work. Executes against the split
directories built by `prepare-data` (or committed to a shared Drive folder).
"""
        ),
        md("## 1. Environment"),
        code(
            """\
# In Colab with a GPU runtime:
# !pip install -e .            # install fauxtography + deps (see docs/setup.md)
import sys
assert "fauxtography" in sys.modules or True
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = Path("data")   # <- point at your Drive path in Colab if needed
print("data root:", DATA_ROOT.resolve() if DATA_ROOT.exists() else "MISSING")
"""
        ),
        md("## 2. Split sizes and class balance"),
        code(
            """\
from fauxtography.data.dataset import FauxtoDataset
from torchvision.transforms import ToTensor

for split in ("train", "val", "test"):
    p = DATA_ROOT / split
    if not p.exists():
        print(f"-- {split}: missing ({p})")
        continue
    ds = FauxtoDataset(p, transform=ToTensor())
    counts = {cls_: 0 for cls_ in ds.classes}
    for _, label in ds:
        counts[ds.classes[label]] += 1
    print(f"{split}: {len(ds)} images {counts}")
"""
        ),
        md("## 3. Sample grid (real vs fake)"),
        code(
            """\
from torchvision.utils import make_grid

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for col, cls in enumerate(ds.classes):
    imgs = [ds[i][0] for i in range(len(ds)) if ds.classes[ds[i][1]] == cls][:8]
    grid = make_grid(imgs, nrow=4).permute(1, 2, 0).numpy()
    axes[col].imshow(np.clip(grid, 0, 1))
    axes[col].set_title(f"{cls} ({len(imgs)} samples)")
    axes[col].axis("off")
plt.tight_layout()
plt.show()
"""
        ),
    ]


# --------------------------------------------------------------------------- #
# analysis_clip.ipynb  +  analysis_dinov2.ipynb
# --------------------------------------------------------------------------- #
def model_analysis(model: str, title: str, owner: str) -> list[dict]:
    name = model  # 'clip' or 'dinov2'
    return [
        md(
            f"""\
# Analysis - {title}

**Owner:** {owner}   |   **Model dir:** `results/{name}/`

What this notebook adds: the numbers behind the verdict. It loads the saved
metrics + predictions produced by the eval script, redraws the confusion
matrix for this model, and documents the *manual error inspection* - the
core of the project's analysis strategy.
"""
        ),
        md("## 1. Headline metrics"),
        code(
            f"""\
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

results = Path("results") / "{name}"
metrics = json.loads((results / "metrics.json").read_text()) if (results / "metrics.json").exists() else None
y_true = np.load(results / "y_true.npy") if (results / "y_true.npy").exists() else None
y_pred = np.load(results / "y_pred.npy") if (results / "y_pred.npy").exists() else None

if metrics:
    print(f"accuracy: {{metrics['accuracy']:.4f}}")
    import pandas as pd
    print(pd.DataFrame(metrics["classification_report"]).T.round(3))
else:
    print("No metrics yet - run `fauxtography eval-clip` (for CLIP) first.")
"""
        ),
        md("## 2. Confusion matrix"),
        code(
            """\
from sklearn.metrics import confusion_matrix
from fauxtography.data.dataset import CLASS_NAMES, REAL_IDX, FAKE_IDX

if y_true is not None:
    cm = confusion_matrix(y_true, y_pred, labels=[REAL_IDX, FAKE_IDX])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.title("Confusion matrix")
    plt.show()
    # Where do we fail? (0,1)=real mislabelled fake; (1,0)=fake passed as real
else:
    print("No predictions yet (see cell 1).")
"""
        ),
        md("## 3. Manual error inspection"),
        code(
            """\
# Load a few misclassified images and inspect what confused the model.
# Fill in: what visual artifacts / styles fooled the model? (presentation gold.)
# NOTE: to see the actual images, load the test set with fauxtography
# (FauxtoDataset) and reconstruct the misclassified file paths via the
# dataset's .samples attribute. Fill in your observations below.
import numpy as np
if y_true is not None:
    errors = np.where(y_true != y_pred)[0]
    print(f"{len(errors)} misclassified (of {len(y_true)}): {len(errors)/len(y_true):.1%}")
    for idx in errors[:6]:
        print("sample index", idx, "true=", CLASS_NAMES[y_true[idx]],
              "pred=", CLASS_NAMES[y_pred[idx]])
else:
    print("Nothing to inspect yet.")
"""
        ),
        md("## 4. Verdict"),
        code(
            """\
# Write your verdict here (this is what goes on the slides):
#   - Where does this model stand versus the others?
#   - What kind of image fools it?
#   - One-sentence takeaway.
print("TODO: verdict for this model - see docs/analysis_<model>.md for the write-up")
"""
        ),
    ]


# --------------------------------------------------------------------------- #
# final_report.ipynb
# --------------------------------------------------------------------------- #
def final_report() -> list[dict]:
    return [
        md(
            """\
# Final Report - comparing baseline / CLIP / DINOv2

**Joint step, both collaborators.** Loads every model's saved metrics and
produces the side-by-side comparison that goes on the presentation slides.
"""
        ),
        md("## 1. Comparison table"),
        code(
            """\
import json
import pandas as pd

rows = []
for name in ("baseline", "clip", "dinov2"):
    path = Path("results") / name / "metrics.json"
    if not path.exists():
        continue
    m = json.loads(path.read_text())
    rows.append({
        "model": name,
        "accuracy": m["accuracy"],
        **{k: m["classification_report"].get(k, {}).get("f1-score") for k in ("real", "fake", "macro avg")},
    })
df = pd.DataFrame(rows).fillna("-")
print(df.to_string(index=False))
df.to_csv(Path("results") / "comparison.csv", index=False)
"""
        ),
        md("## 2. Accuracy bar chart"),
        code(
            """\
if not df.empty:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["model"], df["accuracy"], color="steelblue")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Model comparison")
    plt.show()
else:
    print("No results yet - run baseline, eval-clip, and the DINOv2 probe first.")
"""
        ),
        md("## 3. Takeaway verdicts"),
        code(
            """\
# Both collaborators contribute the final verdicts here (joint analysis).
print("TODO: joint conclusions")
"""
        ),
    ]


# --------------------------------------------------------------------------- #
# fauxtography-on-colab.ipynb - run this FIRST in a Colab session; it does
# everything in one go (env, Drive, data split, training).
# --------------------------------------------------------------------------- #
def fauxtography_on_colab() -> list[dict]:
    return [
        md(
            """\
# Fauxtography on Colab - run everything in one go

Run this first in a **Google Colab GPU** runtime. This notebook does the
whole pipeline: environment, Drive, data split and the model training steps.

This notebook prepares a Colab session to run the fauxtography pipeline on a
GPU. It is *not* the analysis (those live in the other notebooks); it:

1. clones the repo (or reuses an already-mounted Drive copy)
2. creates a **Python 3.14** env with `uv` (Colab's own Python is 3.11/3.12,
   which our deps pin away from - see `docs/setup.md`)
3. installs the package
4. verifies the accelerator with `fauxtography env`
5. splits the dataset and runs the model commands (the GPU steps)
6. copies `results/` + checkpoints to Drive so they survive session
   disconnects

### Running this notebook: one rule to avoid stale edits

If you run this from VS Code's Colab extension, the extension syncs the
notebook you have open **and** cell 1 clones the repo (a second copy).
Editing cells while a runtime is attached edits the synced copy, but the
pipeline runs from the cloned copy - so mid-run edits can appear stale.

**Rule:** edit locally (runtime detached) -> save -> launch the runtime -> run
straight through without editing cells. A self-check cell after the clone
warns loudly if the running copy diverges from the repo's version.

Make sure you know where your dataset is BEFORE running (cell for that
below). Run cells top-to-bottom, in order.
"""
        ),
        md(
            """\
## 1. Environment + repo

Colab runtimes persist nothing between sessions, so each session re-does this
setup. Set ``REPO_URL`` if your fork has a different name. If you'd rather use
a Drive copy, mount Drive (cell 2) and ``cd`` into it instead of cloning.
"""
        ),
        code(
            """\
# Install uv (fast Python env manager that can fetch a 3.14 interpreter)
!pip install -q uv

import os
from pathlib import Path

# The repo must be PUBLIC for this clean clone to work (Settings -> General
# -> change visibility). Default branch is what the runtime fetches - it must
# be up to date (push/merge to main before running).
repo = "/content/fauxtography"
if not Path(repo).is_dir():
    !git clone --depth 1 https://github.com/suprch4rg3d/fauxtography.git {repo}
os.chdir(repo)
print("repo ready at", os.getcwd())
""",
        ),
        md(
            """\
### Self-check: is the running copy in sync with the repo?

The pipeline below runs from the clone at `{repo}`. If you edited the notebook
that the runtime synced here, the cells executing may not match what you see.
This check compares this notebook's cells to the repo's committed version and
warns loudly on any divergence.
"""
        ),
        code(
            """\
import json
import subprocess
from pathlib import Path

repo = "/content/fauxtography"
this = Path("/content/notebooks/fauxtography-on-colab.ipynb")
if not this.exists():
    this = Path(repo) / "notebooks" / "fauxtography-on-colab.ipynb"

def _cells(path):
    try:
        nb = json.loads(Path(path).read_text())
    except Exception:
        return []
    return [
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code"
    ]

# Fetch the repo's committed notebook to compare against (no writes).
subprocess.run(["git", "fetch", "origin", "main"], cwd=repo, capture_output=True)
committed = Path(repo) / "notebooks" / "fauxtography-on-colab.ipynb"

here = _cells(this)
comm = _cells(committed)
if here and comm and here == comm:
    print("OK - running notebook matches the repo version.")
else:
    print("WARNING: running notebook may be out of sync with the repo.")
    print("  Cells here: %d, cells in repo: %d" % (len(here), len(comm)))
    print("  If you edited cells before launching, restart the runtime and re-run.")
""",
        ),
        md(
            """\
## 2. Mount Google Drive (for dataset + persistent results)

The dataset and the outputs you want to keep (results, checkpoints) should
live on Drive, not on Colab's ephemeral disk.
"""
        ),
        code(
            """\
from google.colab import drive
drive.mount("/content/drive")
print("Drive mounted at /content/drive")
"""
        ),
        md(
            """\
## 3. Create the Python 3.14 env and install the package
"""
        ),
        code(
            """\
venv = "/content/fxt-venv"
!uv venv --python 3.14 {venv}
!uv pip install --python {venv}/bin/python -e .
print("package installed into", venv)
"""
        ),
        md(
            """\
## 4. Verify the accelerator

This is the moment to confirm a GPU is actually attached. You should see
``cuda_available: True`` and a GPU name (e.g. Tesla T4). If you see ``cpu``,
use Runtime > Change runtime type and pick a **GPU** accelerator, then rerun.
"""
        ),
        code(
            """\
!/content/fxt-venv/bin/fauxtography env
""",
        ),
        md(
            """\
## 5. Point everything at your data + Drive-backed outputs

Set these once: every command reads them via the config env-var overrides
(see `src/fauxtography/config.py`).
"""
        ),
        code(
            """\
import os
venv = "/content/fxt-venv"
repo = "/content/fauxtography"
os.chdir(repo)
DATA_ROOT = "/content/drive/MyDrive/fauxtography/data"     # WHERE IS YOUR DATASET?
RUN_DIR = "/content/drive/MyDrive/fauxtography/experiments"   # checkpoints / features
RESULTS_DIR = "/content/drive/MyDrive/fauxtography/results"   # committed metrics
RAW_DIR = "/content/drive/MyDrive/fauxtography/data/raw"   # raw/ test lives here

# Export overrides - the CLI/package pick these up automatically.
os.environ["FAUXTOGRAPHY_DATA_ROOT"] = DATA_ROOT
os.environ["FAUXTOGRAPHY_RUN_DIR"] = RUN_DIR
os.environ["FAUXTOGRAPHY_RESULTS_DIR"] = RESULTS_DIR
os.environ["FAUXTOGRAPHY_HELDOUT_TEST"] = os.path.join(RAW_DIR, "test")
print("configured:", DATA_ROOT)
"""
        ),
        md(
            """\
## 6. Download the dataset into Drive (run once)

Downloads the GenImage slice configured in `configs/genimage.yaml` straight
from HuggingFace into Drive - no need to upload gigabytes from your machine.
Re-running is safe; skip this section if the data is already in Drive.

Output layout (see `docs/dataset_research.md`):
    <RAW_DIR>/train/{real,fake}          training pool
    <RAW_DIR>/test/{real,fake}           official held-out test
    <RAW_DIR>/cross/<gen>/{real,fake}    cross-generator sets
"""
        ),
        code(
            """\
import os
venv = "/content/fxt-venv"
repo = "/content/fauxtography"
os.chdir(repo)

RAW_DIR = "/content/drive/MyDrive/fauxtography/data/raw"   # where raw/ goes
os.environ["FAUXTOGRAPHY_RAW_DIR"] = RAW_DIR

# Skip the download if the training pool already exists in Drive.
train_real = os.path.join(RAW_DIR, "train", "real")
if os.path.isdir(train_real) and os.listdir(train_real):
    print("raw training pool already present - skipping download")
else:
    !{venv}/bin/fauxtography fetch-genimage
print("raw data at:", RAW_DIR)
""",
        ),
        md(
            """\
## 7. Split the dataset into train/val/test from the downloaded raw pool

Builds `data/train|val|test` from the GenImage SD v1.4 *training pool* only
(`raw/train`), keeping the official held-out `raw/test` separate for later
evaluation (train-baseline scores it when `data.heldout_test` is set).
"""
        ),
        code(
            """\
# NOTE: splits only raw/train - the official test stays untouched.
!/content/fxt-venv/bin/fauxtography prepare-data \\
    --source-real /content/drive/MyDrive/fauxtography/data/raw/train/real \\
    --source-fake /content/drive/MyDrive/fauxtography/data/raw/train/fake \\
    --data-root /content/drive/MyDrive/fauxtography/data
""",
        ),
        md(
            """\
## 8. Training / evaluation (GPU steps - run one at a time)

The three model commands are heavy; run them in separate cells so a disconnect
can't lose more than one step. Each reads its own config and writes results to
Drive via the env vars above.
"""
        ),
        code(
            """\
# Baseline - fine-tune the shared ResNet (joint)
!/content/fxt-venv/bin/fauxtography train-baseline
""",
        ),
        code(
            """\
# CLIP zero-shot - DemetraP20. NOT IMPLEMENTED YET.
# Uncomment (remove the leading #) once eval-clip is implemented.
# !/content/fxt-venv/bin/fauxtography eval-clip
print("SKIPPED: CLIP eval-clip not implemented yet (see docs/milestones).")
""",
        ),
        code(
            """\
# DINOv2 - extract features then train the linear probe - suprch4rg3d.
!/content/fxt-venv/bin/fauxtography extract-dinov2
!/content/fxt-venv/bin/fauxtography train-linear-probe
""",
        ),
        md(
            """\
## 9. Backup sanity

Confirm the outputs actually landed on Drive (they survive session drops).
Then open `00_data_overview.ipynb` and the model analysis notebooks to
inspect the results and write the verdicts.
"""
        ),
        code(
            """\
!ls /content/drive/MyDrive/fauxtography/results
!echo "---" && ls /content/drive/MyDrive/fauxtography/results/clip 2>/dev/null || echo "no clip results yet"
""",
        ),
        md(
            """\
## 10. Publish results to git

Colab runs on your branch - these cells copy the results from Drive into the
repo and **stage only `results/`** (the small committed artifacts: json, svg,
csv). Checkpoints and features are already git-ignored, so nothing heavy gets
staged.

Deliberately *not* automated: review what's staged, then commit and push
**from your own branch** and open a PR into `dev` (see `docs/work_division.md`).
No credentials are handled here - set up GitHub auth on the machine/terminal
that pushes (e.g. a PAT scoped to this repo).
"""
        ),
        code(
            """\
# Bring the results (on Drive) into the cloned repo so git sees them
import shutil
shutil.rmtree("/content/fauxtography/results", ignore_errors=True)
shutil.copytree("/content/drive/MyDrive/fauxtography/results", "/content/fauxtography/results")
# Stage only results/ (checkpoints + features stay out; they are gitignored)
!cd /content/fauxtography && git add results/
!cd /content/fauxtography && git status --short results/
print("review the staged list, then commit + push from your branch")
""",
        ),
        code(
            """\
# ---- do this on your branch / terminal, NOT automatically here ----
#   cd /content/fauxtography
#   git add results/
#   git commit -m "Add <model> results"
#   git push origin <your-branch>
#   # then open a PR: <your-branch> -> dev
print("manual step - see cells above")
""",
        ),
    ]


def main() -> None:
    write("00_data_overview.ipynb", data_overview())
    write("analysis_clip.ipynb", model_analysis("clip", "CLIP zero-shot", "DemetraP20"))
    write(
        "analysis_dinov2.ipynb", model_analysis("dinov2", "DINOv2 + linear probe", "suprch4rg3d")
    )
    write("final_report.ipynb", final_report())
    write("fauxtography-on-colab.ipynb", fauxtography_on_colab())


if __name__ == "__main__":
    main()
