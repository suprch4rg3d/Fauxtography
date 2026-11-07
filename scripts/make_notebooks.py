#!/usr/bin/env python3
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
# 00 — Data Overview

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
# Analysis — {title}

**Owner:** {owner}   |   **Model dir:** `results/{name}/`

What this notebook adds: the numbers behind the verdict. It loads the saved
metrics + predictions produced by the eval script, redraws the confusion
matrix for this model, and documents the *manual error inspection* — the
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
    print("No metrics yet — run `fauxtography eval-clip` (for CLIP) first.")
"""
        ),
        md("## 2. Confusion matrix"),
        code(
            f"""\
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
            f"""\
# Load a few misclassified images and inspect what confused the model.
# Fill in: what visual artifacts / styles fooled the model? (presentation gold.)
# NOTE: to see the actual images, load the test set with fauxtography
# (FauxtoDataset) and reconstruct the misclassified file paths via the
# dataset's .samples attribute. Fill in your observations below.
import numpy as np
if y_true is not None:
    errors = np.where(y_true != y_pred)[0]
    print(f"{{len(errors)}} misclassified (of {{len(y_true)}}): {{len(errors)/len(y_true):.1%}}")
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
print("TODO: verdict for this model — see docs/analysis_<model>.md for the write-up")
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
# Final Report — comparing baseline / CLIP / DINOv2

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
    print("No results yet — run baseline, eval-clip, and the DINOv2 probe first.")
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


def main() -> None:
    write("00_data_overview.ipynb", data_overview())
    write("analysis_clip.ipynb",
          model_analysis("clip", "CLIP zero-shot", "Collaborator A"))
    write("analysis_dinov2.ipynb",
          model_analysis("dinov2", "DINOv2 + linear probe", "Collaborator B"))
    write("final_report.ipynb", final_report())


if __name__ == "__main__":
    main()