# Notebooks — results & verdicts

The **analysis layer** of the project lives here. These are the notebooks you
open in Google Colab (or locally) to inspect results, document the manual
error inspection, and write the verdicts that go on the presentation slides.

| Notebook | Purpose | Owner |
|----------|---------|-------|
| `00_data_overview.ipynb` | confirm splits + class balance, sample grid | shared |
| `analysis_clip.ipynb` | CLIP metrics, confusion matrix, error inspection, verdict | Collaborator A |
| `analysis_dinov2.ipynb` | DINOv2 probe metrics, errors, verdict | Collaborator B |
| `final_report.ipynb` | side-by-side comparison + joint conclusions | both |

## Running in Colab (with a GPU)

1. Mount this repo (clone or upload), then in a code cell:

   ```python
   # bootstrap Python 3.14 env with uv, then install the package
   !pip install -q uv && uv venv --python 3.14 /content/fxt-venv
   /content/fxt-venv/bin/pip install -e .
   ```

   (Or use `requirements-dev.txt`. See `docs/setup.md` for both paths.)

2. Point `DATA_ROOT` in `00_data_overview.ipynb` at your dataset
   (a Colab Drive mount or a folder you upload).
3. Run the notebooks top-to-bottom. Each is self-contained and will print a
   friendly placeholder until its `results/<model>/metrics.json` exists.

## Regenerating the skeletons

Run `python scripts/make_notebooks.py` from the repo root to rebuild the
notebooks from templates (e.g. after changing the template structure).
**Don't edit the generated JSON by hand** — edit the templates and re-run.

## Conventions

- Commit the *notebooks*, not their outputs (`results/`, model checkpoints and
  raw data stay git-ignored). Clear outputs before committing to keep diffs
  reviewable, or rely on the fact that notebooks read results from disk.
- Verdicts: one sentence per finding; the fuller write-up lives in
  `docs/analysis_<model>.md`.