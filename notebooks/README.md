# Notebooks — results & verdicts

The **analysis layer** of the project lives here. These are the notebooks you
open in Google Colab (or locally) to inspect results, document the manual
error inspection, and write the verdicts that go on the presentation slides.

| Notebook | Purpose | Owner |
|----------|---------|-------|
| `setup_colab.ipynb` | **run first** in Colab: env, Drive, data split, training | shared |
| `00_data_overview.ipynb` | confirm splits + class balance, sample grid | shared |
| `analysis_clip.ipynb` | CLIP metrics, confusion matrix, error inspection, verdict | Collaborator A |
| `analysis_dinov2.ipynb` | DINOv2 probe metrics, errors, verdict | Collaborator B |
| `final_report.ipynb` | side-by-side comparison + joint conclusions | both |

## Running in Colab (with a GPU)

1. **Open `setup_colab.ipynb` first** and run its cells top-to-bottom: it
   installs a Python 3.14 env with `uv`, installs the package, verifies the
   accelerator with `fauxtography env`, splits the dataset, and runs the model
   commands. Outputs go to your Drive via `FAUXTOGRAPHY_*` env vars, so they
   survive Colab session disconnects.
2. Then open `00_data_overview.ipynb` and the per-model analysis notebooks.
   Point `DATA_ROOT` / `results` at your Drive path if they don't already
   match the env vars.
3. Run the analysis notebooks top-to-bottom. Each is self-contained and will
   print a friendly placeholder until its `results/<model>/metrics.json`
   exists.

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