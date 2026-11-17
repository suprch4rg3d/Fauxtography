# Contributing

**Fauxtography** is a semester group project (MAI644 - Computer Vision,
M.Sc. in Artificial Intelligence, University of Cyprus, 2026) that
compares approaches for detecting AI-generated images. The project's wider
aim is a **reference point**: a repo where, for each foundation model,
anyone can reproduce the numbers, rerun the eval, and add or swap models
without touching the comparison machinery.

For the full framing and the step-by-step pattern for adding a model, see
the README section [**Adding another model (open to contributors)**]
(https://github.com/suprch4rg3d/fauxtography/blob/main/README.md#contributing).

## Two ways to contribute

1. **Rerun the repo** and report your numbers. The dataset splits and
   seeds are fixed and results are committed, so a fresh run should match
   the table; a differing run is a valuable bug report.
2. **Add a model** - an existing or a newer foundation model - following
   the one-file-at-a-time pattern in the README:
   1. `configs/<model>.yaml` - model settings (copy `configs/dinov2.yaml`).
   2. `src/fauxtography/models/<model>.py` - a wrapper exposing the model
      (copy `models/dinov2.py`; labels are `0=real, 1=fake`).
   3. A CLI command in `cli.py` that writes `results/<model>/`. `compare`
      and `analyze-fusion` pick it up automatically.
   4. `notebooks/analysis_<model>.ipynb` - the verdict, written from the
      committed results.

## What makes a good contribution

- Results you can reproduce, with the numbers committed under `results/`.
- An analysis notebook (see `notebooks/analysis_dinov2.ipynb`) that reads
  those results and states the verdict plainly.
- No fabricated numbers. If CLIP (or any model) is not merged yet, its
  column stays as a TODO - see `docs/contributions.md` for the novelty
  framing and what is not a contribution.

## Git workflow

- Personal branches from `dev` -> PR into `dev` (integration buffer).
- `dev` -> PR into `main` (final deliverable).
- See `docs/work_division.md` for how a model and its analysis are owned.