# Work Division

**Semester Group Project — MAI644 Computer Vision.**

Two collaborators, one model each plus its written analysis. Joint analysis is
also split so both contribute to the final report and presentation.

## Ownership map

| Area | Owner | Files |
|------|-------|-------|
| Data prep & split infrastructure | **shared** | `prepare-data` CLI + `src/fauxtography/data/` |
| Baseline ResNet fine-tune | **shared** (reference point) | `train-baseline` CLI, `src/fauxtography/models/baseline.py` |
| CLIP zero-shot **+ its analysis** | **Collaborator A** | `eval-clip` CLI, `src/fauxtography/models/clip.py`, `notebooks/analysis_clip.ipynb`, `docs/analysis_clip.md` |
| DINOv2 + linear probe **+ its analysis** | **Collaborator B** | `extract-dinov2` + `train-linear-probe` CLIs, `models/dinov2.py`, `models/linear_probe.py`, `notebooks/analysis_dinov2.ipynb`, `docs/analysis_dinov2.md` |
| Joint comparison & presentation | **split / joint** | `compare` CLI, `notebooks/final_report.ipynb`, `docs/final_report.md`, slides |

## How the model + analysis pairing works

Each collaborator:

1. Implements their model code on their own branch (from `dev`).
2. Runs their eval command → commits results under `results/<model>/`.
3. Renders the verdict in `notebooks/analysis_<model>.ipynb` and writes the
   `docs/analysis_<model>.md` write-up: confusion matrix reading, the **manual
   visual error inspection** (which images fool the model, what visual
   artifacts), plus the method-design decisions they made.

## Joint analysis (split, not duplicated)

| Section | Primary writer |
|---------|----------------|
| Dataset description & baseline | collaborator with the shared part |
| CLIP results interpretation | Collaborator A |
| DINOv2 results interpretation | Collaborator B |
| Side-by-side comparison & conclusion | both (A writes first half, B second) |
| Presentation slides | both |

## Git flow

- Everyone works on their **own branch from `dev`**.
- PR into `dev` per milestone/feature (never straight to `main`).
- `main` is the final deliverable, merged from `dev` only.

## Milestones (proposed)

1. **M0 — data**: acquire dataset, run `prepare_data.py`, commit split counts.
2. **M1 — baseline**: ResNet fine-tune, report loss/acc curves (shared).
3. **M2/M3 — models in parallel** (each owner): CLIP zero-shot, DINOv2 probe.
4. **M4 — analysis**: per-owner error inspection docs.
5. **M5 — comparison + report + presentation** (joint).