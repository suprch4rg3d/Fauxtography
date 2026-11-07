# Work Division

Two collaborators, one model each plus its written analysis. Joint analysis is
also split so both contribute to the final report and presentation.

## Ownership map

| Area | Owner | Files |
|------|-------|-------|
| Data prep & split infrastructure | **shared** | `scripts/prepare_data.py`, `src/fauxtography/data/` |
| Baseline ResNet fine-tune | **shared** (reference point) | `src/fauxtography/models/baseline.py`, `scripts/train_baseline.py` |
| CLIP zero-shot **+ its analysis** | **Collaborator A** | `src/fauxtography/models/clip.py`, `scripts/eval_clip.py`, `Docs/analysis_clip.md` |
| DINOv2 + linear probe **+ its analysis** | **Collaborator B** | `src/fauxtography/models/dinov2.py`, `models/linear_probe.py`, `scripts/extract_dinov2.py`, `scripts/train_linear_probe.py`, `Docs/analysis_dinov2.md` |
| Joint comparison & presentation | **split / joint** | `scripts/compare_models.py`, `Docs/final_report.md`, slides |

## How the model + analysis pairing works

Each collaborator:

1. Implements their model code on their own branch (from `dev`).
2. Runs their eval script → commits results under `results/<model>/`.
3. Writes `Docs/analysis_<model>.md`: confusion matrix reading, the **manual
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