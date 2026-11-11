# DINOv2 Implementation Notes

**Ownership: suprch4rg3d.**

This note explains what DINOv2 is, why we use it, and how the codebase wires it
up. Written for a teammate who wants to understand the milestone without
reading every line of code.

## 1. What is DINOv2?

DINOv2 is a **vision foundation model** from Meta AI (FAIR). It is a ViT
(Vision Transformer) trained on a huge collection of unlabeled images using a
**self-supervised** method called DINO (self-DIstillation with NO labels).

The key idea: instead of learning to predict a class from labeled photos,
DINOv2 learns **visual representations** from the images themselves. The model
is shown a crop of an image and a differently-augmented view of the same
image, and is trained to produce *similar* features for the two views. After
this training, the features it extracts capture rich, general visual
information (edges, textures, object parts, composition) without ever seeing a
label.

Why that matters for us: those features transfer well to new tasks. We can
freeze DINOv2, extract a feature vector per image, and train a *tiny* linear
classifier on top. Compression artifacts, lighting, texture and watermark
patterns are all represented in the features, so a real-vs-fake classifier
built on them is both strong and cheap to train.

## 2. The three model families we compare

| Model | Idea | Training we do |
|-------|------|----------------|
| ResNet baseline | Classic CNN, sees labels during fine-tuning | Full fine-tune (shared) |
| CLIP | Contrasts images with text captions | None (zero-shot) |
| **DINOv2** | Self-supervised vision features | **Linear head only** |

DINOv2 sits between the two: much stronger features than a raw CNN, but no
text modality, so we add a small trainable head instead of using text prompts.

## 3. How the pipeline works in this repo

Two commands, run in order:

```
poetry run fauxtography extract-dinov2
poetry run fauxtography train-linear-probe
```

### Step 1: `extract-dinov2` (cache frozen features)

- Loads the frozen DINOv2 backbone (HuggingFace `Dinov2Model`).
- Runs every image in train/val/test (and the held-out test if configured)
  through the backbone **in eval mode, no gradient**.
- Keeps only the **`[CLS]` token** - the special first-token that aggregates
  the whole image into one vector. For the `vitb14` arch that is a **768-dim**
  vector per image.
- Saves the features per split as `.pt` files under
  `<run_dir>/dinov2/features/{canonical,square}/{train,val,test,heldout}.pt`
  (the subfolder is the preprocessing variant, so square vs canonical runs
  never reuse each other's cache), alongside a `metadata.json` (arch, image
  size, variant).

This step is the expensive one (one full forward pass per image), so the
results are **cached**. Re-running skips splits that already have a cache
file.

### Step 2: `train-linear-probe` (train the head, evaluate)

- Loads the cached features (no images, no backbone).
- Trains a **single `LinearProbe`** - literally one `nn.Linear(768, 2)` matrix
  per class - with AdamW + cosine LR for a few epochs.
- Tracks **best validation accuracy** and restores those weights at the end.
- Evaluates on the internal test split and the held-out GenImage test.
- Writes outputs in the **same layout as baseline**:
  `results/dinov2/metrics.json`, `heldout_metrics.json`, confusion matrices,
  and `y_true.npy` / `y_pred.npy`.

## 4. Design decisions (and why)

| Decision | Choice | Reason |
|----------|--------|--------|
| Backbone source | HuggingFace `transformers`, not torchvision | The pinned `torchvision==0.29.2` build ships **no DINOv2** weights (verified via `list_models()`). `transformers` is already a dependency and offers the exact Meta checkpoints |
| Arch names | `dinov2_vits14` / `vitb14` / `vitl14` | Mirror torchvision naming; config picks one (`vitb14` default = balanced) |
| Backbone frozen? | Yes, 100% | That is the whole point of a foundation model - features transfer, so we only train the head. Faster + far fewer trainable params |
| Train vs eval transform | **Identical** (official DINOv2 preproc: resize short side to 256 + center-crop 224, ImageNet-normalize, no augmentation; `square_resize: true` opts back into plain square resize) | A frozen backbone needs *consistent* feature geometry, and official DINOv2 preproc avoids the aspect-ratio distortion of square stretching |
| Head | Single linear layer | Baseline for linear-probe evaluation, cheap, and readable |
| Feature cache | `.pt` per split, skip-if-present | Extraction is the expensive step; the head can be re-tuned without re-running the backbone |

## 5. Key files

| File | Role |
|------|------|
| `src/fauxtography/models/dinov2.py` | `DINOv2FeatureExtractor`: loads frozen backbone, `extract()` -> `[CLS]` features |
| `src/fauxtography/models/linear_probe.py` | `LinearProbe`: the single linear head |
| `src/fauxtography/data/transforms.py` | `dinov2_transforms`: resize + normalize, shared train/eval |
| `src/fauxtography/cli.py` | `extract-dinov2` and `train-linear-probe` commands |
| `configs/dinov2.yaml` | All knobs: arch, batch sizes, probe lr/epochs, data roots |

## 6. Config knobs (`configs/dinov2.yaml`)

- `arch`: `dinov2_vits14` (small/fast) | `dinov2_vitb14` (default) | `dinov2_vitl14` (big/strong)
- `feature_batch_size`: images per forward pass during extraction
- `probe_batch_size`, `probe_epochs`, `probe_lr`, `probe_weight_decay`: linear-head training
- `data.root`, `data.cross_dir`, `output.run_dir`, `output.results_dir`: standard paths
  (override via the `FAUXTOGRAPHY_*` env vars)

## 7. Cross-generator analysis (`eval-cross`)

### The question

The DINOv2 probe (and the baseline) is trained on fakes from **one** generator:
Stable Diffusion v1.4. Does that knowledge survive fakes from a *different*
generator - Midjourney, ADM or BigGAN? A detector that only recognizes "the
one AI look it was trained on" is not really a fake detector; it is a
Stable-Diffusion detector.

The GenImage paper shows this is a real trap for fine-tuned CNNs: with
ResNet-50 trained on SD v1.4, same-generator accuracy is ~99.9% but the
same detector drops to ~54.9% on Midjourney, ~53.5% on ADM and ~52.0% on
BigGAN (see `docs/dataset_research.md`). Barely above coin-flip.

### What `eval-cross` does

Scores the **already-trained** probe (no retraining - that would defeat the
point) on the cross-generator sets under `data/raw/cross/<generator>/`
(`real/` + `fake/`, 1000 each). It:

1. Loads the probe checkpoint written by `train-linear-probe`.
2. Extracts features for each generator with the **same** frozen backbone
   and the **same** eval transform as the main run (variant-keyed feature
   cache, so `square_resize` changes never reuse wrong-transform features).
3. Writes per-generator metrics to `results/dinov2/cross/<generator>/`
   (`metrics.json`, `y_pred.npy`, `y_true.npy`, `confusion_matrix.svg`).

```
poetry run fauxtography eval-cross
```

Expected output table (SD v1.4-trained; ours upcoming):

| Tested on   | Paper ResNet-50 | Ours (DINOv2) |
|-------------|-----------------|---------------|
| SD v1.4     | ~99.9%          | ~88.7%        |
| Midjourney  | ~54.9%          | TBD           |
| ADM         | ~53.5%          | TBD           |
| BigGAN      | ~52.0%          | TBD           |

### How to read the result

- DINOv2 stays high across generators: foundation-model features generalize,
  CNNs memorize the generator's look.
- DINOv2 collapses like ResNet: the problem is harder than features; strong
  representations still need training signal.

Either is a defensible, interesting result for the final report.

## 8. References

- DINOv2 paper: "DINOv2: Learning Robust Visual Features without Supervision"
  (Oquab et al., 2023)
- The DINO self-supervised method it builds on: "Emerging Properties in
  Self-Supervised Vision Transformers" (Caron et al., 2021)
- HuggingFace checkpoints: `facebook/dinov2-base` (and `-small`, `-large`)