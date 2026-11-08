# Dataset Research

**MAI644 Computer Vision - Semester Group Project (Fauxtography)**

This note documents the dataset selection for the project: detecting AI-generated
images vs real photographs, comparing a ResNet baseline, CLIP zero-shot and
DINOv2 with a linear probe.

**Verdict:** use **GenImage** - specifically the **Stable Diffusion v1.4 subset**
for the main train/val/test split, plus a **cross-generator test set** from 2-3
other generators (Midjourney, ADM, BigGAN) as an analysis extension.

---

## Selection criteria

The analysis strategy (see `technical_plan.md`) is built around **manual visual
error inspection** - documenting the artifacts and image styles that fool each
model. That is the core presentation material, so the dataset choice was driven
by how well it supports that story, alongside standard practical concerns:

1. **Resolution** - high enough that detection artifacts are visible and that
   CLIP / DINOv2 (pretrained at 224x224 inputs) can do their job.
2. **Reference numbers** - the dataset should have published baselines that our
   ResNet can be fairly compared against.
3. **Diversity of generators** - a chance to show that detection transfers (or
   fails to transfer) across generation methods, which is the interesting
   "understanding the method" analysis.
4. **License** - permissive enough for a university project.
5. **Access friction** - downloadable without accounts/scraping, practical size
   for a Colab GPU free tier.

## Candidate comparison

### CIFAKE - real vs AI-generated (Stable Diffusion v1.4)

- 60k real + 60k fake images; real side from CIFAR-10, fake side generated
  with Stable Diffusion v1.4.
- **32x32 resolution.** Pretrained CLIP/DINOv2 expect larger inputs (224x224),
  so foundation-model approaches would run on upscaled thumbnails, and the
  small canvas makes "visual artifact" inspection hard to appreciate.
- MIT license (inherits CIFAR-10's); one-click download, ~110 MB.
- Published baseline numbers exist (standard CNNs ~95-98% accuracy).

**Assessment:** very accessible and a reasonable sanity check, but the 32x32
resolution undercuts both foundation-model approaches and the visual
error-inspection objective. Not chosen as the main dataset.

### GenImage - million-scale benchmark (Huawei Noah's Ark Lab)

- 2.68M images total: 1.33M real (ImageNet) + 1.35M fake, across **eight
  generators**: Midjourney, Stable Diffusion v1.4/v1.5, ADM, GLIDE, Wukong,
  VQDM, BigGAN.
- **Resolution 256-1024 (mostly 512x512),** so artifacts are visible and
  CLIP/DINOv2 operate at their intended input scale.
- Published ResNet-50 baselines in the paper: ~99.9% accuracy when trained and
  tested on the same generator (SD v1.4), ~72% average when trained on SD v1.4
  and tested across all eight generators. These numbers are directly comparable
  to our baseline.
- The paper's central finding - that a detector trained on one generator fails
  to generalize to others - is exactly the kind of analysis we want to reproduce
  and discuss.
- Access: Google Drive mirror; per-generator/split subsets available via a
  Hugging Face Arrow mirror (SD v1.4 subset on the order of ~18 GB).
- License: CC BY-NC-SA 4.0 (non-commercial, fine for a course project); a
  smaller curated multi-generator subset is available under Apache-2.0.

**Assessment:** the strongest match for every selection criterion. The SD v1.4
subset gives a clean large train/val/test split, and holding out other
generators as a cross-generator test set provides a genuine analysis extension.

## Cross-generator evaluation (and why it matters for this project)

The real photographs in GenImage are paired with fake images from **eight
different generators** (Midjourney, Stable Diffusion v1.4/v1.5, ADM, GLIDE,
Wukong, VQDM, BigGAN). This lets us evaluate a detector in two very different
ways:

- **Same-generator:** train on Stable Diffusion fakes, test on Stable
  Diffusion fakes. The GenImage paper reports ~99.9% accuracy here.
- **Cross-generator:** train on Stable Diffusion fakes, then test on fakes
  from generators never seen during training (e.g. Midjourney, BigGAN, ADM).
  The same detector drops to ~72% on average and below ~55% on some
  individual generators.

Why the accuracy collapses: a detector trained on one generator tends to learn
that generator's specific artifacts (noise pattern, upsampling signature)
rather than a general "this is AI-synthesised" concept. When it meets a
different generator's artifacts, its crutch is gone. This is the paper's
central finding, and it is a valuable thing for us to reproduce:

- It separates **memorising one generator** from **understanding synthetic
  imagery**.
- It gives the manual error inspection a rich set of misclassifications to
dissect for the analysis and presentation.
- **CLIP is the interesting wildcard**: it was never trained on any fake
  images, so it has no single generator's artifacts to latch onto. Seeing
  whether CLIP generalises more evenly across generators than the fine-tuned
  ResNet is a genuinely insightful comparison.

Concretely, our plan is to train the baseline ResNet and DINOv2 probe on the
**Stable Diffusion v1.4** subset, then evaluate twice:

1. **Main test:** the official SD v1.4 test split - the clean in-domain number.
2. **Cross-generator test:** a few hundred to ~1,000 images each from
   Midjourney, ADM and BigGAN fakes - measuring how much accuracy drops
   (with real images held consistent, since the real class stays the same).

### Lightweight Hugging Face "AI vs Real" datasets

- Several small (1k-10k sample) AI-vs-real classification sets exist on the Hub
  with MIT/CC-BY licenses.
- Fine for a demo or a smoke test, too small for a course project that needs
  enough data to train a baseline and produce robust confusion matrices.

**Assessment:** not chosen.

---

## Plan with GenImage

**Main split (for all three models):**
- Real images: ImageNet "nature" subset.
- Fake images: the Stable Diffusion v1.4 generated subset.
- Prepared into `train/val/test` under `real/` and `fake/` folders via
  `prepare-data`, mirroring the canonical `data/` layout.

**Analysis extension (per-model error inspection):**
- Hold out small evaluation sets from **Midjourney**, **ADM** and **BigGAN**
  generators (a few hundred to a couple thousand images each).
- Evaluate our trained baseline and DINOv2 probe on these held-out generator
  sets to measure cross-generator transfer, and inspect where they fail.

**Expected reference points (from the GenImage paper, ResNet-50):**
- Same-generator test (SD v1.4): ~99.9% accuracy.
- Cross-generator average (trained SD v1.4, tested on all others): ~72%.

---

## Handling notes

- **Labels:** some mirrors label `0=Fake, 1=Real`, others the reverse. Our data
  layer treats folder names (`real/` / `fake/`) as the source of truth and maps
  them internally to `real=0, fake=1`, so download-script label flips cannot
  corrupt the experiments.
- **Size/Colab:** the SD v1.4 subset (~18 GB) fits on a free-tier Colab GPU.
  A lighter option to iterate faster is the curated multi-generator subset
  (~10k images, Apache-2.0) before committing to the full SD v1.4 training set.
- **License note:** GenImage is CC BY-NC-SA 4.0 - fine for a university project;
  include the citation in the report.

## References

- GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image.
  Zhu, Chen, Yan, Huang, Lin, Li, Tu, Hu, Hu, Wang (Huawei Noah's Ark Lab).
  NeurIPS 2023. arXiv:2306.08571.
- Project homepage / GitHub: `github.com/GenImage-Dataset/GenImage`
  (Google Drive download; per-generator Arrow mirror on Hugging Face).
- Related bias analysis: "Fake or JPEG? Revealing Common Biases in Generated
  Image Detection Datasets" (UnbiasedGenImage) - useful context on shortcut
  learning (compression / file size) that can inflate in-dataset accuracy.