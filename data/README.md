# Data

This directory is **git-ignored**; raw images and prepared splits live only on
your machines (or a shared drive). `fauxtography prepare-data` rebuilds the
split layout from raw folders, so teammates clone the repo and re-run
preparation.

Expected layout after `fauxtography prepare-data --source-real ... --source-fake ...`:

```
data/
  train/real/...   train/fake/...
  val/real/...     val/fake/...
  test/real/...    test/fake/...
```

## Choosing the dataset
We use **GenImage** (see `docs/dataset_research.md`):

- **Real (nature):** ImageNet images.
- **AI-generated:** Stable Diffusion v1.4 subset for training, with other
  generators (Midjourney, ADM, BigGAN) held out for cross-generator analysis.

Current split counts (one re-run of `prepare-data`, committed results are
comparable because the split is seed-fixed):

```
data/train/   real 7,000 | fake 7,000
data/val/     real 1,500 | fake 1,500
data/test/    real 1,500 | fake 1,500

data/raw/train/           real 10,000 | fake 10,000   (SD v1.4 pool)
data/raw/test/            real 6,000  | fake 6,000    (official held-out)
data/raw/cross/<gen>/     real 1,000  | fake 1,000    (Midjourney, ADM, BigGAN)
```