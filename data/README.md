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

Update the counts below when the subsets are final so results stay comparable
between teammates.