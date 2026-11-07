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
Document here where real photographs and AI-generated images come from, e.g.:

- **Real:** ... (dataset name + count + license)
- **AI-generated:** ... (generator/model used + count)

Keep versioned counts up to date so results stay comparable between teammates.