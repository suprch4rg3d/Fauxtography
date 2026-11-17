# Experiments

Runtime artifacts live here (checkpoints, cached DINOv2 features, recompressed
mirrors). Everything under `experiments/` is **git-ignored** - these are large
files that are regenerable from `data/`, and uploading them to GitHub (e.g. via
LFS) is unnecessary. They live only on the machine/Drive that ran them.

Typical layout after the Colab run (paths follow `run_dir`; the colab env
points it at `Drive/fauxtography/experiments`):

```
experiments/
  baseline/baseline_best.pt            fine-tuned ResNet checkpoint
  dinov2/features/canonical/{train,val,test,heldout,cross_*}.pt
                                       cached DINOv2 features (variant-keyed)
  dinov2/probe.pt                      linear probe weights
  dinov2/recompressed/test/            re-compressed mirror for probe-compression
```

Committed outputs that *are* shared live under `results/` (metrics JSONs,
confusion matrices, comparison table, figures).