# Results

Committed, presentation-ready outputs - small files only. The notebooks and
the final report read these directly; nothing here is produced by hand.

Layout:

```
results/
  baseline/              train + heldout metrics, confusion matrices, y_pred/y_true
  baseline/cross/<gen>/  cross-generator scores (Midjourney, ADM, BigGAN)
  dinov2/                in-domain metrics + confusion matrices
  dinov2/cross/<gen>/    cross-generator scores
  dinov2/compression/    re-compression probe (file-size shortcut test)
  dinov2/embedding/      feature-space embedding (embedding.svg, classes, xy)
  dinov2_square/         archived square-resize run
  file_size/             file-size threshold probe + histogram
  fusion/                ensemble summary (summary.json/csv)
```

Produced by the CLI: `train-baseline`, `extract-dinov2`,
`train-linear-probe`, `eval-cross`, `probe-file-size`, `probe-compression`,
`visualize-features`, `analyze-fusion`, and `compare` (which writes
`comparison.csv`).