# Experiments

Runtime artifacts live here (checkpoints, cached DINOv2 features, runs):
everything under `experiments/` is git-ignored and regenerable via the scripts.

- `runs/baseline/baseline.pt`       — fine-tuned ResNet checkpoint
- `runs/dinov2/features/*.pt`       — cached DINOv2 features
- `runs/dinov2/probe.pt`            — linear probe weights

Committed outputs that *are* shared live under `results/` (metrics figures,
confusion matrices, comparison table).