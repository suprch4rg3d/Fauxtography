# Contribution and novelty

This note states, in one place, what this project contributes beyond
reproducing the GenImage paper. The final report and the analysis notebooks
should reuse the framing here whenever they claim novelty.

Em-dashes are banned in this repo, so the argument below uses hyphens and
plain words only.

## The honesty rule

The cross-generator experiment itself is exactly what the GenImage paper
did: train a detector on Stable Diffusion v1.4 fakes, then test it on fakes
from generators it never saw. We must never claim the experiment is new.

What is new is what we change and what we learn. The paper's published
result is its Table (ResNet-50, trained on SD v1.4):

| Tested on     | Paper ResNet-50 |
|---------------|-----------------|
| SD v1.4 (same)| ~99.9%          |
| Midjourney    | ~54.9%          |
| ADM           | ~53.5%          |
| BigGAN        | ~52.0%          |

Everything below extends that table; nothing replaces it.

## Contribution 1 - the model axis

The GenImage paper evaluates fine-tuned CNNs (ResNet-50) and nothing else.
Its central claim - "training on one generator collapses on others" - is
therefore a claim about one model class.

We run the identical cross-generator protocol on two very different model
families using the same GenImage test sets:

- **DINOv2 + linear probe**: frozen self-supervised ViT features. These
  learned *natural-image structure*, not generator artifacts. If the paper's
  collapse is a fine-tuning artifact, DINOv2 should retain accuracy across
  generators. If DINOv2 collapses too, the paper's finding is not about CNNs
  at all - it is intrinsic to fake detection.
- **CLIP zero-shot**: never trained on any fake image, so it has no single
  generator's artifacts to latch onto. The paper had no prior-free model to
  compare.

That extends the paper's question from "does a CNN generalise?" to "is
cross-generator collapse about the generator, the model, or the task?"

## Contribution 2 - the comparison axis

Because we re-use GenImage's SD v1.4 test sets, our numbers line up
column-by-column with the paper's published ResNet-50 row. The intended
comparison table (values TBD until the cross-generator runs land):

| Tested on | Paper ResNet-50 | Our ResNet | DINOv2 | CLIP |
|-----------|-----------------|------------|--------|------|
| SD v1.4   | ~99.9%          | ~97%       | ~88.7% | ~67% |
| Midjourney| ~54.9%          | TBD        | TBD    | TBD  |
| ADM       | ~53.5%          | TBD        | TBD    | TBD  |
| BigGAN    | ~52.0%          | TBD        | TBD    | TBD  |

If CLIP results are not available or not merged, publish the three-column
version (paper / ResNet / DINOv2) and keep CLIP's row as "work in progress"
or drop it - do not fabricate or over-draw it.

Caveat to state honestly: our same-generator numbers are lower than the
paper's 99.9% because we train on a much smaller slice (~7k fake images per
class vs the paper's full generator pool). Cross-generator comparisons in
the table are still informative, but the same-generator row is not a
like-for-like replication.

## Contribution 3 - the interpretation axis

The paper reports numbers; it does not show why misclassifications happen.
The project's manual error inspection (which real images fool the
SD-trained DINOv2 probe, which fake styles are most confusable, how the
confusion cells differ per generator) turns the accuracy collapse into a
story. That is analysis depth a report can add where the paper stops.

## How to read the cross-generator result

Two defensible outcomes, both contributions:

1. **DINOv2 stays high across generators**: foundation-model features
   generalise; fine-tuned CNNs memorise the generator's look. Supports
   representation-learning for detection.
2. **DINOv2 collapses like ResNet**: the collapse replicates for
   self-supervised features, so it is harder to fix than the paper implied -
   a negative result with real weight for detection practice.

Either way the report must say clearly *what the paper knew* and *what we
added*, never implying the experiment itself is novel.

## Contribution 4 - cross-model fusion and disagreement

We produce predictions from three model families (supervised CNN, frozen
self-supervised ViT + probe, zero-shot CLIP) on the same sets. Majority-vote
fusion and the disagreement map show whether the families are redundant or
complementary. The finding is *which pair of families cooperates* and where
the zero-shot model's priors split from the trained models. Command:
`fauxtography analyze-fusion`.

## Contribution 5 - is the compression short-cut causal?

The file-size probe (corpus stats, 112 KB real vs 447 KB fake) is only a
correlation. The recompress-and-eval probe re-encodes both classes to a
similar file size, re-extracts frozen DINOv2 features and re-scores the
unchanged probe (`fauxtography probe-compression`). If accuracy collapses,
the detector was reading the size gap, not the pixels - a causal statement
about the GenImage short-cut, not just an observation.

## Contribution 6 - feature geometry

DINOv2 features projected with PCA->TSNE and colored by generator
(`fauxtography visualize-features`). Shows whether the cross-generator
collapse is visible in the representation itself (generators form separate
clusters even before any head is trained).

## What is NOT a contribution (keep out of claims)

- Running the cross-generator experiment on GenImage: reproduction.
- Same-generator accuracy: reproduction at smaller scale.
- The exact accuracies: numbers, not findings. The finding is the *relative
  behaviour across models and generators*.
- File-size correlations without the recompress experiment: observation, not
  cause. Only cite the causal probe as a contribution.