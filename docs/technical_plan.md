# Fauxtography - Detecting AI-Generated Images

**Semester group project for MAI644 - Computer Vision, M.Sc. in Artificial Intelligence, University of Cyprus, 2026.**

## Technical Plan

Build a classifier that tells whether an image is a real photograph or
AI-generated, comparing a fine-tuned baseline against two foundation
models on a shared test set.

1. **Dataset.** Collect real photographs plus AI-generated images, with
   clear labels, and split them into training, validation and testing
   sets.
2. **Baseline - ResNet.** Fine-tune a standard CNN (ResNet) on the
   training data. Track loss and accuracy during training to confirm it
   learns the basic differences between real and fake images.
3. **CLIP - zero-shot.** Feed test images to CLIP with text prompts asking
   whether each is a real photo or an AI generation, and record the
   predictions with no training.
4. **DINOv2 - linear probe.** DINOv2 is a vision model without a text
   component. Use it to extract deep visual features from the images and
   train a simple linear classifier on top for the final prediction.

## Analysis Strategy

Evaluate all three models on the test set and report overall accuracy for
each. Generate a confusion matrix per model to see how often a real photo
was flagged as fake and how often a fake photo slipped through as real.
Then do a manual visual inspection of the errors: look at the images where
each model failed and document the visual artifacts or image styles that
confused it. That inspection is the core of the presentation and shows the
methods are understood, not just run.