# Multi-view Pig Posture Recognition: Public LB 4th Place Solution

## Overview

The core of this solution is a DINOv2 Giant (ViT-G/14) trained from its default pretrained weights for 5-class pig posture classification. Predictions are made at the individual pig instance level: each bounding box is cropped from the full image, padded, and resized to 224×224.

Public score: **0.878** (T2), 4th place.

## Model

`vit_giant_patch14_reg4_dinov2.lvd142m` via timm. I tried several architectures early on (SwinV2, ConvNeXt, DINOv2 variants) and DINOv2 outperformed everything else by a clear margin. Scaling up from DINOv2-S to B to L to G gave consistent gains each step, so I committed to the giant variant for all serious experiments.

## Training

**K-Fold CV (3-fold).** The split is done at the image level, not the row level, so pigs from the same frame always end up in the same fold. Stratification is based on camera type (orb/tur) + dominant class per image. Test predictions are the weighted average of all three fold models, weighted by each fold's best validation F1. OOF macro F1 came out to **0.9508**.

Fold best val F1s: 0.9479 / 0.9486 / 0.9406

**Layer-wise LR Decay (LLRD).** The classification head gets the full learning rate (5e-5) and each transformer block gets 0.85× the rate of the block above it. This made a meaningful difference compared to a flat learning rate.

**Loss.** A custom `PairwisePenaltyLoss` wrapping Focal Loss (γ=2.5). Confusion matrix analysis showed that the hardest confusions were Sternal vs. Lateral (54+47 errors) and Sternal vs. Sitting. Penalty weights were set manually based on error counts to push the model harder on those pairs.

**Augmentation.** Class-specific augmentation policies rather than one-size-fits-all. Sternal_lying gets aggressive rotation (up to 180°) since orientation doesn't matter for that posture. Lateral_left/right get restricted rotation to preserve the directional cue. Sitting gets heavy perspective distortion because it's a small projection under overhead cameras. MixUp (α=0.2) is applied on top.

Other standard ingredients: WeightedRandomSampler (boosting Sternal and Lateral), Model EMA (decay=0.9999), AMP, cosine LR schedule with 3-epoch warmup, early stopping (patience=8).

**TTA (8×).** 4 rotations × 2 flips. Left/Right labels are swapped appropriately for horizontally flipped passes.

## What Didn't Work

Non-DINOv2 architectures: SwinV2 and ConvNeXt fell significantly short of DINOv2, even when comparing similarly sized variants. The gap was large enough that I stopped pursuing them early.

Pseudo-labeling: I tried a two-stage approach where confident test predictions were added back as training data. It hurt performance and was removed entirely.

Every other addition was evaluated incrementally, and each contributed a measurable improvement to the validation F1: LLRD, class-specific augmentation, the custom penalty loss, and TTA were all additive.

## What I'd Do Differently

I intentionally left K-Fold to the end, the plan was to finalize the model first and then run K-Fold on the best version since DINOv2-G training is slow. I got too focused on improving the model and by the time I remembered, there was little time left. The peak single-model score was 0.867 at epoch 28, and even with only 3 folds at 22 epochs due to the time constraint, K-Fold pushed the score up to 0.878. With 4-5 folds at the full 28 epochs the result would likely be higher.

## Hardware

NVIDIA A100-SXM4-40GB (42.4 GB VRAM).

## Running the Code

The dataset should be placed at:

```
DATA_ROOT  = Path("/root/.cache/kagglehub/competitions/multi-view-pig-posture-recognition/multiview_pig_posture_recognition")
```

This path is set via `DATA_ROOT` in the script.

For inference, 3 folds pretrained weights are loaded from my Kaggle dataset:

```
OUT_DIR    = Path("/root/.cache/kagglehub/datasets/ramazanturann/pig-posture-prediction-dinov2g-3fold/versions/1")
```

This path is set via `OUT_DIR` when running in inference mode. Submission files are also saved to this directory. In training notebook, i set `OUT_DIR` to `outputs`.

**train notebook and inference notebook is same, only difference is inference_only parameter set to True.**

## Environment

Google Colab PRO, A100 40GB. Train code runs with 38GB VRAM, Inference requires lower VRAM.
