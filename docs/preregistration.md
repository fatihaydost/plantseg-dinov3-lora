> English translation of the original Turkish pre-registration (`preregistration.tr.md`), which was written before the first run and is the authoritative, time-stamped document. If the two differ, the Turkish file prevails.

# T1 PlantSeg × DINOv3: pre-registration (written before any run, 2026-10-10 ~21:40)

This file was written before the first run and is not modified afterwards. Deviations are in `deviations.md`.

## Data and split
- PlantSeg (Wei et al., Sci Data 2026, DOI 10.1038/s41597-025-06513-4; Zenodo 17719108, CC BY-NC 4.0).
- Official 70/10/20 split file (Metadata.csv). **Training = official train, dev = official validation (10%).** The test split (20%)
  is never loaded in any run and no choice is made by looking at it.
- Data hygiene (no model, no labels): near-duplicate counts train↔val and train↔test via pHash (image hashes only;
  the card's K/leakage item asks for this). This count enters no model or setting choice.
- Classes: 116 (background + 115 diseases), as in the dataset paper's code (`num_classes=116`); mIoU is computed over 116 classes
  (mmseg IoUMetric: a class that never appears in dev is excluded from the mean, nan is skipped).

## Metric
- Primary: dev mIoU (%). Secondary: mAcc. Evaluation: resize to short side 512 (dataset paper
  `Resize scale=(2048,512) keep_ratio`), whole-image inference, logits bilinearly upsampled to the original label resolution.
  No TTA.

## What is compared (all with the same preprocessing, head, number of repeats, and data)
- **Head (identical in all runs):** UperNet-lite = PPM + FPN, 256 channels, 4 input levels (1/4, 1/8, 1/16, 1/32).
  For the ViTs, 4 intermediate layers (ViT-B: blocks 3, 6, 9, 12) plus a simple feature pyramid (ViTDet style: 2× deconv, 1×,
  identity, maxpool) count as part of the head.
- **B0 baseline (small-scale reproduction):** ConvNeXt-T, ImageNet-22k pre-trained (timm `convnext_tiny.fb_in22k`),
  full fine-tuning. The closest variant to the dataset paper's ConvNeXt-L UperNet (46.24 mIoU on test, 40k iter × 16 batch, 512 crop)
  that fits a 4 GB GPU. The paper's numbers (46.24 / ConvNeXt-B 40.69 / Swin-B 38.67) are reported separately and are not compared directly
  with B0 (different split: dev vs test; different size; different schedule).
- **P1 proposal, simplest (H1 primary):** DINOv3 ViT-B/16 (timm `vit_base_patch16_dinov3.lvd1689m`), **frozen** + the same head.
- **P2 proposal, adaptation (H1 secondary):** DINOv3 ViT-B/16 + LoRA (r=16, qkv and proj) + the same head. If time permits.
- **M2 strong modern additional baseline:** DINOv2 ViT-B/14 frozen + the same head (for the question "DINOv3 or any foundation model?"). If time permits.
- **H2 simplest:** P2 (or P1 if P2 is not run) + tail-aware loss (logit adjustment, τ=1, from training pixel frequencies)
  vs. the same model with plain CE. If time permits; signal only.

## Training (identical in all runs)
- Augmentation: the dataset paper's pipeline (RandomResize scale (2048,512) ratio 0.5-2.0, RandomCrop 512 cat_max_ratio 0.75,
  flip, photometric distortion). If it does not fit in 4 GB, the crop is reduced in all runs together and recorded as a deviation.
- AdamW, lr 2e-4 (head and LoRA), fine-tuned backbone 2e-4 × 0.1, wd 0.01,
  poly schedule + 300-iteration warmup, bf16/fp16 mixed precision, effective batch 8 (gradient accumulation if needed).
- **Schedule:** every run uses the same number of iterations N (same number of samples seen). N is chosen after measuring throughput so that a run takes ≤ ~75 min,
  and is written, before the first run, not to `deviations.md` but to `log/plan.txt` (shipped as `pilot/results/plan.txt`); all runs use the same N.
- Selection: the model at the last iteration (no early stopping, no checkpoint selection by looking at dev). Dev is measured only at the end.

## Seeds
- B0 and P1: 2 seeds (0, 1). P2/M2/H2: 2 if time permits, otherwise 1 (a single seed is reported as signal only).

## Hypotheses and decision rule
- **H1 (safe):** P1 (frozen DINOv3 ViT-B + the same head) beats B0 under the same protocol by **δ = +2.0 points** in dev mIoU
  (difference of 2-seed means ≥ 2.0 and each seed of P1 higher than each seed of B0).
  - Rationale for δ: the card's δ is 2.0; above the median δ of the Q2 anchors (~+1.4), i.e. on the safe side; with a dev set of ~1,100 images
    seed noise in segmentation is typically 0.3-1 points, and 2 points is clearly above it.
  - If P1 fails but P2 holds: reported as "H1 did not hold in the frozen form, it held with LoRA" (a weaker result).
- **H2 (ambitious, simplest form):** the tail-aware loss beats plain CE by ≥ +1.5 in dev mIoU (the card's δ₂). In this smoke test it is
  signal only (no decision if there are few seeds).
- **H3 (analysis):** the P1 − B0 difference is reported separately for class-frequency terciles (head/mid/tail, by number of training images) and for
  mask-ratio subsets (small lesion: mask ratio in the image < median).

## Time budget
- One run ≤ 3 h (target ≤ 75 min), total ≤ 1 day; the GPU is shared with another experimenter via `flock`.

## Known limitations (stated in advance)
- A full ConvNeXt-L/40k reproduction will not be done on 4 GB; E1's "±2 points" condition can only be discussed by reporting the small-scale baseline and the paper's
  number side by side.
- The dataset paper's code uses test as validation (`val_dataloader` → `images/test`); our protocol
  uses dev, so an absolute comparison with the paper's number may be biased.
