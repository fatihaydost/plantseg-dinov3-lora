# T1 PlantSeg × DINOv3: E1/E2 pilot (smoke test) report (2026-10-11)

Pre-registration: `preregistration.md` (written before the runs). Deviations: `deviations.md`. Raw numbers: `pilot/results/analysis.json`, `pilot/results/*/result.json`.
Code: `pilot/` (`prepare.py`, `train_pilot.py`, `analyze.py`, `queue*.sh`). The test split was not evaluated with any model.

## 1. Data (E1)
- PlantSeg **v7** (Zenodo 17719108; the 7th and last version of concept record 13293890), CC BY-NC 4.0. 1.06 GB, zip CRC OK.
- 7,774 images + pixel masks, 115 diseases + background = 116 classes (all label values 0-115 are present).
  Official split (Metadata.csv): **train 5,367 / dev (validation) 846 / test 1,561**.
- Training images per class: min 2, median 38, max 225. 80.8% of the training pixels are background.
- Without applying the EXIF orientation some labels do not match the image; after applying it the mismatch is 0.
- **Leakage (pHash, Hamming ≤ 4):** 109/846 dev images (12.9%) and 193/1,561 test images (12.4%) have a near-duplicate in train;
  test↔dev 43. 244 of the duplicate pairs have different class labels (e.g. `apple_black_rot_80` and `grape_black_rot_42` are pixel-identical).
  So the test scores reported by the dataset paper are partly leaked, and there is also label noise. Clean dev (737) is reported separately;
  it does not change the ranking (below).

## 2. Protocol (identical in all runs)
The dataset paper's pipeline (RandomResize (2048,512) 0.5-2.0 · RandomCrop 512 cat_max 0.75 · flip · photometric), the same UperNet-lite head
(PPM+FPN, 256 channels), AdamW lr 2e-4 (fine-tuned backbone ×0.1), poly, batch 2×4 = 8, bf16, **16k iterations (32k samples ≈ 6 epochs)**,
the last iteration is evaluated; on dev: short side 512, whole image, mIoU at the original resolution (116 classes).

## 3. Results (dev mIoU, %)

| Code | Model | Iter | Seeds | dev mIoU | sd | mAcc | Clean dev | Tail | Small lesion | Train min |
|---|---|---|---|---|---|---|---|---|---|---|
| — | Paper: ConvNeXt-L UperNet (test, 40k×16) | — | — | **46.24** (test) | | 59.97 | | | | |
| B0 | ConvNeXt-T IN-22k, full fine-tuning | 16k | 2 | **10.71** | 0.42 | 16.56 | 10.30 | 0.00 | 7.99 | 54 |
| B0L | same, 2× schedule | 32k | 1 | **18.67** | — | 27.16 | 18.19 | 3.95 | 14.24 | 108 |
| P1 | DINOv3 ViT-B/16 **frozen** | 16k | 2 | **18.20** | 0.56 | 29.60 | 17.91 | 1.88 | 12.89 | 51 |
| P2 | DINOv3 ViT-B/16 + **LoRA** r16 | 16k | 2 | **23.13** | 1.76 | 35.26 | 23.42 | 4.46 | 16.55 | 78 |
| M2 | DINOv2 ViT-B/14 frozen | 16k | 1 | 17.38 | — | 27.90 | 16.98 | 2.17 | 13.18 | 62 |
| H2 | P2 + logit adjustment τ=1 | 16k | 1 | 19.95 | — | 56.56 | 19.83 | 14.99 | 10.45 | 77 |
| P1L | P1, 2× schedule | 32k | 1 | stopped (22,450/32,000 iter, no evaluation) | | | | | | |

Paired bootstrap (dev images, 2,000 resamples, 95%): P1−B0 s0 +7.39 [6.25; 9.56], s1 +7.59 [6.44; 9.96];
P2−B0 s0 +11.48 [9.87; 14.22], s1 +13.38 [11.45; 16.10]; P1−M2 +0.42 [−0.64; 1.69]; H2−P2 −1.94 [−5.67; −0.71];
B0L−B0 +8.26 [6.93; 9.51].

## 4. H1 decision and fair comparison
- **Decision under the pre-registration (equal iterations, 16k, against B0): H1 HELD.** P1 − B0 = **+7.49** (δ = 2.0), and P1 > B0 for both seeds.
- **But this baseline is under-trained.** When the B0 schedule was doubled it gained 8.3 points (B0L 18.67); the gap at 16k is largely the effect
  "few parameters converge quickly". Under an equal training budget:
  - Frozen DINOv3 (P1, 16k, 51 min) **18.20** ≈ ConvNeXt-T 2× schedule (B0L, 108 min) **18.67**. It reaches the same accuracy in half the time
    but is **not better**; the δ=2 advantage could not be shown in the frozen form under an equal budget.
  - **LoRA (P2, 16k, 78 min) 23.13 vs B0L 18.67: +4.5**, with shorter training. This is a comparison against the single-seed B0L (P2's two seeds are 21.89 and
    24.38; both are +3.2 / +5.7 above B0L). This is the only form of H1 that survives under an equal budget.
  - There is no equal-iteration 32k comparison (P1L vs B0L): P1L slowed down because of GPU sharing and was stopped at 22,450/32,000 when the report was written.
- Honest reading: **the claim "frozen DINOv3 beats the ConvNeXt baseline" was not demonstrated in this smoke test; the signal "DINOv3 + LoRA wins" is
  strong** (two seeds, +4.5 above the equal-budget baseline), but both the baseline and P2 are unsaturated; with the full schedule the gap may narrow or widen.
- DINOv3 vs DINOv2 (frozen, same head): +0.42 [−0.64; 1.69], no difference (single seed).

## 5. H2 signal
- H2 in its simplest form: **logit-adjusted CE on top of P2 (DINOv3 ViT-B + LoRA) (τ=1, training pixel frequency)**, counterpart P2 s0 with plain CE.
- mIoU **−1.94** (21.89 → 19.95), mAcc **+24.1** (32.42 → 56.56), aAcc 85.96 → 57.64. Tail-class mIoU 3.45 → **14.99**,
  mid 19.71 → 16.65, head 40.76 → 27.25; background IoU 0.88 → 0.56; the predicted foreground is 2.9 times the true foreground.
- Reading: the loss does open up the tail (classes with IoU>0: 77 → 97), but because of the background's 81% prior the adjustment is excessive; the model calls everything
  disease. In this form H2 **did not hold** (δ₂ = −1.9 instead of +1.5); the signal is "right direction but not calibrated": tuning τ or applying the prior
  only among the foreground classes is the next attempt. Single seed, not a decision.

## 6. H3 (analysis, first look)
- The tail (39 classes with 2-24 training images) is almost zero for every plain-CE model (B0 0.0; P1 1.9; P2 4.5). The gain is in head and mid
  classes: P2 − B0 head +17.2, mid +16.0, tail +4.5.
- Small lesions (mask ratio < dev median) are ~10 points lower than large lesions in every model; P2 − B0 small +8.6, large +15.9.
  So DINOv3's gain is **not concentrated in small lesions and the tail**; the opposite of the card's H3 expectation (first signal).
- On leaked dev images (109) scores are slightly higher than on clean ones (e.g. P1 20.8 vs 17.9), with the same ranking.

## 7. Why the absolute scores are so far below the paper's (46.24)
1. **Schedule:** the paper uses 40k iter × batch 16 = 640k samples; we use 32k samples (5%). B0 → B0L (2×) gains 8.3 points: the curve is still steep, far from saturation.
2. **Model size:** the paper uses ConvNeXt-L (121M); we use ConvNeXt-T (28M backbone) and ViT-B. Even in the paper ConvNeXt-B → L gains 5.6.
3. **Split and selection:** we evaluate on dev with the last iteration; in the paper's code `val_dataloader` points to the test folder, i.e. validation may
   have been done on test. Test is 12.4% leaked; dev is also 12.9% leaked, but clean dev does not change the result.
4. **39 of the 116 classes are in the tail (≤24 training images)** and these are not learned at all in the short schedule; since mIoU is a class mean,
   this alone pulls ~1/3 of the classes to zero.
- Conclusion: **the baseline was not reproduced within ±2 points**; E1's reproduction condition could not be met on this hardware.

## 8. Level, time, full-experiment estimate
- **Level: E1 partial** (data downloaded and counted, leakage measured; baseline not reproduced) · **E2 conditional** (the pre-registered H1 held at small scale, but
  under an equal budget only the LoRA form beats the baseline). By plan §3.3 this does not count as full E2: not against a reproduced baseline.
- Time: data + preparation ~30 min; GPU runs 10 Oct 21:40 → 11 Oct 10:08 (9 runs, ~10.5 h of training; the GPU was shared with another experimenter)
  + P1L (stopped midway).
- **Full-experiment estimate (not measured, by scaling):** on the 3050, ConvNeXt-T takes 0.2 s / 2 images → the paper's schedule (640k samples) is a single run of
  ~18 h; ConvNeXt-L and ViT-L do not fit on this card. A reasonable plan: a 24 GB+ GPU (A100/L4 class), ~6-10 h per run; ConvNeXt-L
  reproduction ×3 seeds + DINOv3 ViT-L frozen/LoRA ×3 + H2 ×3 ≈ 12 runs ≈ **~100 A100-hours** (3-4× on a T4). **A remote GPU is required.**

## 9. Uncertainties
- Because the baseline was not reproduced, this experiment says nothing about whether "DINOv3 beats 46.24"; only the relative order under an equal protocol.
- Single-seed comparisons (B0L, M2, H2) are signal; P2's seed deviation is 1.8 points (high).
- The LoRA hyperparameters (r, lr) were not tuned; the head is "UperNet-lite", not identical to the paper's UperNet.
- The pHash threshold (≤4) catches near-duplicates but can miss different frames that were cropped/resized; the count is a lower bound.
