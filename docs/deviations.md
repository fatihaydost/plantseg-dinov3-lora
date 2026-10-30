# Deviations from the pre-registration

Each item says whether it was written before or after the run concerned. Dates are 2026; "11 Oct" etc. Item numbers are cited elsewhere in the repository.

Short commit hashes (e.g. `f86ba3b`) refer to the original development repository, where the pre-registration and every result were committed with timestamps; this public repository is a cleaned snapshot of it.

1. **Data version (before the runs).** Zenodo record 17719108 = version 7 of concept record 13293890 (versions index 6, `is_last`,
   2025-11-25), i.e. the v7 requested by the reviewer. Official split in Metadata.csv: train 5,367 / validation 846 / test 1,561 (total 7,774;
   69/11/20%). The pre-registration said "a dev set of ~1,100 images"; the actual dev set has 846 images.
2. **EXIF orientation (before the runs).** For some images the label was drawn on the EXIF-rotated image (e.g. `apple_rust_google_0065`).
   Images were read with the EXIF orientation applied, as mmcv/cv2 do; afterwards the image-label size mismatch is 0.
3. **Near-duplicate leakage (counted before the runs; reviewer note).** pHash (64 bit, Hamming ≤ 4), images only:
   109/846 dev images (12.9%) and 193/1,561 test images (12.4%) have a near-duplicate in train; test↔dev 43. 244 of the duplicate pairs carry
   **different class labels** (e.g. `apple_black_rot_80` = `grape_black_rot_42`, pixel-identical). Details: `pilot/results/leakage.json`.
   - The training set was not cleaned (the official split is kept, for comparability with the paper).
   - Additional evaluation: **clean dev** = the 737 dev images with no near-duplicate in train. The primary metric is the full dev set, as pre-registered;
     clean dev is reported additionally for every run (same predictions of the same model, on a subset).
   - The test count was done with hashes only, without models or labels; test labels were not opened and the test set was not evaluated with any model.
4. **Batch and schedule (before the runs).** A batch of 4 × 512 did not fit in 4 GB (OOM); batch 2 × gradient accumulation 4 (effective 8, as pre-registered).
   N = 16,000 iterations (32k samples ≈ 6 epochs; 5% of the paper's schedule). The chosen N was written to the original run plan (`pilot/results/plan.txt`); the queue is `pilot/queue.sh`.
5. **Additional B0-long run (added before the runs).** The short schedule may favour the frozen backbone (few parameters converge quickly). To measure this,
   B0 is run with a 2× schedule (N=32,000), one seed. The H1 decision is made on the same-N comparison as pre-registered; B0-long is for interpretation only.
6. **Evaluation resolution (before the runs).** Images were pre-resized to short side 512 (the dataset paper's test scale); the logits
   were bilinearly upsampled to the original label resolution in channel chunks and the argmax was taken (mathematically identical to a full argmax).
   For DINOv2 (patch 14) the input was padded right/bottom to a multiple of 14, for the others to a multiple of 32, and the logit padding was cropped off to realign.
7. **Label consistency (information).** In 8 train/dev images the foreground value in the mask differs from the Metadata class or there are several; left untouched.
8. **Additional P1-long run (added after seeing the B0_s0 and P1_s0 results, 23:45).** B0_s0 reached 10.41 mIoU and P1_s0 17.80; both are
   far below the paper's level (46.24), i.e. there is clear under-training at 16k iterations. To see whether the gap persists under a long schedule,
   P1 was also added with 32k iterations (matched to B0-long), one seed. Because it was added after seeing results, it is for interpretation only; the H1 decision
   is made on the pre-registered paired 16k comparison.
9. **H2 implementation (as pre-registered; note written after the result).** Logit adjustment with τ=1, from the training pixel frequencies (background 80.8%),
   added to the logits during training only, on top of P2 (DINOv3 ViT-B + LoRA), one seed (s0). τ was not tuned on dev. Because the background prior is so
   large, the model over-predicted foreground by ~2.9× (background IoU 0.88 → 0.56). Tuning τ or removing the background from the prior
   is the next step; it was not run in this smoke test (instruction: no new runs).
10. **P1-long (P1L_s0) was already running (started at 10:09) when the instruction "do not start new runs" was given; it was then stopped.** While the report was being written
   it was at iteration 22,450/32,000 (slowed down by GPU sharing); in line with the rule to shut down one's own processes it was stopped on 11 Oct at ~12:15 and not evaluated.
11. **H1 decision (tied to the pre-registration).** The pre-registered decision is made against B0 with the same 16k iterations. That B0 is seriously under-trained at 16k
   (+8.3 with B0L) was seen only after the run; therefore the equal-training-time comparison is also reported but does not change the decision.
12. **The M2 baseline changed (17 Oct).** The pre-registration said "DINOv2 ViT-B/14 frozen + the same head, if time permits". The run actually performed was DINOv2 ViT-L/14 +
   LoRA (`h_v2L_lora_s0`), an exact counterpart of the DINOv3 arm (same recipe, size, LoRA), single seed: to test "DINOv3 or any foundation model?" under
   equal conditions. The setting was not chosen by looking at dev results; the test for this run was opened on 17 Oct (item 16). Result: indistinguishable from
   DINOv3.

## Full experiment (after 11 Oct; written during the 23 Oct audit)

The items below are decisions made during the full experiment, but they were not written to this file at the time; they were found missing in the read-only
audit of 23 Oct and added retroactively. Each item says whether the decision was made before or after the test. The test was opened on 13 Oct at 19:56;
the first commit of the original repository (`f86ba3b`, 13 Oct 19:23) precedes it.

13. **The full experiment's design differs from the pre-registration (after the pilot, before the full-experiment runs, 11 Oct).** The pre-registration defines only the pilot:
   frozen (P1) and LoRA (P2) ViT-B, ConvNeXt-T, 16k iterations, 2 seeds. Full experiment: DINOv3 ViT-L/16 + LoRA,
   ConvNeXt-L (our protocol) and ConvNeXt-L with the dataset paper's recipe (arm `c`), frozen DINOv3 ViT-L (Kaggle), 40k iterations ×
   batch 16, 3 seeds for DINOv3 and ConvNeXt. Rationale: in the pilot the baselines were not saturated and the only form of H1 that survived under an equal budget was
   LoRA (`docs/pilot_report.md` §4). The decision was made after the pilot's dev result was seen; no separate pre-registration was written for the full experiment, and
   the hypotheses were carried over from the pre-registration. The initial queue is `src/queue_initial.json` (in the original repository's first commit).
14. **The primary form of H1 did not hold (note written after the result).** The pre-registered primary H1 was that frozen DINOv3 beats the baseline on dev by δ=+2.0.
   In the full experiment, frozen DINOv3 ViT-L reached 42.48 and ConvNeXt-L 42.78 ± 0.57 (dev): it did not hold. The LoRA form held (dev +3.69,
   worst DINOv3 seed 45.67 > best ConvNeXt seed 43.29). Under the pre-registration's rule the result is reported as "H1 did not hold in the frozen form, it held
   with LoRA (a weaker result)".
15. **H2 changed and τ was chosen by looking at dev (before the full-experiment H2 run).** Pre-registration: τ=1, all classes, training pixel
   frequency. Full experiment: logit adjustment on foreground classes only (`la_mod='fg'`, `adj_c = τ · log(115 · q_c)`, background 0) and
   τ=0.5; τ was chosen in a short-schedule sweep with ViT-B on Kaggle **by looking at dev mIoU** (`results/kaggle/h2_tau_summary.json`, README §9,
   `results/kaggle/h2_tau_summary.json`). The statement "τ was not tuned on dev" in item 9 applies only to the pilot. Result: +1.31 on dev (below the pre-registered
   threshold of +1.5; the interval contains 0), +0.34 on test → H2 did not hold according to the pre-registration; the dev number is optimistic because τ was selected on the same split.
16. **The test split was opened more than once.** 13 Oct: a–f and i (7 runs). 16 Oct: g (ConvNeXt-L s2). 17 Oct: h (DINOv2 ViT-L + LoRA) and the v5 extra inference
   with the same weights (class-agnostic mask, example figures; `extra_inference.py` with the test-confirmation flag). g and h were in the initial queue
   as "deferred (budget)", i.e. they are not runs fitted after the test; but the decisions to schedule them were made after the test
   (`9a3ea5c` 15 Oct, `a2c203c` 16 Oct). No training setting, checkpoint, or hyperparameter was changed according to the test result. g only adds a baseline seed, and h's result does not favour
   v3; the risk of bias is low, but the statement "the test was opened once" was wrong.
17. **Analyses defined after the test (exploratory).** The leakage-interaction bootstrap (`c8ea05e`, 13 Oct after the test; its direction had been written
   as a dev observation before the test, `b8a338f`), H3's directional "tail − head" and "small − large" tests (the pre-registration's H3 says only
   "reported separately"), image-level class accuracy, class-agnostic mask IoU. No multiple-comparison correction was
   applied; these are reported as exploratory.
18. **The statistical method was corrected (23 Oct, `927ebb3`, after the test).** BCa instead of the percentile bootstrap interval; in the clean/leaked and
   small/large lesion subsets mIoU is computed over the GT classes of that subset (previously, classes with no GT in the subset but wrongly predicted
   entered the mean as 0), and over the shared GT classes for subset comparisons. The full set is unchanged (mmseg rule). The reason was two errors
   found in the audit; which results changed under the new method is in README §4.
19. **Number of seeds.** DINOv3 + LoRA and ConvNeXt-L have 3 seeds (the pre-registration said 2); the paper recipe, frozen DINOv3, H2, and DINOv2 have a single seed
   (signal only). The planned `j_H2_s1` run was not performed (budget).
