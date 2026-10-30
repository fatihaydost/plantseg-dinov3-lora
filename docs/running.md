# How to run the full experiment (Colab, `src/`)

## Colab (5 steps)
1. Upload the contents of `src/` and `notebooks/01_train_colab.ipynb` to Google Drive under `MyDrive/plantseg-dinov3-lora/` (the notebook, `queue_initial.json`, and the code in `src/`).
2. Open `01_train_colab.ipynb` in Colab and set the runtime type to **A100 GPU**, high RAM.
3. *Runtime → Run all*; approve the Drive permission dialog. **Keep the tab open and do not let the computer sleep** (Colab Pro does not run in the background).
4. When the queue finishes or stops, the last cell shuts the runtime down itself. If the session drops (24 h limit, connection), repeat steps 2-3: finished runs are skipped, interrupted ones resume.
5. If the code is updated, replace only the code folder on Drive; the queue file (`queue.json`), `ckpt/`, `log/`, `results/`, and `model/` are not deleted.

## Monitoring
- Drive `plantseg-dinov3-lora/status.md`: current run, %, it/s, estimated finish, time per run, estimated total A100-hours; a warning in CAPITAL LETTERS if the budget (61) is exceeded. Minute-level detail in `log/<run>.log`, the result in `results/<run>.json` (dev mIoU, mAcc, per-class IoU, clean dev, head/mid/tail, small/large lesion) + `results/<run>_dev_per_image.npz` (for bootstrap), the final weights in `model/<run>.pt`.
- In the first session, `MEASURE_SPEED=True` measures 40 iterations for each model family (~15 min); the "prior" rows in `status.md` turn into "measured". The first real estimate is read from here.
- H2 (i, j): after the Kaggle τ sweep, fill in `la_tau` in `queue.json` and set `status` to `pending` (it runs at the very end of the queue). A failed run is skipped after 3 attempts; to retry, set `status: pending, attempt: 0`.
- The test split is closed: in the notebook set `OPEN_TEST=True` (+ `TEST_RUNS`) or use `src/evaluate_test.py --yes_test`. Clean test = the 1,336 test images without a pHash ≤4 near-duplicate in train or dev (computed locally; consistent with the pilot's counts: 109/193/43, 569/244 pairs).
- Kaggle: `kaggle kernels push -p kaggle/<directory>`. `plantseg-h2-tau` (`kaggle/h2-tau-sweep`; T4x2, two queues in parallel, ~4-6 h estimated; output `h2_tau_summary.json`), `plantseg-temiz-test` (`kaggle/clean-test`; GPU off, CPU is enough; `clean_test.json`, threshold sensitivity), `plantseg-v3l-frozen` (`kaggle/v3l-frozen`; T4x2 DDP, 640k samples; it checkpoints and stops at 11.3 h) → if unfinished, push `plantseg-v3l-frozen-devam` (`kaggle/v3l-frozen-resume`; it takes the first kernel's output as its source and continues). If the code changes, first run `python3 kaggle/build_kernels.py`.
- Kaggle kernel slugs keep their original ids (`plantseg-temiz-test`, `plantseg-v3l-frozen-devam`) so that the published kernels stay linked.
