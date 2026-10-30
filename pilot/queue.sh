#!/bin/bash
# Sequential run queue; each run holds the GPU lock. A finished run (one with result.json) is skipped.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$HOME/.cache/plantseg/hf
PY=${PY:-python}  # Python interpreter of your venv
run() { tag=$1; shift
  [ -f results/$tag/result.json ] && { echo "skip $tag"; return; }
  echo "$(date '+%F %T') start $tag"
  flock $HOME/.cache/plantseg-gpu.lock $PY train_pilot.py --tag $tag --bs 2 --accum 4 "$@" > results/$tag.out 2>&1
  echo "$(date '+%F %T') done $tag $(tail -1 results/$tag.out | cut -c1-120)"
}
run B0_s0  --bb convnext_t --mode full   --iters 16000 --seed 0
run P1_s0  --bb dinov3_b   --mode frozen --iters 16000 --seed 0
run B0_s1  --bb convnext_t --mode full   --iters 16000 --seed 1
run P1_s1  --bb dinov3_b   --mode frozen --iters 16000 --seed 1
run P2_s0  --bb dinov3_b   --mode lora   --iters 16000 --seed 0
run P2_s1  --bb dinov3_b   --mode lora   --iters 16000 --seed 1
run M2_s0  --bb dinov2_b   --mode frozen --iters 16000 --seed 0
run H2_s0  --bb dinov3_b   --mode lora   --iters 16000 --seed 0 --logit_adj 1.0
run B0L_s0 --bb convnext_t --mode full   --iters 32000 --seed 0
echo "$(date '+%F %T') queue done"
