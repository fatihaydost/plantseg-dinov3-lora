#!/bin/bash
# After queue.sh finishes: P1 with the long schedule (matched to B0L), for schedule sensitivity.
cd "$(dirname "$0")"
until grep -q "queue done" results/queue.log; do sleep 60; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$HOME/.cache/plantseg/hf
PY=${PY:-python}  # Python interpreter of your venv
tag=P1L_s0
[ -f results/$tag/result.json ] || { echo "$(date '+%F %T') start $tag"; flock $HOME/.cache/plantseg-gpu.lock $PY train_pilot.py --tag $tag --bs 2 --accum 4 --bb dinov3_b --mode frozen --iters 32000 --seed 0 > results/$tag.out 2>&1; echo "$(date '+%F %T') done $tag $(tail -1 results/$tag.out | cut -c1-120)"; }
echo "$(date '+%F %T') queue2 done"
