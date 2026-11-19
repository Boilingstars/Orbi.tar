#!/bin/bash
# decoder_v2 240k: CPUs 0-2. Workers stay OMP=1; parent PPO is single-threaded.
set -euo pipefail
cd /root/Orbi.tar
export PYTHONUNBUFFERED=1
export PYTHONPATH=/root/Orbi.tar
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE
export OMP_WAIT_POLICY=PASSIVE
export ORBI_TRAINER_THREADS=1
export MALLOC_ARENA_MAX=2
unset ORBI_SMOKE
RUN=/root/Orbi.tar/experiments/runs/11_18_26_decoder_v2_240k
LOG=$RUN/tmux_launch.log
{
  echo "start $(date -Is)"
  echo "affinity=$(taskset -cp $$ 2>/dev/null || true)"
  echo "nproc=$(nproc)"
  echo "ORBI_TRAINER_THREADS=$ORBI_TRAINER_THREADS"
  free -h
  echo "load $(cat /proc/loadavg)"
  echo "---"
} | tee "$LOG"
taskset -c 0-2 .venv/bin/python -u "$RUN/launch.py" 2>&1 | tee -a "$LOG"
echo "end $(date -Is) exit=${PIPESTATUS[0]}" | tee -a "$LOG"
