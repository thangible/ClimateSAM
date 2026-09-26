#!/bin/bash
# wait_gpu.sh <GiB>: block until the GPU has at least <GiB> free memory. Waiters are serialised with a lock, and the
# lock is kept for 120 s after returning, so the job started next can allocate its memory before anyone else checks.
LOCK="$(dirname "$0")/../results/logs/.gpu.lock"
exec 9>"$LOCK"
flock 9
need=$(( $1 * 1024 )); ok=0
while [ $ok -lt 2 ]; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -ge "$need" ]; then ok=$((ok + 1)); else ok=0; fi
  sleep 20
done
( sleep 120 ) &
