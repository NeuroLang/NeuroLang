#!/bin/bash
set -u
# The measurement of the result cache on Margaret (sbatch -c 8 --mem=64G):
# W holds `master` and `branch` (git archive of each, a uv venv each) and the
# benchmark queries pain_map.dl, decode_mask.dl; per repetition, master,
# the branch with an empty cache, then the branch again on the same cache
# (a restart). Then: compare.py out/fmaster1 out/fmaster2 out/fcold1 ...
export HOME=/scratch/dwasserm/nl_memprobe HF_HUB_OFFLINE=1
W=/scratch/dwasserm/nl_impl/g3-result-cache
D=$HOME/neurolang_data
Q="$W/pain_map.dl $W/decode_mask.dl"
run() {  # name checkout cache
  local name=$1 co=$2 cache=$3
  echo "=== $name $(date +%T) load $(cut -d' ' -f1 /proc/loadavg)"
  ( cd $W/$co && .venv/bin/python $W/branch/benchmarks/result_cache/first_screen.py \
      neurosynth_iit $D $W/out/$name ${cache:+--cache $cache} --sessions 2 --queries $Q ) \
      > $W/out/$name.log 2>&1
  echo "exit $? $(date +%T)"
}
mkdir -p $W/out
for rep in 1 2; do
  run fmaster$rep master ""
  run fcold$rep branch $W/cache$rep
  run frestart$rep branch $W/cache$rep
done
ls -la $W/cache1 $W/cache2
