#!/usr/bin/env bash
# Phase 2 of report/WORKFLOW_2.md: build and score pools 1-4 (skips files that already exist). Train split only.
# Usage (repo root, Git Bash): bash _build_pools2.sh > generator_comparison/pools2/build.log 2>&1
set -e
cd "$(dirname "$0")"
D=generator_comparison/pools2
mkdir -p "$D"
for k in 1 2 3 4; do
  seed=$((10 * k + 1))   # 11, 21, 31, 41: no template-noise seed shared across pools
  if [ ! -f "$D/pool_$k.npz" ]; then
    echo "=== pool $k (seed $seed) $(date +%T) ==="
    py -3.14 -u _beat_gen_pool.py generator_comparison/template_gen/checkpoints/tpl_8x64_last.pt 0,3200,1800,1400 "$D/pool_$k.npz" --weights ema --seed "$seed"
  fi
  if [ ! -f "$D/scores_$k.npz" ]; then
    echo "=== scores $k $(date +%T) ==="
    py -3.14 -u -m generator_comparison.quality_filter "$D/pool_$k.npz" "$D/scores_$k.npz"
  fi
done
echo "=== all done $(date +%T) ==="
