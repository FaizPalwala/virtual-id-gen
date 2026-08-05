#!/bin/bash
# ==========================================
# partition_seeds.sh — Split the 750-seed pool into 15 shard partitions.
#
# Run on the Mac AFTER download.py (which selects seeds via CLIP+KMeans).
# Partitioning BEFORE uploading to Aire guarantees each shard gets its
# own 50 unique seeds — no cross-shard reuse (the v1.0.0 bug).
#
# Usage:
#   bash scripts/partition_seeds.sh <seeds_dir> [nshards=15]
#
# Output: <seeds_dir>/shard_00/ … shard_14/, each with 50 seed_*.jpg.
# ==========================================
set -euo pipefail

SEEDS_DIR="${1:?Usage: $0 <seeds_dir> [nshards] [seed=42]}"
NSHARDS="${2:-15}"
SEED="${3:-42}"

if [ ! -d "$SEEDS_DIR" ]; then
    echo "ERROR: seeds directory not found: $SEEDS_DIR"
    exit 1
fi

# Collect filenames, shuffle deterministically, partition.
mapfile -t seeds < <(ls "$SEEDS_DIR"/seed_*.jpg | sort)
nseeds="${#seeds[@]}"
echo "Found $nseeds seeds in $SEEDS_DIR, partitioning into $NSHARDS shards"

# Deterministic shuffle with a fixed seed (different from the RNG seed used
# for generation — this is a static permutation so the partition itself is
# reproducible).
perl -e '
use List::Util qw(shuffle);
srand(7);
print join("\n", shuffle(@ARGV)) . "\n";
' "${seeds[@]}" > /tmp/seed_shuffle.txt

# Create shard dirs and distribute seeds round-robin from shuffled list.
for i in $(seq 0 $((NSHARDS - 1))); do
    shard=$(printf '%02d' "$i")
    mkdir -p "$SEEDS_DIR/shard_$shard"
done

i=0
while read -r seedpath; do
    shard=$(printf '%02d' $((i % NSHARDS)))
    cp "$seedpath" "$SEEDS_DIR/shard_$shard/"
    i=$((i + 1))
done < /tmp/seed_shuffle.txt
rm -f /tmp/seed_shuffle.txt

echo "Done.  Partitioned $nseeds seeds across $NSHARDS shards:"
for d in "$SEEDS_DIR"/shard_*; do
    echo "  $(basename "$d"): $(ls "$d"/*.jpg | wc -l) seeds"
done
