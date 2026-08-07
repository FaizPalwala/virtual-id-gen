#!/bin/bash
# ==========================================
# hpc_zip_data.sh — Archive the full data directory for off-node transfer
# ==========================================
#SBATCH --job-name=msc_zipdata
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --output=logs/zipdata_%j.out
#SBATCH --error=logs/zipdata_%j.err

# CPU-only job: NO --partition, NO --gres (don't burn GPU quota on zipping).
# The data (JPEG crops + PNG portraits) is already compressed, so the archive
# is a TRANSFER BUNDLE, not a compression exercise — store/fastest modes.
#
# Tool selection (first available):
#   7z    -> real .zip, multithreaded (-mmt)        [preferred: literal zip]
#   zstd  -> .tar.zst, -T0 uses all cores, fastest  [best speed]
#   pigz  -> .tar.gz, parallel gzip
#   gzip  -> .tar.gz single-threaded fallback

set -euo pipefail

DATA_DIR="/scratch/$USER/datagen/data"
OUT_DIR="/scratch/$USER/datagen"
STAMP=$(date +%Y%m%d)

echo "[$(date)] === Pre-flight ==="
if [ ! -d "$DATA_DIR" ]; then
    echo "[ERROR] $DATA_DIR not found — nothing to archive."
    exit 1
fi
du -sh "$DATA_DIR"
echo "[$(date)] Scratch space:"
df -h /scratch 2>/dev/null | tail -1 || echo "  (df unavailable for /scratch — continuing)"
echo "[$(date)] Cores allocated: ${SLURM_CPUS_PER_TASK:-8}"

echo ""
echo "[$(date)] === Archiving ==="
if command -v 7z >/dev/null 2>&1; then
    OUT="$OUT_DIR/data_${STAMP}.zip"
    echo "[$(date)] Using 7z (-mmt parallel, -mx=1 fastest deflate)"
    7z a -tzip -mmt -mx=1 "$OUT" "$DATA_DIR"
elif command -v zstd >/dev/null 2>&1; then
    OUT="$OUT_DIR/data_${STAMP}.tar.zst"
    echo "[$(date)] Using tar + zstd (-T0 all cores, -1 fastest)"
    tar -C "$(dirname "$DATA_DIR")" -cf - "$(basename "$DATA_DIR")" \
        | zstd -T0 -1 -o "$OUT"
elif command -v pigz >/dev/null 2>&1; then
    OUT="$OUT_DIR/data_${STAMP}.tar.gz"
    echo "[$(date)] Using tar + pigz (-p cores)"
    tar -C "$(dirname "$DATA_DIR")" -cf - "$(basename "$DATA_DIR")" \
        | pigz -p "${SLURM_CPUS_PER_TASK:-8}" > "$OUT"
else
    OUT="$OUT_DIR/data_${STAMP}.tar.gz"
    echo "[WARN] No parallel compressor (7z/zstd/pigz) — single-threaded gzip."
    echo "[WARN] Install p7zip for parallel zips: conda install -c conda-forge p7zip"
    tar -C "$(dirname "$DATA_DIR")" -czf "$OUT" "$(basename "$DATA_DIR")"
fi

echo ""
echo "[$(date)] === Integrity check ==="
case "$OUT" in
    *.zip)    7z t "$OUT" ;;
    *.tar.zst) zstd -t "$OUT" && tar -tf "$OUT" >/dev/null ;;
    *.tar.gz)  gzip -t "$OUT" && tar -tf "$OUT" >/dev/null ;;
esac
echo "[$(date)] Integrity OK: $OUT"

echo ""
echo "[$(date)] === Checksum ==="
sha256sum "$OUT" | tee "$OUT.sha256"

echo ""
echo "[$(date)] === Done ==="
ls -lh "$OUT" "$OUT.sha256"
echo "Transfer from the login node:"
echo "  rsync -avP aire:${OUT} /path/on/mac/"
