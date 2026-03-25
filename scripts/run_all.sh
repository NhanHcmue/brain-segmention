#!/bin/bash
# run_all.sh — Chạy toàn bộ pipeline
# Usage:
#   bash scripts/run_all.sh --data_dir /path/to/BraTS2020
#   bash scripts/run_all.sh --gdrive_id 1AbCdEfGh
set -e

DATA_DIR=""; GDRIVE_ID=""; BATCH_SIMCLR=""; BATCH_UNET=""; WORKERS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)     DATA_DIR="$2";      shift 2 ;;
        --gdrive_id)    GDRIVE_ID="$2";     shift 2 ;;
        --batch_simclr) BATCH_SIMCLR="$2";  shift 2 ;;
        --batch_unet)   BATCH_UNET="$2";    shift 2 ;;
        --workers)      WORKERS="$2";       shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "=== Brain MRI SimCLR + UNet Pipeline ==="
pip install -r requirements.txt -q

# Download data nếu cần
if [ -n "$GDRIVE_ID" ]; then
    python setup_data.py --gdrive_id "$GDRIVE_ID"
fi

# Build CLI args
D=""; B1=""; B2=""; W=""
[ -n "$DATA_DIR" ]     && D="--data_dir $DATA_DIR"
[ -n "$BATCH_SIMCLR" ] && B1="--batch $BATCH_SIMCLR"
[ -n "$BATCH_UNET" ]   && B2="--batch $BATCH_UNET"
[ -n "$WORKERS" ]      && W="--workers $WORKERS"

echo -e "\n=== Step 1/3: SimCLR Pretraining ==="
python train_simclr.py --config configs/simclr.yaml $D $B1 $W

echo -e "\n=== Step 2/3: UNet Pretrained ==="
python train_unet.py --config configs/unet_pretrained.yaml $D $B2 $W

echo -e "\n=== Step 3/3: UNet Baseline ==="
python train_unet.py --config configs/unet_baseline.yaml $D $B2 $W

echo -e "\n✓ Done! Results in outputs/"
