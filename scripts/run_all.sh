#!/bin/bash
# run_all.sh — Chạy toàn bộ pipeline
#
# Usage:
#   bash scripts/run_all.sh --data_dir /path/to/MICCAI_BraTS2020_TrainingData
#   bash scripts/run_all.sh --gdrive_id 1AbCdEfGh --workers 8
#
# Nếu đã chạy SimCLR rồi, bỏ qua bước 1:
#   bash scripts/run_all.sh --data_dir /path/to/... --skip_simclr
#
# Nếu đã preprocessing rồi, bỏ qua bước 2:
#   bash scripts/run_all.sh --data_dir /path/to/... --skip_preprocess

set -e

# ── Parse args ──
DATA_DIR=""; GDRIVE_ID=""; PATCH_DIR="data/BraTS2020_patches"
BATCH_SIMCLR=""; BATCH_UNET=""; WORKERS=""
SKIP_SIMCLR=0; SKIP_PREPROCESS=0
N_PATCHES=32; FP16=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)       DATA_DIR="$2";       shift 2 ;;
        --gdrive_id)      GDRIVE_ID="$2";      shift 2 ;;
        --patch_dir)      PATCH_DIR="$2";      shift 2 ;;
        --batch_simclr)   BATCH_SIMCLR="$2";   shift 2 ;;
        --batch_unet)     BATCH_UNET="$2";     shift 2 ;;
        --workers)        WORKERS="$2";        shift 2 ;;
        --n_patches)      N_PATCHES="$2";      shift 2 ;;
        --fp16)           FP16="--fp16";       shift 1 ;;
        --skip_simclr)    SKIP_SIMCLR=1;       shift 1 ;;
        --skip_preprocess) SKIP_PREPROCESS=1;  shift 1 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "  Brain MRI SimCLR + UNet Pipeline"
echo "========================================"
pip install -r requirements.txt -q

# ── Download data nếu cần ──
if [ -n "$GDRIVE_ID" ]; then
    echo -e "\n=== Downloading data from Google Drive ==="
    python setup_data.py --gdrive_id "$GDRIVE_ID"
fi

# Resolve DATA_DIR
if [ -z "$DATA_DIR" ]; then
    # Thử tìm path mặc định sau khi setup_data.py giải nén
    for candidate in \
        "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData" \
        "data/MICCAI_BraTS2020_TrainingData"; do
        if [ -d "$candidate" ]; then
            DATA_DIR="$candidate"
            break
        fi
    done
fi

if [ -z "$DATA_DIR" ] && [ $SKIP_PREPROCESS -eq 0 ]; then
    echo "ERROR: --data_dir chưa được chỉ định và không tìm thấy data/"
    exit 1
fi

# Build CLI args
W_ARG="";  B1_ARG=""; B2_ARG=""
[ -n "$WORKERS" ]      && W_ARG="--workers $WORKERS"
[ -n "$BATCH_SIMCLR" ] && B1_ARG="--batch $BATCH_SIMCLR"
[ -n "$BATCH_UNET" ]   && B2_ARG="--batch $BATCH_UNET"

# ─────────────────────────────────────
# Step 1 — SimCLR Pretraining
# ─────────────────────────────────────
if [ $SKIP_SIMCLR -eq 0 ]; then
    echo -e "\n=== Step 1/3: SimCLR Pretraining ==="
    python train_simclr.py \
        --config configs/simclr.yaml \
        --data_dir "$DATA_DIR" \
        $B1_ARG $W_ARG
else
    echo -e "\n=== Step 1/3: SimCLR — SKIPPED (--skip_simclr) ==="
    if [ ! -f "outputs/simclr/best_encoder.pth" ]; then
        echo "WARNING: outputs/simclr/best_encoder.pth không tồn tại!"
    else
        echo "  ✓ Found: outputs/simclr/best_encoder.pth"
    fi
fi

# ─────────────────────────────────────
# Step 2 — Offline Preprocessing
# ─────────────────────────────────────
if [ $SKIP_PREPROCESS -eq 0 ]; then
    echo -e "\n=== Step 2/3: Offline Preprocessing (.nii → .pt patches) ==="
    echo "  src       : $DATA_DIR"
    echo "  dst       : $PATCH_DIR"
    echo "  n_patches : $N_PATCHES per volume"
    echo "  fp16      : ${FP16:-no}"

    python preprocess_patches.py \
        --src "$DATA_DIR" \
        --dst "$PATCH_DIR" \
        --n_patches $N_PATCHES \
        $FP16

    echo "  ✓ Preprocessing done → $PATCH_DIR"
else
    echo -e "\n=== Step 2/3: Preprocessing — SKIPPED (--skip_preprocess) ==="
    if [ ! -f "$PATCH_DIR/manifest.json" ]; then
        echo "ERROR: $PATCH_DIR/manifest.json không tồn tại!"
        echo "Chạy lại không có --skip_preprocess hoặc kiểm tra --patch_dir"
        exit 1
    fi
    echo "  ✓ Found: $PATCH_DIR/manifest.json"
fi

# ─────────────────────────────────────
# Step 3 — UNet Pretrained
# ─────────────────────────────────────
echo -e "\n=== Step 3/3a: UNet Pretrained (SimCLR encoder) ==="
python train_unet.py \
    --config configs/unet_pretrained.yaml \
    $B2_ARG $W_ARG

# ─────────────────────────────────────
# Step 4 — UNet Baseline
# ─────────────────────────────────────
echo -e "\n=== Step 3/3b: UNet Baseline (random init) ==="
python train_unet.py \
    --config configs/unet_baseline.yaml \
    $B2_ARG $W_ARG

echo -e "\n✓ Pipeline hoàn thành! Results in outputs/"
echo "  SimCLR encoder : outputs/simclr/best_encoder.pth"
echo "  UNet pretrained: outputs/unet_pretrained/best_model.pth"
echo "  UNet baseline  : outputs/unet_baseline/best_model.pth"
