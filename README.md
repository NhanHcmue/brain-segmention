# SimCLR + UNet — Brain MRI Segmentation

Self-supervised pretraining (SimCLR) + fine-tuning (UNet) trên BraTS2020.

## Cấu trúc project

```
├── src/
│   ├── augmentation.py   # GPUAugmentation3D — augment trên GPU, không dùng CPU
│   ├── encoder.py        # ResNet50_3D_Encoder — ResNet50 convert sang 3D
│   ├── models.py         # SimCLRProjector, ResNet50UNet (ASPP + Deep Supervision)
│   ├── losses.py         # NTXentLoss, TverskyLoss, FocalLoss, CombinedSegLoss
│   ├── dataset.py        # SimCLRDataset, BrainMRISegDataset
│   └── __init__.py
├── configs/
│   ├── simclr.yaml           # Hyperparams pretraining
│   ├── unet_pretrained.yaml  # Hyperparams fine-tuning với pretrained encoder
│   └── unet_baseline.yaml    # Hyperparams baseline (random init)
├── train_simclr.py       # Script 1: SimCLR pretraining
├── train_unet.py         # Script 2/3: UNet training (dùng cho cả pretrained & baseline)
├── outputs/              # Checkpoints, plots (không push lên git)
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
```

## Chạy training

bash scripts/run_all.sh --gdrive_id 17zSUKeFJzTWh64OQz74ufzavdA0vl-Ps

### Bước 1 — Sửa đường dẫn data trong config

Mở `configs/simclr.yaml`, sửa:
```yaml
data:
  train_dir: "/path/to/MICCAI_BraTS2020_TrainingData"
  val_dir:   "/path/to/MICCAI_BraTS2020_ValidationData"
```

### Bước 2 — SimCLR Pretraining

```bash
python train_simclr.py --config configs/simclr.yaml

# Override từ CLI (không cần sửa yaml):
python train_simclr.py \
  --config configs/simclr.yaml \
  --data_dir /data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
  --epochs 100 \
  --batch 16
```

Output: `outputs/simclr/best_encoder.pth`

### Bước 3 — UNet Pretrained (fine-tuning)

```bash
# Sửa pretrain.encoder_path trong configs/unet_pretrained.yaml,
# hoặc override bằng --encoder:
python train_unet.py \
  --config configs/unet_pretrained.yaml \
  --encoder outputs/simclr/best_encoder.pth
```

### Bước 4 — UNet Baseline (so sánh)

```bash
python train_unet.py --config configs/unet_baseline.yaml
```

## Chạy trên các GPU cloud

### Vast.ai / RunPod / Lambda Labs

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt

# Mount data vào /data/ rồi chạy:
python train_simclr.py --config configs/simclr.yaml --data_dir /data/MICCAI_BraTS2020_TrainingData
```

### Google Colab

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
!pip install -r requirements.txt

!python train_simclr.py \
  --config configs/simclr.yaml \
  --data_dir /content/drive/MyDrive/BraTS2020/MICCAI_BraTS2020_TrainingData \
  --output /content/drive/MyDrive/outputs/simclr
```

### Kaggle

```python
# Trong Kaggle notebook:
import subprocess
subprocess.run(['git', 'clone', 'https://github.com/YOUR_USERNAME/YOUR_REPO.git'])
%cd YOUR_REPO
subprocess.run(['pip', 'install', '-r', 'requirements.txt', '-q'])

!python train_simclr.py \
  --config configs/simclr.yaml \
  --data_dir /kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
  --output /kaggle/working/outputs/simclr
```

## GPU Pipeline

```
CPU Workers:  load .nii → normalize → foreground crop → pin_memory
                    ↓  (non_blocking transfer)
GPU:  GPUAugmentation3D (flip/rot/noise/crop-resize/cutout)
              ↓
      Encoder (ResNet50 3D) → Projector → NT-Xent Loss   [SimCLR]
      Encoder (ResNet50 3D) → ASPP → UNet Decoder → Seg  [UNet]
```

CPU chỉ làm I/O — augmentation nặng hoàn toàn trên GPU.

## Kết quả kỳ vọng

| Model | Val Dice |
|---|---|
| UNet Baseline (random init) | ~0.75–0.82 |
| UNet Pretrained (SimCLR)    | ~0.88–0.93 |
