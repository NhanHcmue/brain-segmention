"""
BraTS2020 Dataset cho SimCLR pretraining và UNet segmentation.

CPU chỉ làm:  load .nii → normalize → foreground crop → tensor
Augmentation: hoàn toàn trên GPU (GPUAugmentation3D trong train loop)
→ GPU không bao giờ phải chờ CPU augment.
"""
import os
import random
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────

def get_simclr_files(root_dir: str, modality: str) -> List[str]:
    """
    Trả về list đường dẫn .nii file (không cần mask).
    Dùng cho SimCLR pretraining — không cần nhãn.
    """
    files = []
    if not root_dir or not os.path.exists(root_dir):
        return files
    for pid in sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')
    ):
        ppath = os.path.join(root_dir, pid)
        for ext in ['.nii', '.nii.gz']:
            fp = os.path.join(ppath, f'{pid}_{modality}{ext}')
            if os.path.exists(fp):
                files.append(fp)
                break
    return files


def get_seg_files(
    data_dir: str,
    modality: str,
    seg_suffix: str = 'seg',
) -> Tuple[List[str], List[str]]:
    """
    Trả về (img_files, mask_files) cho segmentation.
    Bỏ qua patient nếu thiếu image hoặc mask.
    """
    img_files, mask_files = [], []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f'Không tìm thấy: {data_dir}')

    patient_dirs = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
    )
    skipped = 0
    for pid in patient_dirs:
        ppath  = os.path.join(data_dir, pid)
        img_p  = mask_p = None
        for ext in ['.nii', '.nii.gz']:
            p = os.path.join(ppath, f'{pid}_{modality}{ext}')
            if os.path.exists(p):
                img_p = p; break
        for ext in ['.nii', '.nii.gz']:
            p = os.path.join(ppath, f'{pid}_{seg_suffix}{ext}')
            if os.path.exists(p):
                mask_p = p; break
        if img_p and mask_p:
            img_files.append(img_p)
            mask_files.append(mask_p)
        else:
            skipped += 1

    print(f'[Dataset] Found {len(img_files)} pairs, skipped {skipped}')
    assert len(img_files) > 0 and len(img_files) == len(mask_files)
    return img_files, mask_files


# ─────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    """Z-score normalize trên brain mask (BraTS đã skull-strip → background=0)."""
    brain = v > 0
    if brain.sum() > 100:
        v = (v - v[brain].mean()) / (v[brain].std() + 1e-8)
        v[~brain] = 0.0
    return v.astype(np.float32)


def _foreground_patch(
    img: np.ndarray,
    seg: np.ndarray,
    patch_size: Tuple[int, int, int],
    fg_prob: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Foreground-biased patch sampling.
    fg_prob=0.8 → 80% patch center tại tumor voxel,
                  20% random (giúp model học background context).
    """
    d, h, w    = img.shape
    pd, ph, pw = patch_size

    # Pad nếu volume nhỏ hơn patch
    if d < pd or h < ph or w < pw:
        img = np.pad(img, [(max(0, pd-d), 0), (max(0, ph-h), 0), (max(0, pw-w), 0)])
        seg = np.pad(seg, [(max(0, pd-d), 0), (max(0, ph-h), 0), (max(0, pw-w), 0)])
        d, h, w = img.shape

    fg = np.argwhere(seg > 0)
    if len(fg) > 0 and random.random() < fg_prob:
        c  = fg[random.randint(0, len(fg) - 1)]
        sd = int(np.clip(c[0] - pd // 2, 0, d - pd))
        sh = int(np.clip(c[1] - ph // 2, 0, h - ph))
        sw = int(np.clip(c[2] - pw // 2, 0, w - pw))
    else:
        sd = random.randint(0, d - pd)
        sh = random.randint(0, h - ph)
        sw = random.randint(0, w - pw)

    return img[sd:sd+pd, sh:sh+ph, sw:sw+pw], seg[sd:sd+pd, sh:sh+ph, sw:sw+pw]


# ─────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────

class SimCLRDataset(Dataset):
    """
    Dataset cho SimCLR pretraining — không cần nhãn.
    CPU chỉ load + normalize + foreground crop.
    Augmentation tạo 2 views thực hiện trên GPU trong train loop.
    """

    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        modality: str = 't1ce',
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        patches_per_volume: int = 8,
    ):
        self.patch_size         = patch_size
        self.patches_per_volume = patches_per_volume

        files_tr    = get_simclr_files(train_dir, modality)
        files_va    = get_simclr_files(val_dir,   modality) if val_dir else []
        self.files  = files_tr + files_va

        print(f'[SimCLRDataset] Train: {len(files_tr)} | Val: {len(files_va)} | '
              f'Total samples: {len(self)}')
        if len(self.files) == 0:
            raise RuntimeError(
                'Không tìm thấy file nào!\n'
                'Kiểm tra train_dir/val_dir trong config.'
            )

    def __len__(self):
        return len(self.files) * self.patches_per_volume

    def __getitem__(self, idx: int) -> torch.Tensor:
        fidx = idx // self.patches_per_volume
        vol  = nib.load(self.files[fidx]).get_fdata(dtype=np.float32)
        vol  = _normalize(vol)

        d, h, w    = vol.shape
        pd, ph, pw = self.patch_size
        if d < pd or h < ph or w < pw:
            vol = np.pad(vol, [(max(0,pd-d),0),(max(0,ph-h),0),(max(0,pw-w),0)])
            d, h, w = vol.shape

        # Foreground crop
        patch = None
        for _ in range(15):
            sd = random.randint(0, d - pd)
            sh = random.randint(0, h - ph)
            sw = random.randint(0, w - pw)
            p  = vol[sd:sd+pd, sh:sh+ph, sw:sw+pw]
            if (p != 0).mean() > 0.1:
                patch = p; break
        if patch is None:
            patch = vol[(d-pd)//2:(d+pd)//2, (h-ph)//2:(h+ph)//2, (w-pw)//2:(w+pw)//2]

        return torch.from_numpy(patch.copy()).unsqueeze(0)   # (1, D, H, W)


class BrainMRISegDataset(Dataset):
    """
    Dataset BraTS2020 segmentation.
    CPU: load .nii → normalize → foreground crop → tensor
    Augmentation nặng → GPU (trong train loop).
    """

    def __init__(
        self,
        image_files: List[str],
        mask_files: List[str],
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        patches_per_volume: int = 8,
        is_train: bool = True,
    ):
        assert len(image_files) == len(mask_files)
        self.image_files        = image_files
        self.mask_files         = mask_files
        self.patch_size         = patch_size
        self.patches_per_volume = patches_per_volume
        self.is_train           = is_train

        print(f'[SegDataset] {len(image_files)} vols × {patches_per_volume} = {len(self)} samples')

    def __len__(self):
        return len(self.image_files) * self.patches_per_volume

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fidx = idx // self.patches_per_volume
        img  = nib.load(self.image_files[fidx]).get_fdata(dtype=np.float32)
        seg  = nib.load(self.mask_files[fidx]).get_fdata(dtype=np.float32)
        img  = _normalize(img)
        seg  = (seg > 0).astype(np.float32)

        fg_prob     = 0.8 if self.is_train else 1.0
        img, seg    = _foreground_patch(img, seg, self.patch_size, fg_prob)

        return (
            torch.from_numpy(img).unsqueeze(0),  # (1, D, H, W)
            torch.from_numpy(seg).unsqueeze(0),
        )
