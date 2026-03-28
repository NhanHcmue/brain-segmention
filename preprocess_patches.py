"""
preprocess_patches.py — Offline preprocessing cho BraTS2020

Chạy 1 lần trước khi train (không cần GPU):
    python preprocess_patches.py \
        --src data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
        --dst data/BraTS2020_patches \
        --n_patches 32 \
        --fp16

Upload data/BraTS2020_patches/ lên Drive 1 lần.
Sau đó train chỉ cần torch.load() — không cần nibabel, không normalize, không crop.
"""

import argparse
import json
import os
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm


def normalize(v: np.ndarray) -> np.ndarray:
    brain = v > 0
    if brain.sum() > 100:
        v = (v - v[brain].mean()) / (v[brain].std() + 1e-8)
        v[~brain] = 0.0
    return v.astype(np.float32)


def foreground_patch(img, seg, patch_size, fg_prob=0.8):
    pd = ph = pw = patch_size
    d, h, w = img.shape
    if d < pd or h < ph or w < pw:
        img = np.pad(img, [(max(0,pd-d),0),(max(0,ph-h),0),(max(0,pw-w),0)])
        seg = np.pad(seg, [(max(0,pd-d),0),(max(0,ph-h),0),(max(0,pw-w),0)])
        d, h, w = img.shape
    fg = np.argwhere(seg > 0)
    if len(fg) > 0 and random.random() < fg_prob:
        c  = fg[random.randint(0, len(fg)-1)]
        sd = int(np.clip(c[0]-pd//2, 0, d-pd))
        sh = int(np.clip(c[1]-ph//2, 0, h-ph))
        sw = int(np.clip(c[2]-pw//2, 0, w-pw))
    else:
        sd = random.randint(0, d-pd)
        sh = random.randint(0, h-ph)
        sw = random.randint(0, w-pw)
    return img[sd:sd+pd, sh:sh+ph, sw:sw+pw], seg[sd:sd+pd, sh:sh+ph, sw:sw+pw]


def load_nii(patient_dir, pid, suffix):
    for ext in ['.nii', '.nii.gz']:
        p = os.path.join(patient_dir, f'{pid}_{suffix}{ext}')
        if os.path.exists(p):
            return nib.load(p).get_fdata(dtype=np.float32)
    return None


def preprocess(args):
    src   = Path(args.src)
    dst   = Path(args.dst)
    dtype = torch.float16 if args.fp16 else torch.float32

    all_pids = sorted(d for d in os.listdir(src) if (src/d).is_dir() and not d.startswith('.'))
    random.seed(args.seed)
    random.shuffle(all_pids)
    n_val    = max(1, int(len(all_pids) * args.val_split))
    val_set  = set(all_pids[:n_val])
    manifest = {'train': [], 'val': []}

    for split in ['train', 'val']:
        out_dir = dst / split
        out_dir.mkdir(parents=True, exist_ok=True)
        pids = [p for p in all_pids if (p in val_set) == (split == 'val')]

        print(f'\n[{split.upper()}] {len(pids)} volumes')
        for pid in tqdm(pids, desc=split, ncols=80, leave=True):
            pdir    = str(src / pid)
            img_vol = load_nii(pdir, pid, args.modality)
            seg_vol = load_nii(pdir, pid, 'seg')
            if img_vol is None or seg_vol is None:
                tqdm.write(f'  skip {pid}: missing file')
                continue

            img_vol = normalize(img_vol)
            seg_vol = (seg_vol > 0).astype(np.float32)

            saved = 0
            for _ in range(args.n_patches * 10):
                if saved >= args.n_patches:
                    break
                img_p, seg_p = foreground_patch(img_vol, seg_vol, args.patch)
                if (img_p != 0).mean() < 0.05:
                    continue
                out_path = out_dir / f'{pid}_p{saved:02d}.pt'
                torch.save({
                    'img': torch.from_numpy(img_p.copy()).unsqueeze(0).to(dtype),
                    'seg': torch.from_numpy(seg_p.copy()).unsqueeze(0).to(dtype),
                }, str(out_path))
                manifest[split].append(str(out_path.relative_to(dst)))
                saved += 1

    manifest_path = dst / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    total_size = sum(
        os.path.getsize(dst / p)
        for p in manifest['train'] + manifest['val']
        if (dst / p).exists()
    )
    print(f'\n✓ Done!')
    print(f'  Train : {len(manifest["train"])} patches')
    print(f'  Val   : {len(manifest["val"])} patches')
    print(f'  Disk  : {total_size/1e9:.1f} GB')
    print(f'  → {manifest_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src',       required=True)
    p.add_argument('--dst',       required=True)
    p.add_argument('--modality',  default='t1ce')
    p.add_argument('--patch',     type=int, default=128)
    p.add_argument('--n_patches', type=int, default=32,
                   help='Patches per volume (32 fp16 ≈ 47GB, 16 fp16 ≈ 23GB)')
    p.add_argument('--val_split', type=float, default=0.2)
    p.add_argument('--fp16',      action='store_true', help='Save float16 (half disk)')
    p.add_argument('--seed',      type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)

    est = 369 * args.n_patches * (args.patch**3) * (2 if args.fp16 else 4) / 1e9
    print(f'Patch size  : {args.patch}³')
    print(f'N patches   : {args.n_patches} per volume')
    print(f'Dtype       : {"float16" if args.fp16 else "float32"}')
    print(f'Est. disk   : ~{est:.0f} GB')
    preprocess(args)


if __name__ == '__main__':
    main()
