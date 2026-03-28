"""
src/dataset_preprocessed.py — Dataset load từ preprocessed .pt patches

Dùng thay thế BrainMRISegDataset sau khi chạy preprocess_patches.py.
__getitem__ chỉ còn torch.load() → không nibabel, không normalize, không crop.
"""

import json
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset


class PreprocessedPatchDataset(Dataset):
    """
    Load preprocessed patches từ .pt files (output của preprocess_patches.py).

    Tốc độ: ~5ms/patch thay vì ~1.8s với nib.load()

    Args:
        patch_dir : thư mục output của preprocess_patches.py
        split     : "train" hoặc "val"
        ram_cache : load tất cả vào RAM (chỉ bật nếu RAM > 60GB)
    """

    def __init__(self, patch_dir: str, split: str = 'train', ram_cache: bool = False):
        self.patch_dir = Path(patch_dir)
        self.ram_cache = ram_cache

        manifest_path = self.patch_dir / 'manifest.json'
        assert manifest_path.exists(), (
            f'Không tìm thấy manifest.json tại {self.patch_dir}\n'
            f'Chạy preprocess_patches.py trước!'
        )
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.files = [str(self.patch_dir / p) for p in manifest[split]]
        assert len(self.files) > 0, f'Không có patch nào trong split="{split}"'

        self._cache = {}
        if ram_cache:
            print(f'[{split}] Caching {len(self.files)} patches vào RAM...')
            for path in self.files:
                self._cache[path] = torch.load(path, map_location='cpu', weights_only=True)
            sz = sum(v['img'].nbytes + v['seg'].nbytes for v in self._cache.values())
            print(f'  RAM: {sz/1e9:.1f} GB')

        print(f'[PreprocessedPatchDataset:{split}] {len(self.files)} patches | '
              f'{"RAM" if ram_cache else "disk"}')

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]
        if self.ram_cache:
            data = self._cache[path]
        else:
            data = torch.load(path, map_location='cpu', weights_only=True)
        return data['img'].float(), data['seg'].float()
