"""
GPU Augmentation 3D — chạy hoàn toàn trên GPU tensor.
Không dùng numpy/scipy → GPU không phải chờ CPU.
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPUAugmentation3D(nn.Module):
    """
    Augmentation 3D batched trên GPU.
    Input/Output: (B, 1, D, H, W) float32 trên DEVICE.

    Pipeline:
      1. Random flip      (3 axes, p=0.5 mỗi axis)
      2. Random rot90     (1 trong 3 plane, k=0..3)
      3. Intensity jitter (scale+shift per-batch)
      4. Gaussian noise   (p=0.5)
      5. Crop-resize      (p=0.5, dùng F.interpolate CUDA kernel)
      6. Cutout 3D        (p=0.4, zero sub-volume)
    """

    def __init__(self, p_flip=0.5, p_noise=0.5, p_crop=0.5, p_cutout=0.4):
        super().__init__()
        self.p_flip   = p_flip
        self.p_noise  = p_noise
        self.p_crop   = p_crop
        self.p_cutout = p_cutout

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._random_flip(x)
        x = self._random_rot90(x)
        x = self._intensity_jitter(x)
        x = self._gaussian_noise(x)
        x = self._random_crop_resize(x)
        x = self._random_cutout(x)
        return x

    def _random_flip(self, x):
        for dim in [2, 3, 4]:
            if torch.rand(1).item() < self.p_flip:
                x = x.flip(dim)
        return x

    def _random_rot90(self, x):
        k     = torch.randint(0, 4, (1,)).item()
        plane = random.choice([(2, 3), (2, 4), (3, 4)])
        if k > 0:
            x = torch.rot90(x, k=k, dims=plane)
        return x

    def _intensity_jitter(self, x):
        scale = torch.empty(1, device=x.device).uniform_(0.85, 1.15)
        shift = torch.empty(1, device=x.device).uniform_(-0.15, 0.15)
        return x * scale + shift

    def _gaussian_noise(self, x):
        if torch.rand(1).item() < self.p_noise:
            x = x + torch.randn_like(x) * 0.05
        return x

    def _random_crop_resize(self, x):
        if torch.rand(1).item() < self.p_crop:
            B, C, D, H, W = x.shape
            r  = torch.empty(1).uniform_(0.75, 0.95).item()
            cd = int(D * r); ch = int(H * r); cw = int(W * r)
            sd = torch.randint(0, D - cd + 1, (1,)).item()
            sh = torch.randint(0, H - ch + 1, (1,)).item()
            sw = torch.randint(0, W - cw + 1, (1,)).item()
            x  = F.interpolate(
                x[:, :, sd:sd+cd, sh:sh+ch, sw:sw+cw],
                size=(D, H, W), mode='trilinear', align_corners=False,
            )
        return x

    def _random_cutout(self, x):
        if torch.rand(1).item() < self.p_cutout:
            B, C, D, H, W = x.shape
            cd = D // 5; ch = H // 5; cw = W // 5
            sd = torch.randint(0, D - cd + 1, (1,)).item()
            sh = torch.randint(0, H - ch + 1, (1,)).item()
            sw = torch.randint(0, W - cw + 1, (1,)).item()
            x  = x.clone()
            x[:, :, sd:sd+cd, sh:sh+ch, sw:sw+cw] = 0.0
        return x


class GPUAugmentationPair(nn.Module):
    """Tạo 2 views augmented độc lập — dùng cho SimCLR."""

    def __init__(self):
        super().__init__()
        self.aug = GPUAugmentation3D()

    @torch.no_grad()
    def forward(self, x):
        return self.aug(x), self.aug(x)
