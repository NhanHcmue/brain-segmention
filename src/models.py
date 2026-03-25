"""
Models:
  - SimCLRProjector  : MLP projection head cho pretraining
  - DoubleConv3D     : building block cho UNet decoder
  - DecoderBlock     : upsample + concat skip + double conv
  - ASPP3D           : multi-scale context bottleneck
  - ResNet50UNet     : full segmentation model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoder import ResNet50_3D_Encoder


# ─────────────────────────────────────────────
# SimCLR Projection Head
# ─────────────────────────────────────────────

class SimCLRProjector(nn.Module):
    """
    2-layer MLP projection head (SimCLR v2 style).
    Input : (B, 2048) — encoder output
    Output: (B, proj_dim) — L2-normalized embedding
    """

    def __init__(self, in_dim: int = 2048, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # L2 normalize được làm trong loss


# ─────────────────────────────────────────────
# UNet Decoder Blocks
# ─────────────────────────────────────────────

class DoubleConv3D(nn.Module):
    """Double Conv3d + InstanceNorm + LeakyReLU với residual shortcut."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.shortcut = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class DecoderBlock(nn.Module):
    """Upsample × 2 → concat skip connection → DoubleConv3D."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv3D(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ASPP3D(nn.Module):
    """
    Atrous Spatial Pyramid Pooling 3D.
    Multi-scale context ở bottleneck — giúp model nhận diện
    tumor ở nhiều kích thước khác nhau.
    """

    def __init__(self, in_ch: int, out_ch: int = 256, rates: tuple = (1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            ) for r in rates
        ])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_ch, out_ch, 1, bias=False),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv3d(out_ch * (len(rates) + 1), out_ch, 1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout3d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sz    = x.shape[2:]
        feats = [b(x) for b in self.branches]
        feats.append(F.interpolate(self.gap(x), size=sz, mode='trilinear', align_corners=False))
        return self.proj(torch.cat(feats, dim=1))


# ─────────────────────────────────────────────
# Full Segmentation Model
# ─────────────────────────────────────────────

class ResNet50UNet(nn.Module):
    """
    UNet với ResNet50 3D encoder + ASPP bottleneck + Deep Supervision.

    Encoder output (input 128³):
      x0: 32³   64ch     x1: 32³  256ch
      x2: 16³  512ch     x3: 8³  1024ch     x4: 4³  2048ch

    Decoder:
      ASPP(x4) → 256ch
      dec4(256+1024→512)  → 8³
      dec3(512+512 →256)  → 16³
      dec2(256+256 →128)  → 32³   skip=x1
      merge(128+64  → 64) → 32³   skip=x0 (không upsample, cùng resolution)
      up×2×2              → 128³
      head 1×1            → num_classes

    Deep supervision (training only):
      ds4: từ dec4 output → upsample về input size
      ds3: từ dec3 output → upsample về input size
    """

    def __init__(self, encoder: ResNet50_3D_Encoder, num_classes: int = 1):
        super().__init__()
        self.encoder = encoder

        self.aspp  = ASPP3D(2048, 256)
        self.dec4  = DecoderBlock(256,  1024, 512)
        self.dec3  = DecoderBlock(512,   512, 256)
        self.dec2  = DecoderBlock(256,   256, 128)
        self.merge = DoubleConv3D(128 + 64, 64)

        self.up = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, stride=2),
            nn.InstanceNorm3d(32, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.ConvTranspose3d(32, 16, 2, stride=2),
            nn.InstanceNorm3d(16, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.head = nn.Conv3d(16, num_classes, 1)

        # Deep supervision heads (training only)
        self.ds4 = nn.Conv3d(512, num_classes, 1)
        self.ds3 = nn.Conv3d(256, num_classes, 1)

    def forward(self, x: torch.Tensor):
        x0, x1, x2, x3, x4 = self.encoder(x)

        b  = self.aspp(x4)
        d4 = self.dec4(b,  x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)

        if d2.shape[2:] != x0.shape[2:]:
            x0 = F.interpolate(x0, size=d2.shape[2:], mode='trilinear', align_corners=False)
        d1  = self.merge(torch.cat([d2, x0], dim=1))
        out = self.head(self.up(d1))

        if self.training:
            sz  = x.shape[2:]
            ds4 = F.interpolate(self.ds4(d4), size=sz, mode='trilinear', align_corners=False)
            ds3 = F.interpolate(self.ds3(d3), size=sz, mode='trilinear', align_corners=False)
            return out, ds4, ds3
        return out
