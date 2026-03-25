"""
ResNet50 3D Encoder — convert ResNet50 2D → 3D convolution.
Dùng chung cho SimCLR pretraining và UNet segmentation.
"""
import copy
import torch
import torch.nn as nn
from torchvision.models import resnet50


def _to3d(x):
    return (x[0],) * 3 if isinstance(x, (tuple, list)) else (x, x, x)


def convert_to_3d_inplace(module: nn.Module) -> None:
    """Đệ quy thay Conv2d → Conv3d, BatchNorm2d → BatchNorm3d."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, nn.Conv3d(
                child.in_channels, child.out_channels,
                kernel_size=_to3d(child.kernel_size),
                stride=_to3d(child.stride),
                padding=_to3d(child.padding),
                bias=child.bias is not None,
            ))
        elif isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.BatchNorm3d(child.num_features))
        else:
            convert_to_3d_inplace(child)


class ResNet50_3D_Encoder(nn.Module):
    """
    ResNet50 chuyển sang 3D convolution.

    Input : (B, 1, D, H, W)
    Output: tuple (x0, x1, x2, x3, x4) — skip connections cho UNet decoder

      x0: stride/4   64 ch
      x1: stride/4  256 ch  (layer1 — stride=1 trong ResNet)
      x2: stride/8  512 ch
      x3: stride/16 1024 ch
      x4: stride/32 2048 ch

    Với input 128³:
      x0: 32³    x1: 32³    x2: 16³    x3: 8³    x4: 4³
    """

    def __init__(self):
        super().__init__()
        r2d = resnet50(weights=None)

        # Stem: 3-ch 2D → 1-ch 3D
        self.conv1   = nn.Conv3d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm3d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)

        # ResNet layers — deepcopy rồi convert sang 3D
        self.layer1 = copy.deepcopy(r2d.layer1)
        self.layer2 = copy.deepcopy(r2d.layer2)
        self.layer3 = copy.deepcopy(r2d.layer3)
        self.layer4 = copy.deepcopy(r2d.layer4)

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            convert_to_3d_inplace(layer)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x0 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4

    def forward_flat(self, x) -> torch.Tensor:
        """Global feature vector (B, 2048) — dùng trong SimCLR."""
        *_, x4 = self.forward(x)
        return self.avgpool(x4).flatten(1)
