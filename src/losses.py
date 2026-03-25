"""
Loss functions:
  - NTXentLoss       : SimCLR contrastive loss
  - TverskyLoss      : penalize FN > FP — tốt cho tumor nhỏ
  - FocalLoss3D      : giảm easy negatives
  - CombinedSegLoss  : Tversky + Focal + Deep Supervision
  - dice_score()     : metric (không gradient)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# SimCLR Loss
# ─────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR).

    Với batch size N:
    - 2N embeddings (N cặp positive)
    - Mỗi sample i → positive là sample i+N (và ngược lại)
    - Còn lại 2(N-1) samples là negative
    - loss = cross_entropy(cosine_sim / temp, positive_labels)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1, z2: (B, D) — output của projector
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        B  = z1.shape[0]
        z  = torch.cat([z1, z2], dim=0)                          # (2B, D)
        sim = torch.mm(z, z.T) / self.temperature                # (2B, 2B)
        sim = sim.masked_fill(
            torch.eye(2 * B, dtype=torch.bool, device=z.device), float('-inf')
        )
        labels = torch.cat([
            torch.arange(B, 2 * B),
            torch.arange(0, B),
        ]).to(z.device)
        return F.cross_entropy(sim, labels)


# ─────────────────────────────────────────────
# Segmentation Losses
# ─────────────────────────────────────────────

class TverskyLoss(nn.Module):
    """
    Tversky loss: TP / (TP + alpha*FP + beta*FN).
    alpha=0.3, beta=0.7 → penalize FN nặng hơn FP
    → giảm miss tumor (tốt cho class imbalance nặng).
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-5):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p  = torch.sigmoid(logits)
        tp = (p * targets).sum(dim=(2, 3, 4))
        fp = (p * (1 - targets)).sum(dim=(2, 3, 4))
        fn = ((1 - p) * targets).sum(dim=(2, 3, 4))
        t  = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - t.mean()


class FocalLoss3D(nn.Module):
    """
    Focal loss — giảm contribution của easy negatives (background).
    alpha=0.75 → weight cao hơn cho foreground (tumor).
    gamma=2.0  → down-weight easy examples.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt  = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        at  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (at * (1 - pt) ** self.gamma * bce).mean()


class CombinedSegLoss(nn.Module):
    """
    TverskyLoss + FocalLoss + Deep Supervision.

    Weights:
      main output : 1.0
      ds4         : 0.3
      ds3         : 0.2
    """

    def __init__(self):
        super().__init__()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.focal   = FocalLoss3D(alpha=0.75, gamma=2.0)

    def _base(self, logits, targets):
        return self.tversky(logits, targets) + self.focal(logits, targets)

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, (tuple, list)):
            main, ds4, ds3 = outputs
            return (
                self._base(main, targets)
                + 0.3 * self._base(ds4, targets)
                + 0.2 * self._base(ds3, targets)
            )
        return self._base(outputs, targets)


# ─────────────────────────────────────────────
# Metric
# ─────────────────────────────────────────────

def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thr: float = 0.5,
    smooth: float = 1e-5,
) -> float:
    """Dice Score (không gradient) — dùng trong validation."""
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    preds = (torch.sigmoid(logits) > thr).float()
    inter = (preds * targets).sum(dim=(2, 3, 4))
    union = preds.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))
    return ((2 * inter + smooth) / (union + smooth)).mean().item()
