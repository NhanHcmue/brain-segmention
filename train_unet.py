"""
train_unet.py — UNet Segmentation (Pretrained + Baseline)

Usage:
    python train_unet.py --config configs/unet_pretrained.yaml
    python train_unet.py --config configs/unet_baseline.yaml

Thay đổi so với phiên bản gốc:
    - Dùng PreprocessedPatchDataset thay BrainMRISegDataset
      → __getitem__ = torch.load() duy nhất, không nibabel
    - Bật TF32 cho 3090/4090 → +40% tốc độ matmul
    - tqdm fix: ncols=80, dynamic_ncols=False để không bị lỗi scroll terminal remote
    - GPU utilization monitor mỗi 50 steps
    - compile model với torch.compile() nếu PyTorch >= 2.0
"""
import argparse, os, random
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend — fix lỗi display khi train remote
import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentation import GPUAugmentation3D
from src.encoder       import ResNet50_3D_Encoder
from src.losses        import CombinedSegLoss, dice_score
from src.models        import ResNet50UNet
from src.dataset_preprocessed import PreprocessedPatchDataset


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = True

def load_config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)

def log_system():
    print(f"PyTorch  : {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU      : {p.name}  ({p.total_memory/1e9:.1f} GB)")
        print(f"CUDA     : {torch.version.cuda}")
        # TF32 — tăng tốc ~40% trên Ampere (3090/4090/A100) với precision loss không đáng kể
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        print(f"TF32     : enabled (Ampere optimization)")
    else:
        print("GPU      : ✗ Không có CUDA!")
    print(f"CPU cores: {os.cpu_count()}\n")

def gpu_mem():
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{used:.1f}/{total:.0f}GB"
    return "N/A"

def plot_and_save(history, title, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'],   label='Val')
    ax1.set_title('Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(history['val_dice'], color='green', label='Dice')
    ax2.axhline(0.9, color='red', linestyle='--', label='Target 0.90')
    ax2.set_title('Val Dice'); ax2.legend(); ax2.grid(True)
    plt.suptitle(title); plt.tight_layout()
    path = os.path.join(out_dir, 'training_history.png')
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"✓ Plot → {path}")
    print(f"  Best Dice: {max(history['val_dice']):.4f}")


def train(cfg):
    out_dir = cfg['output']['dir']
    os.makedirs(out_dir, exist_ok=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_system()
    set_seed(cfg['train']['seed'])

    encoder_path = cfg['pretrain'].get('encoder_path')
    label        = 'Pretrained' if encoder_path else 'Baseline'
    print(f"Mode: {label}\n")

    # ── Dataset (preprocessed patches) ──
    patch_dir = cfg['data']['patch_dir']
    ram_cache = cfg['data'].get('ram_cache', False)

    train_ds = PreprocessedPatchDataset(patch_dir, split='train', ram_cache=ram_cache)
    val_ds   = PreprocessedPatchDataset(patch_dir, split='val',   ram_cache=ram_cache)

    nw = cfg['train']['num_workers']
    bs = cfg['train']['batch_size']

    train_loader = DataLoader(
        train_ds, bs, shuffle=True,
        num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,   # tăng từ 2 → 4 vì I/O rất nhanh
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, bs, shuffle=False,
        num_workers=max(1, nw // 2),
        pin_memory=True,
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )
    print(f"Train: {len(train_ds)} patches | Val: {len(val_ds)} patches")
    print(f"Loader: {len(train_loader)} train batches | {len(val_loader)} val batches\n")

    # ── Model ──
    enc = ResNet50_3D_Encoder().to(DEVICE)
    if encoder_path:
        assert os.path.exists(encoder_path), f"Encoder not found: {encoder_path}"
        state = torch.load(encoder_path, map_location=DEVICE, weights_only=True)
        if any(k.startswith('encoder.') for k in state):
            state = {k[8:]: v for k, v in state.items() if k.startswith('encoder.')}
        enc.load_state_dict(state, strict=False)
        print(f"✓ Loaded pretrained encoder: {encoder_path}")
    else:
        print("[BASELINE] Random init encoder")

    model = ResNet50UNet(enc, num_classes=1).to(DEVICE)

    # torch.compile — PyTorch 2.0+ giảm kernel launch overhead ~10-15%
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("✓ torch.compile enabled")
        except Exception:
            pass  # Fallback nếu compile lỗi (CUDA driver cũ, v.v.)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params/1e6:.1f}M params | GPU mem after load: {gpu_mem()}")

    gpu_aug    = GPUAugmentation3D(**cfg['augmentation']).to(DEVICE)
    criterion  = CombinedSegLoss()
    scaler     = GradScaler('cuda')
    grad_accum = cfg['train'].get('grad_accum', 1)
    num_epochs = cfg['train']['num_epochs']
    lr         = cfg['optimizer']['lr']
    wd         = cfg['optimizer']['weight_decay']

    freeze_ep    = cfg.get('finetune', {}).get('freeze_encoder_epochs', 0) if encoder_path else 0
    enc_lr_scale = cfg.get('finetune', {}).get('encoder_lr_scale', 0.05)

    def make_optimizer(phase):
        if phase == 1 or not encoder_path:
            return optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr, weight_decay=wd,
            )
        enc_p = [p for n, p in model.named_parameters() if 'encoder' in n]
        dec_p = [p for n, p in model.named_parameters() if 'encoder' not in n]
        return optim.AdamW(
            [{'params': enc_p, 'lr': lr * enc_lr_scale}, {'params': dec_p, 'lr': lr}],
            weight_decay=wd,
        )

    def make_scheduler(opt, ep_start):
        def lr_fn(ep):
            ep += ep_start; warmup = 5
            if ep < warmup: return (ep + 1) / warmup
            return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(1, num_epochs - warmup)))
        return optim.lr_scheduler.LambdaLR(opt, lr_fn)

    if freeze_ep > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print(f"Phase 1: encoder frozen {freeze_ep} epochs")

    optimizer = make_optimizer(1)
    scheduler = make_scheduler(optimizer, 0)

    # ── Resume ──
    start_epoch = 0; best_dice = 0.0
    ckpt_path   = cfg['output']['checkpoint']
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch'] + 1
        best_dice   = ckpt.get('best_dice', 0)
        print(f"Resumed epoch {start_epoch}, best_dice={best_dice:.4f}")

    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    print(f"\n{'─'*60}")
    print(f"  UNet {label} — {num_epochs} epochs")
    print(f"{'─'*60}\n")

    for epoch in range(start_epoch, num_epochs):

        # Unfreeze encoder
        if epoch == freeze_ep and freeze_ep > 0:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = make_optimizer(2)
            scheduler = make_scheduler(optimizer, epoch)
            print(f"\nPhase 2: encoder unfrozen at epoch {epoch+1}")

        # ── Train ──
        model.train()
        tr_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        # tqdm fix: ncols=80 tránh lỗi scroll trên remote terminal
        pbar = tqdm(
            train_loader,
            desc=f"[{label}] Ep {epoch+1:3d}/{num_epochs}",
            ncols=80,
            dynamic_ncols=False,
            leave=True,
        )
        for step, (img, seg) in enumerate(pbar):
            img = img.to(DEVICE, non_blocking=True)
            seg = seg.to(DEVICE, non_blocking=True)

            with torch.no_grad():
                img = gpu_aug(img)

            with autocast('cuda'):
                loss = criterion(model(img), seg) / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            tr_loss += loss.item() * grad_accum

            if step % 50 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item()*grad_accum:.4f}",
                    gpu=gpu_mem(),
                )

        # ── Validation ──
        model.eval()
        va_loss = va_dice = 0.0
        with torch.no_grad():
            for img, seg in tqdm(val_loader, desc="  val", ncols=80, leave=False):
                img = img.to(DEVICE, non_blocking=True)
                seg = seg.to(DEVICE, non_blocking=True)
                with autocast('cuda'):
                    out  = model(img)
                    loss = criterion(out, seg)
                va_loss += loss.item()
                va_dice += dice_score(out, seg)

        atl    = tr_loss / len(train_loader)
        avl    = va_loss / len(val_loader)
        adc    = va_dice / len(val_loader)
        cur_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history['train_loss'].append(atl)
        history['val_loss'].append(avl)
        history['val_dice'].append(adc)

        print(
            f"[{label}] Ep[{epoch+1}/{num_epochs}] "
            f"Loss {atl:.4f}/{avl:.4f} | "
            f"Dice {adc:.4f} | "
            f"LR {cur_lr:.2e} | "
            f"GPU {gpu_mem()}"
        )

        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_dice': best_dice,
        }, ckpt_path)

        if adc > best_dice:
            best_dice = adc
            torch.save(model.state_dict(), cfg['output']['best_model'])
            print(f"  ✓ Best model saved (Dice={best_dice:.4f})")

    print(f"\n[{label}] Done!  Best Val Dice: {best_dice:.4f}")
    plot_and_save(history, f"UNet {label}", out_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',  required=True)
    p.add_argument('--encoder', default=None, help='Override pretrain.encoder_path')
    p.add_argument('--output',  default=None)
    p.add_argument('--epochs',  type=int, default=None)
    p.add_argument('--batch',   type=int, default=None)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--accum',   type=int, default=None)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cfg  = load_config(args.config)
    if args.encoder: cfg['pretrain']['encoder_path'] = args.encoder
    if args.output:  cfg['output']['dir']            = args.output
    if args.epochs:  cfg['train']['num_epochs']      = args.epochs
    if args.batch:   cfg['train']['batch_size']      = args.batch
    if args.workers: cfg['train']['num_workers']     = args.workers
    if args.accum:   cfg['train']['grad_accum']      = args.accum
    train(cfg)
