"""
train_simclr.py — SimCLR Pretraining

Usage:
    python train_simclr.py --config configs/simclr.yaml
    python train_simclr.py --config configs/simclr.yaml --batch 64 --workers 16
"""
import argparse, os, random
import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentation import GPUAugmentationPair
from src.dataset       import SimCLRDataset
from src.encoder       import ResNet50_3D_Encoder
from src.losses        import NTXentLoss
from src.models        import SimCLRProjector


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = True   # auto-tune conv cho fixed input size

def load_config(path):
    with open(path, encoding ="utf-8") as f: return yaml.safe_load(f)

def log_system():
    print(f"PyTorch  : {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU      : {p.name}  ({p.total_memory/1e9:.1f} GB)")
        print(f"CUDA     : {torch.version.cuda}")
    else:
        print("GPU      : ✗ Không có CUDA!")
    ncpu = os.cpu_count() or 1
    print(f"CPU cores: {ncpu}  →  gợi ý num_workers={ncpu//2}\n")


def train(cfg):
    out_dir = cfg['output']['dir']
    os.makedirs(out_dir, exist_ok=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_system()
    set_seed(cfg['train']['seed'])

    # ── Dataset & DataLoader ──
    dataset = SimCLRDataset(
        train_dir          = cfg['data']['train_dir'],
        val_dir            = cfg['data'].get('val_dir', ''),
        modality           = cfg['data']['modality'],
        patch_size         = tuple(cfg['dataset']['patch_size']),
        patches_per_volume = cfg['dataset']['patches_per_volume'],
    )
    nw = cfg['train']['num_workers']
    bs = cfg['train']['batch_size']
    loader = DataLoader(
        dataset, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None,
    )
    print(f"DataLoader: {len(loader)} batches/epoch (batch={bs}, workers={nw})\n")

    # ── Models ──
    encoder   = ResNet50_3D_Encoder().to(DEVICE)
    projector = SimCLRProjector(2048, 2048, cfg['simclr']['projection_dim']).to(DEVICE)
    aug_pair  = GPUAugmentationPair().to(DEVICE)
    criterion = NTXentLoss(cfg['simclr']['temperature']).to(DEVICE)
    print(f"Encoder   : {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params")
    print(f"Projector : {sum(p.numel() for p in projector.parameters())/1e3:.0f}K params")

    # ── Optimizer & Scheduler ──
    all_params = list(encoder.parameters()) + list(projector.parameters())
    optimizer  = optim.AdamW(all_params, lr=cfg['optimizer']['lr'],
                              weight_decay=cfg['optimizer']['weight_decay'])
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['train']['num_epochs'], eta_min=cfg['scheduler']['eta_min'])
    scaler     = GradScaler()
    grad_accum = cfg['train'].get('grad_accum', 1)
    if grad_accum > 1:
        print(f"Grad accum: {grad_accum}  (effective batch = {bs*grad_accum})")

    # ── Resume ──
    start_epoch = 0; best_loss = float('inf'); loss_history = []
    ckpt_path = cfg['output']['checkpoint']
    if os.path.exists(ckpt_path):
        print(f'\nResuming from {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        encoder.load_state_dict(ckpt['encoder'])
        projector.load_state_dict(ckpt['projector'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss   = ckpt.get('best_loss', float('inf'))
        loss_history= ckpt.get('history', [])
        print(f"  epoch={start_epoch}, best_loss={best_loss:.4f}")

    num_epochs = cfg['train']['num_epochs']
    print(f"\n{'─'*50}\n  SimCLR Pretraining — {num_epochs} epochs\n{'─'*50}\n")

    for epoch in range(start_epoch, num_epochs):
        encoder.train(); projector.train()
        ep_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"Epoch {epoch+1:3d}/{num_epochs}")
        for step, x in enumerate(pbar):
            x = x.to(DEVICE, non_blocking=True)           # CPU→GPU non-blocking
            with torch.no_grad():
                v1, v2 = aug_pair(x)                       # GPU aug, không tốn CPU
            with autocast():
                loss = criterion(
                    projector(encoder.forward_flat(v1)),
                    projector(encoder.forward_flat(v2)),
                ) / grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(all_params, 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

            ep_loss += loss.item() * grad_accum
            if step % 10 == 0:
                mem = torch.cuda.memory_allocated()/1e9 if DEVICE.type=='cuda' else 0.0
                pbar.set_postfix(loss=f"{loss.item()*grad_accum:.4f}", GPU=f"{mem:.1f}GB")

        avg = ep_loss / len(loader)
        loss_history.append(avg)
        scheduler.step()
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] Loss: {avg:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        torch.save({'epoch':epoch,'encoder':encoder.state_dict(),
            'projector':projector.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'best_loss':best_loss,'history':loss_history},
            ckpt_path)

        if avg < best_loss:
            best_loss = avg
            torch.save(encoder.state_dict(), cfg['output']['best_encoder'])
            print(f"  ✓ Best encoder saved (loss={best_loss:.4f})")

    torch.save(encoder.state_dict(), cfg['output']['final_encoder'])
    print(f"\n✓ Done!  Best loss: {best_loss:.4f}")
    print(f"✓ Encoder saved → {cfg['output']['best_encoder']}")

    plt.figure(figsize=(10,4))
    plt.plot(loss_history, marker='o', linewidth=2, markersize=4)
    plt.title('SimCLR NT-Xent Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.grid(True, alpha=0.5); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'loss_curve.png'), dpi=150)
    print(f"✓ Loss curve → {out_dir}/loss_curve.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',   default='configs/simclr.yaml')
    p.add_argument('--data_dir', default=None)
    p.add_argument('--val_dir',  default=None)
    p.add_argument('--output',   default=None)
    p.add_argument('--epochs',   type=int, default=None)
    p.add_argument('--batch',    type=int, default=None, help='batch_size')
    p.add_argument('--workers',  type=int, default=None, help='num_workers')
    p.add_argument('--accum',    type=int, default=None, help='grad_accum')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cfg  = load_config(args.config)
    if args.data_dir: cfg['data']['train_dir']    = args.data_dir
    if args.val_dir:  cfg['data']['val_dir']       = args.val_dir
    if args.output:   cfg['output']['dir']         = args.output
    if args.epochs:   cfg['train']['num_epochs']   = args.epochs
    if args.batch:    cfg['train']['batch_size']   = args.batch
    if args.workers:  cfg['train']['num_workers']  = args.workers
    if args.accum:    cfg['train']['grad_accum']   = args.accum
    train(cfg)
