"""
setup_data.py — Tải và giải nén BraTS2020 từ Google Drive

Usage:
    # Tải file .zip từ Google Drive và giải nén vào ./data/
    python setup_data.py --gdrive_id FILE_ID_CUA_BAN

    # Nếu đã tải sẵn file .zip ở local:
    python setup_data.py --zip_path /path/to/BraTS2020.zip

    # Lấy FILE_ID từ link share:
    # https://drive.google.com/file/d/1AbCdEfGhIjKlMn/view
    #                                   ^^^^^^^^^^^^^^^^ đây là FILE_ID

Cấu trúc sau khi giải nén (data/ sẽ được tạo tự động):
    data/
    ├── BraTS2020_TrainingData/
    │   └── MICCAI_BraTS2020_TrainingData/
    │       ├── BraTS20_Training_001/
    │       │   ├── BraTS20_Training_001_t1ce.nii
    │       │   ├── BraTS20_Training_001_seg.nii
    │       │   └── ...
    │       └── ...
    └── BraTS2020_ValidationData/
        └── MICCAI_BraTS2020_ValidationData/
            └── ...
"""

import argparse
import os
import subprocess
import sys
import zipfile


# ─────────────────────────────────────────────
# Google Drive download
# ─────────────────────────────────────────────

def install_gdown():
    """Cài gdown nếu chưa có."""
    try:
        import gdown  # noqa
    except ImportError:
        print("Cài gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])


def download_from_gdrive(file_id: str, dest: str) -> str:
    """
    Tải file từ Google Drive.
    Hỗ trợ file lớn (>100MB) bằng gdown.
    Returns: đường dẫn file đã tải.
    """
    install_gdown()
    import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"\n📥 Downloading from Google Drive...")
    print(f"   File ID : {file_id}")
    print(f"   Save to : {dest}\n")

    gdown.download(url, dest, quiet=False, fuzzy=True)

    if not os.path.exists(dest):
        raise RuntimeError(
            f"Download thất bại!\n"
            f"Kiểm tra:\n"
            f"  1. FILE_ID đúng chưa? ({file_id})\n"
            f"  2. File đã được share 'Anyone with the link' chưa?\n"
            f"  3. Thử chạy: gdown {url} -O {dest}"
        )
    size_mb = os.path.getsize(dest) / 1e6
    print(f"\n✓ Downloaded: {dest} ({size_mb:.0f} MB)")
    return dest


# ─────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────

def extract_zip(zip_path: str, out_dir: str):
    """Giải nén .zip vào out_dir với progress bar."""
    print(f"\n📦 Extracting {zip_path} → {out_dir}/")
    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        total   = len(members)
        for i, member in enumerate(members, 1):
            zf.extract(member, out_dir)
            if i % 500 == 0 or i == total:
                print(f"  {i}/{total} files extracted...", end='\r')

    print(f"\n✓ Extracted {total} files to {out_dir}/")


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

def validate_dataset(data_dir: str, modality: str = 't1ce') -> bool:
    """Kiểm tra cấu trúc BraTS2020 sau khi giải nén."""
    print(f"\n🔍 Validating dataset structure at {data_dir}...")

    candidates = [
        os.path.join(data_dir, "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData"),
        os.path.join(data_dir, "MICCAI_BraTS2020_TrainingData"),
        data_dir,
    ]

    train_root = None
    for c in candidates:
        if os.path.exists(c):
            subdirs = [d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))]
            if any('BraTS20' in d for d in subdirs):
                train_root = c
                break

    if not train_root:
        print("❌ Không tìm thấy cấu trúc BraTS2020!")
        print(f"   Nội dung {data_dir}:", os.listdir(data_dir)[:10])
        return False

    # Check vài patient
    patients = sorted([
        d for d in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, d)) and 'BraTS20' in d
    ])
    found_files = 0
    for pid in patients[:5]:
        ppath = os.path.join(train_root, pid)
        for ext in ['.nii', '.nii.gz']:
            fp = os.path.join(ppath, f'{pid}_{modality}{ext}')
            if os.path.exists(fp):
                found_files += 1
                break

    print(f"  Training root   : {train_root}")
    print(f"  Patient folders : {len(patients)}")
    print(f"  Sample files    : {found_files}/5 found ({'✓' if found_files > 0 else '✗'})")

    if found_files == 0:
        print(f"\n❌ Không tìm thấy file {modality}!")
        print(f"   Files trong {patients[0]}:", os.listdir(os.path.join(train_root, patients[0]))[:6])
        return False

    print(f"\n✓ Dataset OK — {len(patients)} patients tại: {train_root}")
    print(f"\n📝 Cập nhật configs/ với đường dẫn:")
    print(f"   train_dir: \"{train_root}\"")
    return True


# ─────────────────────────────────────────────
# Auto-update configs
# ─────────────────────────────────────────────

def update_configs(train_root: str, val_root: str = None):
    """Tự động cập nhật đường dẫn trong tất cả yaml configs."""
    import glob
    configs = glob.glob("configs/*.yaml")
    if not configs:
        return

    for cfg_path in configs:
        with open(cfg_path) as f:
            content = f.read()

        # Thay train_dir
        import re
        content = re.sub(
            r'(train_dir:\s*)"[^"]*"',
            f'\\1"{train_root}"',
            content,
        )
        # Thay val_dir nếu có
        if val_root:
            content = re.sub(
                r'(val_dir:\s*)"[^"]*"',
                f'\\1"{val_root}"',
                content,
            )

        with open(cfg_path, 'w') as f:
            f.write(content)

    print(f"\n✓ Updated {len(configs)} config files with correct paths")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Download + extract BraTS2020 from Google Drive'
    )
    p.add_argument('--gdrive_id', type=str, default=None,
                   help='Google Drive file ID (từ link share)')
    p.add_argument('--gdrive_id_val', type=str, default=None,
                   help='Google Drive file ID cho validation data (nếu tách riêng)')
    p.add_argument('--zip_path', type=str, default=None,
                   help='Đường dẫn file .zip đã có sẵn (bỏ qua bước download)')
    p.add_argument('--zip_path_val', type=str, default=None,
                   help='Đường dẫn file .zip validation đã có sẵn')
    p.add_argument('--out_dir', type=str, default='data',
                   help='Thư mục giải nén (default: data/)')
    p.add_argument('--no_auto_config', action='store_true',
                   help='Không tự động cập nhật configs/')
    args = p.parse_args()

    if not args.gdrive_id and not args.zip_path:
        p.print_help()
        print("\n\nVí dụ:")
        print("  # Tải từ Google Drive:")
        print("  python setup_data.py --gdrive_id 1AbCdEfGhIjKlMnOpQrStUv")
        print()
        print("  # Dùng file zip có sẵn:")
        print("  python setup_data.py --zip_path ./BraTS2020.zip")
        sys.exit(0)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Training data ──
    zip_path = args.zip_path
    if args.gdrive_id and not zip_path:
        zip_path = os.path.join(args.out_dir, "BraTS2020_training.zip")
        download_from_gdrive(args.gdrive_id, zip_path)

    if zip_path:
        extract_zip(zip_path, args.out_dir)

    # ── Validation data ──
    zip_val = args.zip_path_val
    if args.gdrive_id_val and not zip_val:
        zip_val = os.path.join(args.out_dir, "BraTS2020_validation.zip")
        download_from_gdrive(args.gdrive_id_val, zip_val)

    if zip_val:
        extract_zip(zip_val, args.out_dir)

    # ── Validate ──
    ok = validate_dataset(args.out_dir)

    # ── Auto-update configs ──
    if ok and not args.no_auto_config:
        # Tìm actual path sau khi giải nén
        candidates_train = [
            os.path.join(args.out_dir, "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData"),
            os.path.join(args.out_dir, "MICCAI_BraTS2020_TrainingData"),
        ]
        candidates_val = [
            os.path.join(args.out_dir, "BraTS2020_ValidationData", "MICCAI_BraTS2020_ValidationData"),
            os.path.join(args.out_dir, "MICCAI_BraTS2020_ValidationData"),
        ]
        train_root = next((c for c in candidates_train if os.path.exists(c)), None)
        val_root   = next((c for c in candidates_val   if os.path.exists(c)), None)
        if train_root:
            update_configs(train_root, val_root)

    print("\n🚀 Sẵn sàng train!")
    print("   python train_simclr.py --config configs/simclr.yaml")


if __name__ == '__main__':
    main()
