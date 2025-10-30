import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ===============================
# ⚙️ Cấu hình chung
# ===============================
NUM_CLASSES = 8  # LoveDA có 7 lớp thực + 1 background
TARGET_SIZE = (256, 256)
BATCH_SIZE = 4
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 🧠 Dataset LoveDA
# ===============================
class LoveDADataset(Dataset):
    def __init__(self, base_dirs, split="Train", transform=None):
        """
        base_dirs: list chứa các thư mục gốc, ví dụ:
            ["/kaggle/input/remoteloveda-train", "/kaggle/input/remoteloveda-val"]
        split: "Train" hoặc "Val"
        transform: Albumentations transform
        """
        self.img_paths = []
        self.mask_paths = []
        self.transform = transform

        for base_dir in base_dirs:
            for domain in ["Rural", "Urban"]:
                img_dir = os.path.join(base_dir, split, domain, "images_png")
                mask_dir = os.path.join(base_dir, split, domain, "masks_png")

                if not os.path.exists(img_dir):
                    continue

                img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")])
                for f in img_files:
                    img_path = os.path.join(img_dir, f)
                    mask_path = os.path.join(mask_dir, f)
                    if os.path.exists(mask_path):
                        self.img_paths.append(img_path)
                        self.mask_paths.append(mask_path)

        print(f"📂 {split} set: {len(self.img_paths)} ảnh ({' + '.join(['Rural','Urban'])})")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # Đọc ảnh & mask
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize về kích thước chuẩn
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        # Áp dụng augmentations (nếu có)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"].long()
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1) / 255.0).float()
            mask = torch.from_numpy(mask).long()

        return img, mask


