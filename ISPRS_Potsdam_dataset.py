import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ====================================================
# ⚙️ 1️⃣ CẤU HÌNH NHÃN & BIẾN TOÀN CỤC
# ====================================================
LABEL_MAPPING = {0: 0, 29: 1, 76: 2, 150: 3, 179: 4, 226: 5}
IGNORE_INDEX = [255]
NUM_CLASSES = len(LABEL_MAPPING)
TARGET_SIZE = (256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====================================================
# 🧩 2️⃣ HÀM ÁNH XẠ NHÃN
# ====================================================
def remap_labels(mask: np.ndarray) -> np.ndarray:
    """
    Chuyển nhãn gốc (RGB hoặc ID) thành nhãn huấn luyện (0–5),
    và gán 255 cho pixel bị bỏ qua.
    """
    new_mask = np.full_like(mask, 255, dtype=np.uint8)
    for old_id, new_id in LABEL_MAPPING.items():
        new_mask[mask == old_id] = new_id
    for ign in IGNORE_INDEX:
        new_mask[mask == ign] = 255
    return new_mask


# ====================================================
# 🧠 3️⃣ CLASS DATASET CHUẨN HÓA
# ====================================================
class ISPRSDataset(Dataset):
    def __init__(self, img_dir, mask_dir, target_size=(256, 256), transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # ⚙️ Kiểm tra mask đọc được không
        if mask is None:
            raise FileNotFoundError(f"Không thể đọc mask: {mask_path}")

        # ⚙️ Resize
        if self.target_size is not None:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # ⚙️ Remap labels
        mask = remap_labels(mask)

        # ⚙️ Normalize & tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1))  # (C,H,W)
        mask = torch.from_numpy(mask.astype(np.int64))

        return image, mask
