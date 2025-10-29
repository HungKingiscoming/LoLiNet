import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
    def __init__(self, img_dir, mask_dir=None, transform=None):
        """
        Args:
            img_dir (str): Thư mục ảnh gốc
            mask_dir (str): Thư mục chứa mask tương ứng (None nếu chỉ inference)
            transform (albumentations.Compose): Các phép augment (resize, flip, normalize,...)
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.file_list = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        img_path = os.path.join(self.img_dir, fname)

        # Đọc ảnh RGB
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"❌ Không thể đọc ảnh: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # === Chế độ inference: không có mask ===
        if self.mask_dir is None:
            if self.transform:
                transformed = self.transform(image=img)
                img_tensor = transformed['image']
            else:
                img_tensor = torch.from_numpy(img.transpose(2, 0, 1) / 255.).float()
            dummy_mask = torch.zeros(TARGET_SIZE, dtype=torch.long)
            return img_tensor, dummy_mask

        # === Chế độ train/val: có mask ===
        # Giả định tên mask thay “Image” bằng “Label”
        mask_path = os.path.join(self.mask_dir, fname.replace("Image", "Label"))
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"❌ Không tìm thấy mask cho ảnh: {fname}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"❌ Không thể đọc mask: {mask_path}")

        # Resize (nếu transform không có resize)
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        mask = remap_labels(mask)

        # Augmentation bằng Albumentations
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
            if mask.dtype != torch.long:
                mask = mask.long()
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1) / 255.).float()
            mask = torch.from_numpy(mask).long()

        return img, mask
