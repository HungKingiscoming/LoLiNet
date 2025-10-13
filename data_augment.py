import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random

# ====================================================
# A. ÁNH XẠ NHÃN
# ====================================================
ID_MAPPING = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
    25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17
}
IGNORE_INDEX = 255

def convert_to_train_ids(mask_array):
    train_mask = np.full_like(mask_array, IGNORE_INDEX, dtype=np.uint8)
    for original_id, train_id in ID_MAPPING.items():
        train_mask[mask_array == original_id] = train_id
    return train_mask


# ====================================================
# B. PAIRED TRANSFORM (THÊM TĂNG CƯỜNG DỮ LIỆU)
# ====================================================

class PairedTransform:
    """Áp dụng các phép biến đổi giống nhau cho cả ảnh và mask (có augment)."""
    def __init__(self, size=(512, 512), augment=True):
        self.augment = augment

        # Resize
        if size is not None:
            self.resize = transforms.Resize(size, interpolation=Image.BILINEAR)
            self.resize_mask = transforms.Resize(size, interpolation=Image.NEAREST)
        else:
            self.resize = lambda x: x
            self.resize_mask = lambda x: x

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __call__(self, img, mask):
        # Resize
        img = self.resize(img)
        mask = self.resize_mask(mask)

        # ----------- AUGMENTATION -----------
        if self.augment:
            # Horizontal flip
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            # Random rotation (-10° đến +10°)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                img = img.rotate(angle, resample=Image.BILINEAR)
                mask = mask.rotate(angle, resample=Image.NEAREST)

            # Color jitter
            if random.random() > 0.5:
                color_jitter = transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
                )
                img = color_jitter(img)

            # Random crop (với xác suất nhỏ)
            if random.random() > 0.7:
                i, j, h, w = transforms.RandomCrop.get_params(
                    img, output_size=(int(img.height * 0.9), int(img.width * 0.9))
                )
                img = transforms.functional.crop(img, i, j, h, w)
                mask = transforms.functional.crop(mask, i, j, h, w)
                # Resize lại về kích thước gốc
                img = self.resize(img)
                mask = self.resize_mask(mask)
        # ------------------------------------

        # To tensor & normalize
        img_tensor = self.to_tensor(img)
        img_tensor = self.normalize(img_tensor)

        # Ánh xạ nhãn
        mask_array = np.array(mask, dtype=np.uint8)
        train_id_mask = convert_to_train_ids(mask_array)
        mask_tensor = torch.from_numpy(train_id_mask).long()

        return img_tensor, mask_tensor


# ====================================================
# C. DATASET
# ====================================================

class NightCitySegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None):
        self.img_path = img_dir
        self.mask_path = mask_dir
        self.file_list = sorted(os.listdir(self.img_path))
        self.transform = transform or PairedTransform(size=(512, 512), augment=False)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        img_path = os.path.join(self.img_path, file_name)
        night_img = Image.open(img_path).convert('RGB')

        # ✅ Không có mask (test mode)
        if self.mask_path is None:
            img_tensor = self.transform.to_tensor(self.transform.resize(night_img))
            dummy_mask = torch.zeros(img_tensor.shape[1:], dtype=torch.long)
            return img_tensor, dummy_mask

        # ✅ Có mask (train/val)
        base_name, ext = os.path.splitext(file_name)
        candidates = [
            f"{base_name}_labelIds{ext}",
            file_name,
            file_name.replace("leftImg8bit", "gtFine_labelIds")
        ]

        mask_path = None
        for cand in candidates:
            full_path = os.path.join(self.mask_path, cand)
            if os.path.exists(full_path):
                mask_path = full_path
                break

        if mask_path is None:
            raise FileNotFoundError(f"❌ Không tìm thấy mask cho {file_name} trong {self.mask_path}")

        target_mask = Image.open(mask_path).convert('L')
        night_img_tensor, target_mask_tensor = self.transform(night_img, target_mask)
        return night_img_tensor, target_mask_tensor
