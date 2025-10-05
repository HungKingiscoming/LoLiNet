import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ====================================================
# A. ÁNH XẠ NHÃN GỌN GÀNG (KHÔNG CÓ LABEL NAME)
# ====================================================

# Nhãn cần ánh xạ
ID_MAPPING = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17
}

IGNORE_INDEX = 255  # các nhãn không hợp lệ sẽ bị bỏ qua

def convert_to_train_ids(mask_array):
    """Ánh xạ nhãn gốc sang nhãn train (0-17), các nhãn khác gán IGNORE_INDEX."""
    train_mask = np.full_like(mask_array, IGNORE_INDEX, dtype=np.uint8)
    for original_id, train_id in ID_MAPPING.items():
        train_mask[mask_array == original_id] = train_id
    return train_mask


# ====================================================
# B. PAIRED TRANSFORM (GIỮ NGUYÊN)
# ====================================================

class PairedTransform:
    """Áp dụng các phép biến đổi giống nhau cho cả ảnh và mask."""
    def __init__(self, size=None):
        if size is not None:
            self.resize = transforms.Resize(size, interpolation=Image.BILINEAR)
            self.resize_mask = transforms.Resize(size, interpolation=Image.NEAREST)
        else:
            self.resize = lambda x: x
            self.resize_mask = lambda x: x
            
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, mask):
        # Resize
        img = self.resize(img)
        mask = self.resize_mask(mask)
        
        # To tensor
        img_tensor = self.to_tensor(img)
        
        # Ánh xạ nhãn
        mask_array = np.array(mask, dtype=np.uint8)
        train_id_mask = convert_to_train_ids(mask_array)
        mask_tensor = torch.from_numpy(train_id_mask).long()
        
        return img_tensor, mask_tensor


# ====================================================
# C. DATASET (GIỮ NGUYÊN, NHƯNG DÙNG ÁNH XẠ MỚI)
# ====================================================

class NightCitySegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None): 
        self.img_path = img_dir
        self.mask_path = mask_dir
        self.file_list = sorted(os.listdir(self.img_path))
        self.transform = transform or PairedTransform(size=None)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        img_path = os.path.join(self.img_path, file_name)
        night_img = Image.open(img_path).convert('RGB')

        # ✅ Không có mask (chế độ test)
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
