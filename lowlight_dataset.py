import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

# Giả định các module evaluation và model/unet tồn tại và hoạt động đúng

# ====================================================
# A. ÁNH XẠ NHÃN (LABEL REMAPPING) - ĐÃ CHỈNH SỬA
# ====================================================

# Thông tin nhãn được cung cấp
label_info = {
    7: ("road", (128, 64, 128)),
    8: ("sidewalk", (244, 35, 232)),
    11: ("fence", (190, 153, 153)),
    12: ("pole", (153, 153, 153)),
    13: ("traffic light", (250, 170, 30)),
    17: ("sky", (70, 130, 180)),
    18: ("person", (220, 20, 60)),
    19: ("rider", (255, 0, 0)),
    20: ("car", (0, 0, 142)),
    21: ("truck", (0, 0, 70)),
    22: ("bus", (0, 60, 100)),
    23: ("train", (0, 80, 100)),
    24: ("motorcycle", (0, 0, 230)),
    25: ("bicycle", (119, 11, 32)),
    26: ("barrier", (180, 165, 180)),
    27: ("billboard", (200, 200, 0)),
    28: ("streetlight", (100, 100, 100)),
    31: ("tunnel", (0, 100, 100)),
    32: ("bridge", (50, 50, 50)),
    33: ("building group", (0, 128, 128))
}

# Tạo Bảng ánh xạ từ Original ID sang Train ID (0-19)
# Các Original ID được sắp xếp và gán lại từ 0
ID_MAPPING = {original_id: train_id for train_id, original_id in enumerate(label_info.keys())}
# Kết quả: {7: 0, 8: 1, 11: 2, ..., 33: 19}

# Nhãn cho các ID không hợp lệ/không cần huấn luyện
IGNORE_INDEX = 255 # GIỮ NGUYÊN

def convert_to_train_ids(mask_array):
    """Ánh xạ Original IDs sang Train IDs (0-19) hoặc IGNORE_INDEX (255)."""
    # Tạo một mảng mới với giá trị ignore_index mặc định (bao gồm nhãn 0)
    train_mask = np.full_like(mask_array, IGNORE_INDEX, dtype=np.uint8)

    # Ánh xạ từng ID có ý nghĩa
    for original_id, train_id in ID_MAPPING.items():
        train_mask[mask_array == original_id] = train_id
            
    # Giữ nguyên ID 255 nếu có (đã được Cityscapes gán)
    train_mask[mask_array == 255] = IGNORE_INDEX
    
    return train_mask

# ====================================================
# B. PAIRED TRANSFORM (GIỮ NGUYÊN)
# ====================================================

class PairedTransform:
    """Áp dụng các phép biến đổi giống nhau cho cả ảnh và mask."""
    def __init__(self, size=None):
        # ... (Phần này giữ nguyên)
        if size is not None:
            self.resize = transforms.Resize(size, interpolation=Image.BILINEAR)
            self.resize_mask = transforms.Resize(size, interpolation=Image.NEAREST)
        else:
            self.resize = lambda x: x
            self.resize_mask = lambda x: x
            
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, mask):
        # 1. Resize
        img = self.resize(img)
        mask = self.resize_mask(mask)
        
        # 2. To Tensor (Image: [0, 1])
        img_tensor = self.to_tensor(img)
        
        # 3. Mask to Tensor (ÁP DỤNG ÁNH XẠ)
        mask_array = np.array(mask, dtype=np.uint8)
        
        # Ánh xạ nhãn gốc sang nhãn huấn luyện (0-19 hoặc 255)
        train_id_mask = convert_to_train_ids(mask_array)
        
        # Chuyển đổi sang LongTensor (kích thước [H, W])
        mask_tensor = torch.from_numpy(train_id_mask).long()
        
        return img_tensor, mask_tensor


# ====================================================
# C. DATASET ĐÃ SỬA (GIỮ NGUYÊN)
# ====================================================

class NightCitySegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None): 
        # ... (Phần này giữ nguyên)
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

        # ✅ Nếu không có mask (chế độ test/visualize)
        if self.mask_path is None:
            img_tensor = self.transform.to_tensor(self.transform.resize(night_img))
            dummy_mask = torch.zeros(img_tensor.shape[1:], dtype=torch.long)
            return img_tensor, dummy_mask

        # ✅ Nếu có mask (train/val)
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
            raise FileNotFoundError(
                f"❌ Không tìm thấy mask cho {file_name} trong {self.mask_path}"
            )

        target_mask = Image.open(mask_path).convert('L')
        night_img_tensor, target_mask_tensor = self.transform(night_img, target_mask)
        return night_img_tensor, target_mask_tensor
