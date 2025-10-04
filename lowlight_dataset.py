# ==============================================================================
# Custom NightCity Dataset
# ==============================================================================
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class NightCitySegmentationDataset(Dataset):
    """
    Dataset tùy chỉnh để tải cặp ảnh ban đêm và Mask Nhãn.
    Sử dụng quy ước đặt tên NightCity: {name}.png -> {name}_labelIds.png
    """
    def __init__(self, root_dir, img_subdir, mask_subdir, transform=None):
        
        self.img_path = os.path.join(root_dir, img_subdir)
        self.mask_path = os.path.join(root_dir, mask_subdir)
        
        # Lấy danh sách tên file ảnh
        self.file_list = sorted(os.listdir(self.img_path))
        
        # Transform cơ bản cho ảnh đầu vào (resizing/cropping nên được thêm vào nếu cần)
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        
        # 1. Xây dựng đường dẫn
        img_path = os.path.join(self.img_path, file_name)
        
        # Tên file nhãn: Thay thế .png bằng _labelIds.png
        base_name, ext = os.path.splitext(file_name)
        label_file_name = f"{base_name}_labelIds{ext}"
        mask_path = os.path.join(self.mask_path, label_file_name) 
        
        # 2. Tải ảnh (RGB)
        night_img = Image.open(img_path).convert('RGB')
        
        # 3. Tải Mask (Grayscale/L-mode)
        # Mask thường là ảnh 1 kênh, giá trị pixel là ID lớp (0-255)
        target_mask = Image.open(mask_path).convert('L') 
        
        # 4. Áp dụng Transform
        night_img = self.transform(night_img)
        
        # Mask: Chuyển sang Tensor (H, W). Không áp dụng normalization.
        target_mask = torch.from_numpy(np.array(target_mask, dtype=np.int64)) 
        
        return night_img, target_mask
    
# ----------------- Hết lowlight_dataset.py -----------------
