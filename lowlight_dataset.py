# ==============================================================================
# Custom NightCity Dataset
# ==============================================================================
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class PairedTransform:
    def __init__(self, size=None):
        
        # Chỉ thêm Resize nếu size được cung cấp
        geometric_list = [transforms.RandomHorizontalFlip(p=0.5)]
        if size is not None:
            geometric_list.insert(0, transforms.Resize(size)) 
            
        self.geometric_transforms = transforms.Compose(geometric_list)
        self.image_to_tensor = transforms.ToTensor()

    def __call__(self, img, mask):
        # 1. Apply Geometric Transforms (on PIL images)
        img = self.geometric_transforms(img)
        mask = self.geometric_transforms(mask)
        
        # 2. Convert to Tensor/Numpy
        img = self.image_to_tensor(img) 
        # Chuyển mask (đã biến đổi) sang Tensor (H, W)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64)) 
        
        return img, mask
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


# (Giữ nguyên class PairedTransform ở đây)
class PairedTransform:
    def __init__(self, size=None):
        
        # Chỉ thêm Resize nếu size được cung cấp
        geometric_list = [transforms.RandomHorizontalFlip(p=0.5)]
        if size is not None:
            geometric_list.insert(0, transforms.Resize(size)) 
            
        self.geometric_transforms = transforms.Compose(geometric_list)
        self.image_to_tensor = transforms.ToTensor()

    def __call__(self, img, mask):
        # 1. Apply Geometric Transforms (on PIL images)
        img = self.geometric_transforms(img)
        mask = self.geometric_transforms(mask)
        
        # 2. Convert to Tensor/Numpy
        img = self.image_to_tensor(img) 
        mask = torch.from_numpy(np.array(mask, dtype=np.int64)) 
        
        return img, mask


class NightCitySegmentationDataset(Dataset):
    """
    Dataset tùy chỉnh để tải cặp ảnh ban đêm và Mask Nhãn.
    """
    def __init__(self, root_dir, img_subdir, mask_subdir, transform=None): 
        self.img_path = os.path.join(root_dir, img_subdir)
        self.mask_path = os.path.join(root_dir, mask_subdir)
        self.file_list = sorted(os.listdir(self.img_path))
        
        # LOGIC ĐÃ SỬA: Nếu không có transform nào được truyền vào, 
        # TỰ KHỞI TẠO (INSTANTIATE) PairedTransform
        if transform is None:
            # Khởi tạo instance của PairedTransform khi không truyền đối số
            self.transform = PairedTransform(size=None) 
        else:
            self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        # 1. Xây dựng đường dẫn
        img_path = os.path.join(self.img_path, file_name)
        base_name, ext = os.path.splitext(file_name)
        label_file_name = f"{base_name}_labelIds{ext}"
        mask_path = os.path.join(self.mask_path, label_file_name) 
        # 2. Tải ảnh (RGB)
        night_img = Image.open(img_path).convert('RGB')
        # 3. Tải Mask (Grayscale/L-mode)
        target_mask = Image.open(mask_path).convert('L') 
        
        # 4. Gọi PairedTransform (là một instance)
        return self.transform(night_img, target_mask)

