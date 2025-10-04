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
    """Áp dụng các phép biến đổi giống nhau cho cả ảnh và mask."""
    def __init__(self, size=None):
        if size is not None:
            # BILINEAR cho ảnh (RGB)
            self.resize = transforms.Resize(size, interpolation=Image.BILINEAR)
            # NEAREST cho nhãn (để tránh làm nhòe chỉ số lớp)
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
        
        # 3. Mask to Tensor (GIỮ NGUYÊN Original ID)
        # Chuyển mask sang NumPy array
        mask_array = np.array(mask, dtype=np.uint8) 
        
        # Chuyển đổi sang LongTensor (kích thước [H, W])
        # Torch sẽ sử dụng các giá trị 0, 3, 7, ..., 33 làm chỉ số lớp.
        mask_tensor = torch.from_numpy(mask_array).long()
        
        return img_tensor, mask_tensor


# ====================================================
# B. DATASET ĐÃ SỬA (Chỉ điều chỉnh đường dẫn)
# ====================================================

class NightCitySegmentationDataset(Dataset):
    """
    Dataset tùy chỉnh để tải cặp ảnh ban đêm và Mask Nhãn, 
    sử dụng đường dẫn cụ thể của bạn và không ánh xạ nhãn.
    """
    def __init__(self, root_dir, img_subdir, mask_subdir, transform=None): 
        # Cấu trúc thư mục cụ thể dựa trên thông tin bạn cung cấp:
        # root_dir: /kaggle/input/night-city-data/night_city
        # img_subdir ('train'): /kaggle/input/night-city-data/night_city/NightCity-images/images/train
        # mask_subdir ('train'): /kaggle/input/night-city-data/night_city/NightCity-label/NightCity-label/label/train
        
        # Xây dựng đường dẫn đầy đủ
        self.img_path = os.path.join(root_dir, 'NightCity-images/images', img_subdir)
        self.mask_path = os.path.join(root_dir, 'NightCity-label/NightCity-label/label', mask_subdir)
        
        self.file_list = sorted(os.listdir(self.img_path))
        
        # Khởi tạo PairedTransform nếu chưa có
        if transform is None:
            self.transform = PairedTransform(size=None) 
        else:
            self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        
        # 1. Xây dựng đường dẫn
        img_path = os.path.join(self.img_path, file_name)
        
        # Giả định quy tắc đặt tên file nhãn (như trong code gốc của bạn)
        base_name, ext = os.path.splitext(file_name)
        
        # Cần tìm quy tắc đổi tên file nhãn chính xác. Dùng quy tắc đã có:
        label_file_name = f"{base_name}_labelIds{ext}"
        mask_path = os.path.join(self.mask_path, label_file_name) 

        # Thử lại nếu không tìm thấy, vì bạn có thể đã sửa tên file
        if not os.path.exists(mask_path):
             # Thử với tên file tương tự ảnh nhưng không có hậu tố _labelIds
            label_file_name_alt = file_name 
            mask_path_alt = os.path.join(self.mask_path, label_file_name_alt)
            if os.path.exists(mask_path_alt):
                mask_path = mask_path_alt
            else:
                # Thử quy tắc của Cityscapes (nếu tên file ảnh kết thúc bằng '..._leftImg8bit.png')
                label_file_name_cityscapes = file_name.replace('leftImg8bit', 'gtFine_labelIds')
                mask_path_cityscapes = os.path.join(self.mask_path, label_file_name_cityscapes)
                
                if os.path.exists(mask_path_cityscapes):
                     mask_path = mask_path_cityscapes
                else:
                    raise FileNotFoundError(f"Không tìm thấy file nhãn cho {file_name} trong thư mục {self.mask_path} với bất kỳ quy tắc nào.")

        # 2. Tải ảnh (RGB)
        night_img = Image.open(img_path).convert('RGB')
        # 3. Tải Mask (Grayscale/L-mode)
        target_mask = Image.open(mask_path).convert('L') 
        
        # 4. Gọi PairedTransform (không ánh xạ nhãn)
        night_img_tensor, target_mask_tensor = self.transform(night_img, target_mask)
        
        return night_img_tensor, target_mask_tensor
