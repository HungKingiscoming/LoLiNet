import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ====================================================
# A. ÁNH XẠ NHÃN (LABEL REMAPPING)
# ====================================================

# Bảng ánh xạ từ Original ID sang Train ID (0-19)
# Nhãn 0 và các nhãn không có trong danh sách sẽ được gán là IGNORE_INDEX
IGNORE_INDEX = 255

ID_MAPPING = {
    # Các nhãn cần huấn luyện (20 lớp)
    7: 0,       # road
    8: 1,       # sidewalk
    11: 2,      # fence
    12: 3,      # pole
    13: 4,      # traffic light
    17: 5,      # sky
    18: 6,      # person
    19: 7,      # rider
    20: 8,      # car
    21: 9,      # truck
    22: 10,     # bus
    23: 11,     # train
    24: 12,     # motorcycle
    25: 13,     # bicycle
    26: 14,     # barrier
    27: 15,     # billboard
    28: 16,     # streetlight
    31: 17,     # tunnel
    32: 18,     # bridge
    33: 19,     # building group
    
    # Nhãn 0 và các nhãn không cần thiết khác sẽ được ánh xạ tới IGNORE_INDEX=255
    # Bạn chỉ cần đảm bảo rằng tất cả các giá trị không phải là 7, 8, 11,... 33
    # sẽ được xử lý trong hàm convert_to_train_ids bên dưới.
}

def convert_to_train_ids(mask_array):
    """Ánh xạ Original IDs sang Train IDs (0-19) hoặc IGNORE_INDEX (255)."""
    # Tạo một mảng mới với giá trị ignore_index mặc định (bao gồm nhãn 0)
    # Tất cả các pixel sẽ mặc định là 255 trừ khi nó nằm trong ID_MAPPING
    train_mask = np.full_like(mask_array, IGNORE_INDEX, dtype=np.uint8)

    # Ánh xạ từng ID có ý nghĩa
    for original_id, train_id in ID_MAPPING.items():
        train_mask[mask_array == original_id] = train_id
            
    # Giữ nguyên ID 255 nếu có (đã được Cityscapes gán)
    train_mask[mask_array == 255] = IGNORE_INDEX
    
    return train_mask

# ====================================================
# B. PAIRED TRANSFORM (Giữ nguyên cấu trúc, áp dụng ánh xạ)
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
# C. DATASET ĐÃ SỬA (Giữ nguyên cấu trúc đường dẫn đã sửa)
# ====================================================

class NightCitySegmentationDataset(Dataset):
    """
    Dataset tùy chỉnh. Nhận trực tiếp đường dẫn đến thư mục ảnh và thư mục mask.
    """
    def __init__(self, img_dir, mask_dir, transform=None): 
        # Nhận đường dẫn ảnh/mask đã được tính toán ở ngoài
        self.img_path = img_dir
        self.mask_path = mask_dir
        
        # Lấy danh sách file ảnh
        self.file_list = sorted(os.listdir(self.img_path))
        
        if transform is None:
            self.transform = PairedTransform(size=None) 
        else:
            self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        
        img_path = os.path.join(self.img_path, file_name)
        
        # Logic đặt tên nhãn vẫn cần được giữ nguyên để tìm file nhãn tương ứng
        base_name, ext = os.path.splitext(file_name)
        
        # Quy tắc 1: {base_name}_labelIds{ext}
        label_file_name_1 = f"{base_name}_labelIds{ext}"
        mask_path_1 = os.path.join(self.mask_path, label_file_name_1) 

        # Quy tắc 2: Tên file giống tên ảnh
        mask_path_2 = os.path.join(self.mask_path, file_name)
        
        # Quy tắc 3: Quy tắc Cityscapes
        label_file_name_3 = file_name.replace('leftImg8bit', 'gtFine_labelIds')
        mask_path_3 = os.path.join(self.mask_path, label_file_name_3)
        
        # Chọn đường dẫn mask tồn tại
        if os.path.exists(mask_path_1):
            mask_path = mask_path_1
        elif os.path.exists(mask_path_2):
            mask_path = mask_path_2
        elif os.path.exists(mask_path_3):
            mask_path = mask_path_3
        else:
             raise FileNotFoundError(f"Không tìm thấy file nhãn cho {file_name} trong thư mục {self.mask_path} với bất kỳ quy tắc nào.")

        night_img = Image.open(img_path).convert('RGB')
        target_mask = Image.open(mask_path).convert('L') 
        
        night_img_tensor, target_mask_tensor = self.transform(night_img, target_mask)
        
        return night_img_tensor, target_mask_tensor
