# ==============================================================================
# train_unet.py
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from model import LowLightUNet, SimpleTaskHead, TaskOnlyLoss
from lowlight_dataset import NightCitySegmentationDataset    

# --- Cấu hình Huấn luyện ---
# Sử dụng đường dẫn Kaggle mà bạn cung cấp
ROOT_DATA_DIR = '/kaggle/input/nightcity-data/night_city/'
IMG_SUBDIR = 'NightCity-images/NightCity-images/images/train'
MASK_SUBDIR = 'NightCity-label/NightCity-label/label/train'

BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_CLASSES = 10 # CẦN ĐIỀN SỐ LƯỢNG LỚP CHÍNH XÁC CỦA BẠN (ví dụ: 10, 19, v.v.)

def train_model():
    # Kiểm tra đường dẫn
    if not os.path.exists(os.path.join(ROOT_DATA_DIR, IMG_SUBDIR)):
        print(f"LỖI: Không tìm thấy thư mục ảnh tại {os.path.join(ROOT_DATA_DIR, IMG_SUBDIR)}")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Khởi tạo Mô hình (Backbone và Task Head)
    BASE_CHANNELS = 32 # Tùy chỉnh
    
    backbone = LowLightUNet(
        in_channels=3, 
        base_channels=BASE_CHANNELS, 
        num_stages=4, 
    ).to(device)
    
    task_head = SimpleTaskHead(in_channels=BASE_CHANNELS, num_classes=NUM_CLASSES).to(device)

    # 2. Khởi tạo Dataset và DataLoader
    train_dataset = NightCitySegmentationDataset(
        root_dir=ROOT_DATA_DIR,
        img_subdir=IMG_SUBDIR,
        mask_subdir=MASK_SUBDIR
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    )

    # 3. Khởi tạo Loss và Optimizer
    criterion = TaskOnlyLoss(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(task_head.parameters()), 
        lr=LEARNING_RATE
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    print(f"Bắt đầu huấn luyện Segmentation trên {device}. Tổng số Epoch: {NUM_EPOCHS}")

    # 4. Vòng lặp Huấn luyện Chính
    for epoch in range(NUM_EPOCHS):
        backbone.train() 
        task_head.train()
        running_loss = 0.0

        for batch_idx, (night_image, target_mask) in enumerate(train_loader):
            
            night_image = night_image.to(device)
            # target_mask (B, H, W)
            target_mask = target_mask.to(device) 

            optimizer.zero_grad()

            # --- FORWARD PASS ---
            dec_features = backbone(night_image)
            prediction_logits = task_head(dec_features)

            # --- TÍNH TASK LOSS ---
            # prediction_logits (B, C, H, W) vs target_mask (B, H, W)
            L_Task = criterion(prediction_logits, target_mask)

            # --- BACKWARD PASS ---
            L_Task.backward()       
            optimizer.step()         

            running_loss += L_Task.item()

        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Task Loss: {avg_loss:.4f}")

        # Tùy chọn: Lưu checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                {'backbone': backbone.state_dict(), 'head': task_head.state_dict()}, 
                f'/kaggle/working/checkpoint_epoch_{epoch+1}.pth'
            )

if __name__ == '__main__':
    train_model()