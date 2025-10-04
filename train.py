import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse # Import module argparse

# ⚠️ CÁC IMPORT CẦN THIẾT (Giả định nằm trong các file khác)
from model.backbone.unet import UNet # Giả định UNet đã được định nghĩa ở đây
from dataset_loader import NightCitySegmentationDataset, PairedTransform 


# ===============================================
# A. HÀM XỬ LÝ THAM SỐ DÒNG LỆNH
# ===============================================
def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Segmentation UNet cho ảnh thiếu sáng.")
    
    # --- Tham số Bắt buộc / Quan trọng ---
    parser.add_argument('--rootdir', type=str, required=True,
                        help='Đường dẫn gốc đến thư mục chứa dữ liệu (e.g., ./data/night_city_dataset)')
    parser.add_argument('--size', type=int, default=224,
                        help='Kích thước ảnh đầu vào (N x N) sau khi resize. Mặc định: 224')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Tổng số epoch huấn luyện. Mặc định: 50')
    parser.add_argument('--batchsize', type=int, default=8,
                        help='Kích thước batch. Mặc định: 8')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Tốc độ học (Learning Rate) ban đầu. Mặc định: 1e-4')
    parser.add_argument('--numclasses', type=int, default=20,
                        help='Số lượng lớp nhãn cho bài toán Segmentation. Mặc định: 20')
                        
    # --- Tham số Tùy chọn ---
    parser.add_argument('--imgsubdir', type=str, default='images',
                        help='Tên thư mục con chứa ảnh đầu vào. Mặc định: images')
    parser.add_argument('--masksubdir', type=str, default='labels',
                        help='Tên thư mục con chứa mask nhãn. Mặc định: labels')
    parser.add_argument('--numworkers', type=int, default=4,
                        help='Số lượng worker cho DataLoader. Mặc định: 4')
                        
    return parser.parse_args()


# ===============================================
# B. HÀM HUẤN LUYỆN CHÍNH
# ===============================================

def train_model(args):
    
    # Gán tham số từ args
    ROOT_DATA_DIR = args.rootdir
    TARGET_SIZE = args.size
    NUM_CLASSES = args.numclasses
    BATCH_SIZE = args.batchsize
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    IMG_SUBDIR = args.imgsubdir
    MASK_SUBDIR = args.masksubdir
    NUM_WORKERS = args.numworkers
    
    # Kiểm tra đường dẫn
    if not os.path.exists(os.path.join(ROOT_DATA_DIR, IMG_SUBDIR)):
        print(f"LỖI: Không tìm thấy thư mục ảnh tại {os.path.join(ROOT_DATA_DIR, IMG_SUBDIR)}")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Khởi tạo Mô hình (UNet tùy chỉnh)
    model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(device) 

    # 2. Khởi tạo Dataset và DataLoader
    data_transform = PairedTransform(size=(TARGET_SIZE, TARGET_SIZE)) 
    
    train_dataset = NightCitySegmentationDataset(
        root_dir=ROOT_DATA_DIR,
        img_subdir=IMG_SUBDIR,
        mask_subdir=MASK_SUBDIR,
        transform=data_transform
    )
    
    print(f"Kích thước đầu vào: {TARGET_SIZE}x{TARGET_SIZE} | Số lượng mẫu: {len(train_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 3. Khởi tạo Loss và Optimizer
    criterion = nn.CrossEntropyLoss().to(device) 
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    print(f"Bắt đầu huấn luyện Segmentation trên {device}. Tổng số Epoch: {NUM_EPOCHS}")

    # 4. Vòng lặp Huấn luyện Chính
    for epoch in range(NUM_EPOCHS):
        model.train() 
        running_loss = 0.0

        for batch_idx, (night_image, target_mask) in enumerate(train_loader):
            
            night_image = night_image.to(device)
            target_mask = target_mask.to(device) 

            optimizer.zero_grad()
            prediction_logits = model(night_image) 
            L_Task = criterion(prediction_logits, target_mask)

            L_Task.backward()      
            optimizer.step()        

            running_loss += L_Task.item()

        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f}")

        # Tùy chọn: Lưu checkpoint
        if (epoch + 1) % 10 == 0:
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save(
                {'model': model.state_dict()}, 
                f'./checkpoints/checkpoint_epoch_{epoch+1}.pth'
            )

if __name__ == '__main__':
    args = parse_args()
    train_model(args)
