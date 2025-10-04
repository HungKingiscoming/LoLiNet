import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse # Import module argparse
from evaluation import calculate_multi_class_metrics, overall_pixel_accuracy 
from model.unet import UNet # Giả định UNet đã được định nghĩa ở đây
from lowlight_dataset import NightCitySegmentationDataset, PairedTransform 


# ===============================================
# A. HÀM XỬ LÝ THAM SỐ DÒNG LỆNH
# ===============================================
def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Segmentation UNet cho ảnh thiếu sáng.")
    
    # THAY THẾ rootdir, imgsubdir, masksubdir bằng đường dẫn trực tiếp
    parser.add_argument('--imgdir', type=str, required=True,
                        help='Đường dẫn TRỰC TIẾP đến thư mục chứa ảnh huấn luyện (ví dụ: .../images/train)')
    parser.add_argument('--maskdir', type=str, required=True,
                        help='Đường dẫn TRỰC TIẾP đến thư mục chứa mask nhãn (ví dụ: .../labels/train)')
    
    # --- Tham số Bắt buộc / Quan trọng ---
    parser.add_argument('--size', type=int, default=224,
                        help='Kích thước ảnh đầu vào (N x N) sau khi resize. Mặc định: 224')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Tổng số epoch huấn luyện. Mặc định: 200')
    parser.add_argument('--batchsize', type=int, default=8,
                        help='Kích thước batch. Mặc định: 8')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Tốc độ học (Learning Rate) ban đầu. Mặc định: 1e-4')
    parser.add_argument('--numclasses', type=int, default=20,
                        help='Số lượng lớp nhãn (sau khi ánh xạ). Mặc định: 20')
                        
    # --- Tham số Tùy chọn ---
    parser.add_argument('--numworkers', type=int, default=4,
                        help='Số lượng worker cho DataLoader. Mặc định: 4')
                        
    return parser.parse_args()


# ===============================================
# B. HÀM HUẤN LUYỆN CHÍNH
# ===============================================

def train_model(args):
    
    # Gán tham số từ args
    IMG_DIR = args.imgdir
    MASK_DIR = args.maskdir
    TARGET_SIZE = args.size
    NUM_CLASSES = args.numclasses
    BATCH_SIZE = args.batchsize
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    NUM_WORKERS = args.numworkers
    
    # Kiểm tra đường dẫn TRỰC TIẾP
    if not os.path.exists(IMG_DIR):
        print(f"LỖI: Không tìm thấy thư mục ảnh tại {IMG_DIR}")
        return
    if not os.path.exists(MASK_DIR):
        print(f"LỖI: Không tìm thấy thư mục nhãn tại {MASK_DIR}")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Khởi tạo Mô hình
    model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(device) 

    # 2. Khởi tạo Dataset và DataLoader
    data_transform = PairedTransform(size=(TARGET_SIZE, TARGET_SIZE)) 
    
    # Truyền đường dẫn ảnh và nhãn TRỰC TIẾP
    train_dataset = NightCitySegmentationDataset(
        img_dir=IMG_DIR,
        mask_dir=MASK_DIR,
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
    IGNORE_INDEX = 255 # Do ánh xạ nhãn, nhãn bỏ qua là 255
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX).to(device) 
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    print(f"Bắt đầu huấn luyện Segmentation trên {device}. Tổng số Epoch: {NUM_EPOCHS}")

    # 4. Vòng lặp Huấn luyện Chính (Giữ nguyên logic tính toán Loss và Metrics)
    for epoch in range(NUM_EPOCHS):
        model.train() 
        running_loss = 0.0
        
        total_mIoU, total_mDice, total_mAcc, total_overall_acc = 0.0, 0.0, 0.0, 0.0
        num_batches = len(train_loader)

        for batch_idx, (night_image, target_mask) in enumerate(train_loader):
            
            night_image = night_image.to(device)
            target_mask = target_mask.to(device) 

            optimizer.zero_grad()
            prediction_logits = model(night_image) 
            
            L_Task = criterion(prediction_logits, target_mask)

            L_Task.backward()      
            optimizer.step()       

            running_loss += L_Task.item()
            
            # === Tính toán Metrics ===
            with torch.no_grad():
                metrics = calculate_multi_class_metrics(prediction_logits, target_mask, NUM_CLASSES)
                total_mIoU += metrics['mIoU']
                total_mDice += metrics['mDice']
                total_mAcc += metrics['mAcc']
                
                overall_acc = overall_pixel_accuracy(prediction_logits, target_mask)
                total_overall_acc += overall_acc


        scheduler.step()

        avg_loss = running_loss / num_batches
        avg_mIoU = total_mIoU / num_batches
        avg_mDice = total_mDice / num_batches
        avg_mAcc = total_mAcc / num_batches
        avg_overall_acc = total_overall_acc / num_batches
        
        # === In kết quả Epoch ===
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | mIoU: {avg_mIoU:.4f} | mDice: {avg_mDice:.4f} | mAcc: {avg_mAcc:.4f} | Overall Acc: {avg_overall_acc:.4f}")

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
