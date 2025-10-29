import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

from evaluation import batch_multi_class_metrics, overall_pixel_accuracy
from model.unet import UNet

# Import hai dataset
from lowlight_dataset import NightCitySegmentationDataset, PairedTransform
from ISPRS_Potsdam_dataset import ISPRSDataset


# =======================================================
# 🧩 A. HÀM XỬ LÝ THAM SỐ DÒNG LỆNH
# =======================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình UNet cho bài toán segmentation.")

    # --- Chọn dataset ---
    parser.add_argument('--dataset', type=str, required=True, choices=['lowlight', 'isprs'],
                        help="Chọn bộ dữ liệu để huấn luyện: 'lowlight' hoặc 'isprs'")

    parser.add_argument('--imgdir', type=str, required=True,
                        help='Đường dẫn đến thư mục ảnh huấn luyện (VD: ./images/train)')
    parser.add_argument('--maskdir', type=str, required=True,
                        help='Đường dẫn đến thư mục mask (VD: ./labels/train)')

    # --- Tham số huấn luyện ---
    parser.add_argument('--size', type=int, default=224,
                        help='Kích thước ảnh đầu vào sau khi resize. Mặc định: 224')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Tổng số epoch. Mặc định: 200')
    parser.add_argument('--batchsize', type=int, default=8,
                        help='Kích thước batch. Mặc định: 8')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate. Mặc định: 1e-4')
    parser.add_argument('--numworkers', type=int, default=4,
                        help='Số lượng worker cho DataLoader. Mặc định: 4')

    # --- Tham số dataset (số lớp) ---
    parser.add_argument('--numclasses', type=int, default=None,
                        help='Số lớp nhãn (nếu không truyền, tự động chọn theo dataset)')

    return parser.parse_args()


# =======================================================
# 🚀 B. HÀM HUẤN LUYỆN
# =======================================================
def train_model(args):
    IMG_DIR = args.imgdir
    MASK_DIR = args.maskdir
    TARGET_SIZE = args.size
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batchsize
    LEARNING_RATE = args.lr
    NUM_WORKERS = args.numworkers
    DATASET = args.dataset.lower()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =============================
    # 1️⃣ Chọn Dataset tương ứng
    # =============================
    if DATASET == "lowlight":
        # --- Lowlight NightCity dataset ---
        if args.numclasses is None:
            NUM_CLASSES = 20  # mặc định 20 lớp
        else:
            NUM_CLASSES = args.numclasses

        transform = PairedTransform(size=(TARGET_SIZE, TARGET_SIZE))
        dataset = NightCitySegmentationDataset(
            img_dir=IMG_DIR,
            mask_dir=MASK_DIR,
            transform=transform
        )

    elif DATASET == "isprs":
        # --- ISPRS Potsdam dataset ---
        from ISPRS_Potsdam_dataset import LABEL_MAPPING
        NUM_CLASSES = len(LABEL_MAPPING) if args.numclasses is None else args.numclasses

        dataset = ISPRSDataset(
            img_dir=IMG_DIR,
            mask_dir=MASK_DIR,
            target_size=(TARGET_SIZE, TARGET_SIZE)
        )

    else:
        raise ValueError("Dataset không hợp lệ. Chỉ hỗ trợ: lowlight hoặc isprs.")

    print(f"✅ Dataset: {DATASET.upper()} | Số mẫu: {len(dataset)} | Kích thước: {TARGET_SIZE}x{TARGET_SIZE} | Số lớp: {NUM_CLASSES}")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # =============================
    # 2️⃣ Mô hình, Loss, Optimizer
    # =============================
    model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print(f"\n🚀 Bắt đầu huấn luyện trên {device} | Tổng epoch: {NUM_EPOCHS}\n")

    # =============================
    # 3️⃣ Vòng lặp huấn luyện
    # =============================
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        total_mIoU, total_mDice, total_mAcc, total_overall_acc = 0.0, 0.0, 0.0, 0.0
        num_batches = len(dataloader)

        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            with torch.no_grad():
                metrics = batch_multi_class_metrics(preds, masks, num_classes=NUM_CLASSES)
                total_mIoU += metrics['mIoU']
                total_mDice += metrics['mDice']
                total_mAcc += metrics['mAcc']
                total_overall_acc += overall_pixel_accuracy(preds, masks)

        scheduler.step()

        avg_loss = running_loss / num_batches
        avg_mIoU = total_mIoU / num_batches
        avg_mDice = total_mDice / num_batches
        avg_mAcc = total_mAcc / num_batches
        avg_overall_acc = total_overall_acc / num_batches

        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f} | mIoU: {avg_mIoU:.4f} | "
              f"mDice: {avg_mDice:.4f} | mAcc: {avg_mAcc:.4f} | OverallAcc: {avg_overall_acc:.4f}")

        # === Lưu checkpoint mỗi 20 epoch ===
        if (epoch + 1) % 20 == 0:
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save({'model': model.state_dict()},
                       f'./checkpoints/{DATASET}_epoch_{epoch+1}.pth')


# =======================================================
# 🔰 C. CHẠY CHÍNH
# =======================================================
if __name__ == '__main__':
    args = parse_args()
    train_model(args)
