import os
import torch
from torch.utils.data import DataLoader
from model.unet import UNet
from evaluation import batch_multi_class_metrics, overall_pixel_accuracy
from lowlight_dataset import NightCitySegmentationDataset, PairedTransform

@torch.no_grad()
def test_model(checkpoint_path, imgdir, maskdir, num_classes=20, size=224, batch_size=4, num_workers=2, device=None):
    """
    Hàm test mô hình segmentation.
    
    Args:
        checkpoint_path (str): Đường dẫn tới file .pth của mô hình.
        imgdir (str): Thư mục ảnh test.
        maskdir (str): Thư mục mask nhãn tương ứng.
        num_classes (int): Số lớp segmentation.
        size (int): Kích thước resize ảnh đầu vào.
        batch_size (int): Batch size khi test.
        num_workers (int): Số worker của DataLoader.
        device (torch.device): Thiết bị (cpu/cuda).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 1. Load mô hình =====
    print(f"🔹 Loading model from: {checkpoint_path}")
    model = UNet(n_channels=3, n_classes=num_classes).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # ===== 2. Load dữ liệu test =====
    test_transform = PairedTransform(size=(size, size))
    test_dataset = NightCitySegmentationDataset(
        img_dir=imgdir,
        mask_dir=maskdir,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"🔹 Loaded {len(test_dataset)} test images.")

    # ===== 3. Biến tích lũy kết quả =====
    total_mIoU, total_mDice, total_mAcc, total_overall_acc = 0.0, 0.0, 0.0, 0.0
    num_batches = len(test_loader)

    # ===== 4. Vòng lặp test =====
    for batch_idx, (images, masks) in enumerate(test_loader):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        metrics = batch_multi_class_metrics(logits, masks, num_classes=num_classes)
        total_mIoU += metrics['mIoU']
        total_mDice += metrics['mDice']
        total_mAcc += metrics['mAcc']
        total_overall_acc += overall_pixel_accuracy(logits, masks)

    # ===== 5. Trung bình toàn tập =====
    avg_mIoU = total_mIoU / num_batches
    avg_mDice = total_mDice / num_batches
    avg_mAcc = total_mAcc / num_batches
    avg_overall_acc = total_overall_acc / num_batches

    print("\n📊 === TEST RESULTS ===")
    print(f"mIoU: {avg_mIoU:.4f}")
    print(f"mDice: {avg_mDice:.4f}")
    print(f"mAcc (Recall): {avg_mAcc:.4f}")
    print(f"Overall Pixel Accuracy: {avg_overall_acc:.4f}")

    return {
        "mIoU": avg_mIoU,
        "mDice": avg_mDice,
        "mAcc": avg_mAcc,
        "OverallAcc": avg_overall_acc
    }


# =====================================================
# Chạy test trực tiếp từ terminal:
# python test.py --checkpoint ./checkpoints/checkpoint_epoch_200.pth --imgdir ./data/test/images --maskdir ./data/test/labels
# =====================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test segmentation model on test dataset.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn tới file .pth của mô hình.")
    parser.add_argument("--imgdir", type=str, required=True, help="Thư mục ảnh test.")
    parser.add_argument("--maskdir", type=str, required=True, help="Thư mục mask test.")
    parser.add_argument("--numclasses", type=int, default=20, help="Số lớp segmentation.")
    parser.add_argument("--size", type=int, default=224, help="Kích thước resize ảnh.")
    parser.add_argument("--batchsize", type=int, default=4, help="Batch size khi test.")
    parser.add_argument("--numworkers", type=int, default=2, help="Số worker DataLoader.")
    args = parser.parse_args()

    test_model(
        checkpoint_path=args.checkpoint,
        imgdir=args.imgdir,
        maskdir=args.maskdir,
        num_classes=args.numclasses,
        size=args.size,
        batch_size=args.batchsize,
        num_workers=args.numworkers
    )
z
