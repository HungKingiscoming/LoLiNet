import os
import torch
from torch.utils.data import DataLoader
from model.unet import UNet
from evaluation import batch_multi_class_metrics, overall_pixel_accuracy

# --- Datasets ---
from lowlight_dataset import NightCitySegmentationDataset, PairedTransform
from ISPRS_Potsdam_dataset import ISPRSDataset
from loveDA_dataset import LoveDADataset
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

def get_test_dataset(dataset_name, imgdir, maskdir, size=224):
    dataset_name = dataset_name.lower()
    if dataset_name == "lowlight":
        transform = PairedTransform(size=(size, size))
        dataset = NightCitySegmentationDataset(img_dir=imgdir, mask_dir=maskdir, transform=transform)
        num_classes = 20
    elif dataset_name == "isprs":
        from ISPRS_Potsdam_dataset import LABEL_MAPPING
        dataset = ISPRSDataset(img_dir=imgdir, mask_dir=maskdir, target_size=(size, size))
        num_classes = len(LABEL_MAPPING)
    elif dataset_name == "loveda":
        transform = Compose([
            Resize(size, size),
            Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2()
        ])
        dataset = LoveDADataset(base_dirs=[imgdir], split="Val", transform=transform)
        num_classes = 8
    else:
        raise ValueError("Chỉ hỗ trợ dataset: lowlight, isprs, loveDA")
    
    if len(dataset) == 0:
        raise ValueError(f"Không tìm thấy ảnh test trong {imgdir}")
    
    return dataset, num_classes

@torch.no_grad()
def test_model(checkpoint_path, dataset_name, imgdir, maskdir, size=224, batch_size=4, num_workers=2, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ===== Load model =====
    print(f"🔹 Loading model from: {checkpoint_path}")
    dataset, num_classes = get_test_dataset(dataset_name, imgdir, maskdir, size=size)
    model = UNet(n_channels=3, n_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # ===== DataLoader =====
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    print(f"🔹 Loaded {len(dataset)} test images ({dataset_name.upper()})")

    # ===== Test loop =====
    total_mIoU, total_mDice, total_mAcc, total_overall_acc = 0.0,0.0,0.0,0.0
    num_batches = len(loader)

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        metrics = batch_multi_class_metrics(logits, masks, num_classes=num_classes)
        total_mIoU += metrics['mIoU']
        total_mDice += metrics['mDice']
        total_mAcc += metrics['mAcc']
        total_overall_acc += overall_pixel_accuracy(logits, masks)

    print(f"\n📊 === TEST RESULTS ({dataset_name.upper()}) ===")
    print(f"mIoU: {total_mIoU/num_batches:.4f}")
    print(f"mDice: {total_mDice/num_batches:.4f}")
    print(f"mAcc: {total_mAcc/num_batches:.4f}")
    print(f"OverallAcc: {total_overall_acc/num_batches:.4f}")

    return {
        "mIoU": total_mIoU/num_batches,
        "mDice": total_mDice/num_batches,
        "mAcc": total_mAcc/num_batches,
        "OverallAcc": total_overall_acc/num_batches
    }

# =====================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test UNet segmentation model on multiple datasets.")
    parser.add_argument("--dataset", type=str, required=True, choices=["lowlight","isprs","loveda"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .pth")
    parser.add_argument("--imgdir", type=str, required=True, help="Root image directory for test")
    parser.add_argument("--maskdir", type=str, required=True, help="Root mask directory for test")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--numworkers", type=int, default=2)
    args = parser.parse_args()

    test_model(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        imgdir=args.imgdir,
        maskdir=args.maskdir,
        size=args.size,
        batch_size=args.batchsize,
        num_workers=args.numworkers
    )
