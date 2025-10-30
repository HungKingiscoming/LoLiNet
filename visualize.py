import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model.unet import UNet
from collections import Counter
import argparse
from PIL import Image

# Datasets
from lowlight_dataset import NightCitySegmentationDataset, PairedTransform
from ISPRS_Potsdam_dataset import ISPRSDataset
from loveDA_dataset import LoveDADataset
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# ===============================
# 🎨 Hàm tô màu mask segmentation
# ===============================
def colorize_mask(mask, num_classes):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    cmap = (np.array(plt.cm.tab20.colors[:num_classes]) * 255).astype(np.uint8)
    for c in range(num_classes):
        color_mask[mask == c] = cmap[c]
    color_mask[mask == 255] = [0,0,0]  # ignore index
    return color_mask

# ===============================
# Lấy dataset theo tên
# ===============================
def get_dataset(dataset_name, imgdir, maskdir, size):
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
        raise ValueError("Chỉ hỗ trợ dataset: lowlight, isprs, loveda")
    return dataset, num_classes

# ===============================
# 🖼️ Visualize predictions
# ===============================
@torch.no_grad()
def visualize_predictions(checkpoint_path, dataset_name, imgdir, maskdir,
                          size=224, num_images=3, output_dir="outputs", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🔹 Loading model from: {checkpoint_path}")
    dataset, num_classes = get_dataset(dataset_name, imgdir, maskdir, size)
    model = UNet(n_channels=3, n_classes=num_classes).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)

    for idx, i in enumerate(indices):
        image, mask_true = dataset[i]
        image = image.unsqueeze(0).to(device)

        pred_logits = model(image)
        pred_mask = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()

        img_vis = image.squeeze(0).permute(1,2,0).cpu().numpy()
        img_vis = np.clip(img_vis, 0, 1)
        gt_mask = mask_true.cpu().numpy()

        gt_colored = colorize_mask(gt_mask, num_classes)
        pred_colored = colorize_mask(pred_mask, num_classes)

        fig, axes = plt.subplots(1,3, figsize=(15,5))
        axes[0].imshow(img_vis); axes[0].set_title("Ảnh gốc"); axes[0].axis("off")
        axes[1].imshow(gt_colored); axes[1].set_title("Mask GT"); axes[1].axis("off")
        axes[2].imshow(pred_colored); axes[2].set_title("Mask Dự đoán"); axes[2].axis("off")
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{dataset_name}_visual_{idx+1}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()
        print(f"💾 Saved visualization to: {save_path}")

# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize segmentation predictions on multiple datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=["lowlight","isprs","loveda"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth model file")
    parser.add_argument("--imgdir", type=str, required=True, help="Root image directory")
    parser.add_argument("--maskdir", type=str, required=True, help="Root mask directory")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/outputs")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    visualize_predictions(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        imgdir=args.imgdir,
        maskdir=args.maskdir,
        size=args.size,
        num_images=args.num_images,
        output_dir=args.output_dir,
        device=args.device
    )
