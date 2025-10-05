import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from model.unet import UNet
from lowlight_dataset import NightCitySegmentationDataset, PairedTransform

# ===============================
# 🎨 Hàm tô màu mask segmentation
# ===============================
def colorize_mask(mask, num_classes):
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for c in range(num_classes):
        color_mask[mask == c] = colors[c]
    return color_mask


# ===============================
# 🖼️ Hàm visualize dự đoán
# ===============================
@torch.no_grad()
def visualize_predictions(checkpoint_path, imgdir, maskdir, num_classes=20, size=224, num_images=3, output_dir="outputs", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🔹 Loading model from: {checkpoint_path}")
    model = UNet(n_channels=3, n_classes=num_classes).to(device)

    # Load weight
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Dataset (có cả ảnh và mask)
    test_transform = PairedTransform(size=(size, size))
    dataset = NightCitySegmentationDataset(img_dir=imgdir, mask_dir=maskdir, transform=test_transform)
    print(f"🔹 Loaded {len(dataset)} test images for visualization")

    os.makedirs(output_dir, exist_ok=True)
    indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)

    for idx, i in enumerate(indices):
        image, mask_true = dataset[i]
        image = image.unsqueeze(0).to(device)

        # Predict
        pred_logits = model(image)
        pred_mask = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()

        # Convert sang numpy để hiển thị
        img_vis = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_vis = np.clip(img_vis, 0, 1)
        gt_mask = mask_true.cpu().numpy()

        # Tô màu
        gt_colored = colorize_mask(gt_mask, num_classes)
        pred_colored = colorize_mask(pred_mask, num_classes)

        # Hiển thị 3 ảnh cạnh nhau
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(img_vis)
        axes[0].set_title("Ảnh gốc")
        axes[0].axis("off")

        axes[1].imshow(gt_colored)
        axes[1].set_title("Mask Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred_colored)
        axes[2].set_title("Mask Dự đoán")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
        save_path = os.path.join(output_dir, f"visual_{idx+1}.png")
        plt.savefig(save_path, bbox_inches="tight")

        print(f"💾 Saved visualization to: {save_path}")


# ===============================
# 🚀 Chạy bằng dòng lệnh (CLI)
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hiển thị và lưu kết quả segmentation.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn tới file .pth model.")
    parser.add_argument("--imgdir", type=str, required=True, help="Thư mục ảnh test.")
    parser.add_argument("--maskdir", type=str, required=True, help="Thư mục ground truth masks.")
    parser.add_argument("--numclasses", type=int, default=20, help="Số lớp segmentation.")
    parser.add_argument("--size", type=int, default=224, help="Kích thước resize ảnh.")
    parser.add_argument("--numimages", type=int, default=3, help="Số ảnh hiển thị.")
    parser.add_argument("--outputdir", type=str, default="outputs", help="Thư mục lưu ảnh kết quả.")

    args = parser.parse_args()

    visualize_predictions(
        checkpoint_path=args.checkpoint,
        imgdir=args.imgdir,
        maskdir=args.maskdir,
        num_classes=args.numclasses,
        size=args.size,
        num_images=args.numimages,
        output_dir=args.outputdir
    )
