import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model.unet import UNet
from lowlight_dataset import NightCitySegmentationDataset, PairedTransform
from collections import Counter
import numpy as np
import argparse
from PIL import Image
# ===============================
# 🎨 Hàm tô màu mask segmentation
# ===============================
def colorize_mask(mask, num_classes):
    """Tô màu cho mask segmentation, bỏ qua vùng ignore_index."""
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Tạo bảng màu
    cmap = (np.array(plt.cm.tab20.colors[:num_classes]) * 255).astype(np.uint8)

    # Tô màu cho từng class, bỏ qua vùng ignore_index = 255
    for c in range(num_classes):
        color_mask[mask == c] = cmap[c]

    # Giữ vùng ignore_index là màu đen
    color_mask[mask == 255] = [0, 0, 0]
    return color_mask


# ===============================
# 🖼️ Hàm visualize dự đoán
# ===============================
@torch.no_grad()
def visualize_predictions(
    checkpoint_path,
    imgdir,
    maskdir,
    num_classes=20,
    size=224,
    num_images=3,
    output_dir="outputs",
    device=None,
):
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

    # Dataset
    test_transform = PairedTransform(size=(size, size))
    dataset = NightCitySegmentationDataset(img_dir=imgdir, mask_dir=maskdir, transform=test_transform)
    print(f"🔹 Loaded {len(dataset)} test images for visualization")

    mask = np.array(Image.open("/kaggle/input/night-city-data/night_city/NightCity-label/NightCity-label/label/val/Chicago_0004_labelIds.png"))
    counts = Counter(mask.flatten())
    print(counts)

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
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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

        # ✅ Lưu trước khi hiển thị
        save_path = os.path.join(output_dir, f"visual_{idx+1}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()

        print(f"💾 Saved visualization to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize segmentation predictions")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path tới file weight .pth (ví dụ: /kaggle/input/weight-lowlight/best_model.pth)")
    parser.add_argument("--imgdir", type=str, required=True,
                        help="Thư mục chứa ảnh input")
    parser.add_argument("--maskdir", type=str, required=True,
                        help="Thư mục chứa ground truth mask")
    parser.add_argument("--num_classes", type=int, default=18,
                        help="Số lớp segmentation (mặc định = 18)")
    parser.add_argument("--size", type=int, default=256,
                        help="Kích thước resize ảnh đầu vào")
    parser.add_argument("--num_images", type=int, default=3,
                        help="Số ảnh muốn visualize (mặc định = 3)")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/outputs",
                        help="Thư mục lưu ảnh output")
    parser.add_argument("--device", type=str, default=None,
                        help="Thiết bị chạy: cuda hoặc cpu")

    args = parser.parse_args()
    
    visualize_predictions(
        checkpoint_path=args.checkpoint,
        imgdir=args.imgdir,
        maskdir=args.maskdir,
        num_classes=args.num_classes,
        size=args.size,
        num_images=args.num_images,
        output_dir=args.output_dir,
        device=args.device,
    )
