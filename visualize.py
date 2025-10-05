import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model.unet import UNet
from lowlight_dataset import NightCitySegmentationDataset, PairedTransform

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
    from collections import Counter
    import numpy as np
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
    checkpoint_path = "/kaggle/input/weight-lowlight/best_model.pth"  # 👉 đổi đúng path tới weight của bạn
    imgdir = "/kaggle/input/night-city-data/night_city/NightCity-image/NightCity-image/val"
    maskdir = "/kaggle/input/night-city-data/night_city/NightCity-label/NightCity-label/label/val"
    
    visualize_predictions(
        checkpoint_path=checkpoint_path,
        imgdir=imgdir,
        maskdir=maskdir,
        num_classes=18,     # bạn đang có 18 lớp (0–17)
        size=256,           # resize ảnh để inference
        num_images=3,       # số ảnh muốn visualize
        output_dir="/kaggle/working/outputs"
    )
