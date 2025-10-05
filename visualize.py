import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from model.unet import UNet
from lowlight_dataset import NightCitySegmentationDataset, PairedTransform

# ===============================
# 🎨 Hàm tô màu cho mask dự đoán
# ===============================
def colorize_mask(mask, num_classes):
    """
    Gán màu ngẫu nhiên cho từng class trong mask để dễ nhìn.
    """
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
def visualize_predictions(checkpoint_path, imgdir, num_classes=20, size=224, num_images=3, device=None, output_dir="outputs"):
    """
    Hiển thị và lưu một số ảnh dự đoán segmentation.

    Args:
        checkpoint_path (str): Đường dẫn model .pth
        imgdir (str): Thư mục ảnh test
        num_classes (int): Số lớp segmentation
        size (int): Kích thước resize ảnh
        num_images (int): Số ảnh hiển thị
        output_dir (str): Thư mục lưu kết quả
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🔹 Loading model from: {checkpoint_path}")
    model = UNet(n_channels=3, n_classes=num_classes).to(device)

    # Load trọng số
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Dataset chỉ cần ảnh (không cần mask)
    test_transform = PairedTransform(size=(size, size))
    dataset = NightCitySegmentationDataset(img_dir=imgdir, mask_dir=None, transform=test_transform)
    print(f"🔹 Loaded {len(dataset)} test images for visualization")

    os.makedirs(output_dir, exist_ok=True)
    indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)

    for idx, i in enumerate(indices):
        image, _ = dataset[i]
        image = image.unsqueeze(0).to(device)

        # Dự đoán
        pred_logits = model(image)
        pred_mask = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()

        # Chuẩn bị ảnh hiển thị
        img_vis = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_vis = np.clip(img_vis, 0, 1)
        colored_mask = colorize_mask(pred_mask, num_classes)

        # Hiển thị
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img_vis)
        plt.title("Ảnh gốc")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(colored_mask)
        plt.title("Mask dự đoán")
        plt.axis("off")

        plt.tight_layout()

        save_path = os.path.join(output_dir, f"prediction_{idx+1}.png")
        plt.savefig(save_path)
        plt.show()

        print(f"💾 Saved visualization: {save_path}")


# ===============================
# 🚀 Chạy bằng tham số dòng lệnh
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hiển thị kết quả segmentation dự đoán.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn file .pth model.")
    parser.add_argument("--imgdir", type=str, required=True, help="Thư mục ảnh test.")
    parser.add_argument("--numclasses", type=int, default=20, help="Số lớp segmentation.")
    parser.add_argument("--size", type=int, default=224, help="Kích thước resize ảnh.")
    parser.add_argument("--numimages", type=int, default=3, help="Số ảnh hiển thị.")
    parser.add_argument("--outputdir", type=str, default="outputs", help="Thư mục lưu ảnh kết quả.")

    args = parser.parse_args()

    visualize_predictions(
        checkpoint_path=args.checkpoint,
        imgdir=args.imgdir,
        num_classes=args.numclasses,
        size=args.size,
        num_images=args.numimages,
        output_dir=args.outputdir
    )
