import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model.unet import UNet
from lowlight_dataset import PairedTransform
from PIL import Image

# ===============================
# Hàm tô màu mask segmentation
# ===============================
def colorize_mask(mask, num_classes):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    cmap = (np.array(plt.cm.tab20.colors[:num_classes]) * 255).astype(np.uint8)
    for c in range(num_classes):
        color_mask[mask == c] = cmap[c]
    color_mask[mask == 255] = [0, 0, 0]  # ignore_index
    return color_mask

# ===============================
# Visualize 1 ảnh
# ===============================
@torch.no_grad()
def visualize_single_image(checkpoint_path, image_path, mask_path=None, num_classes=18, size=256, output_dir="outputs_single", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = UNet(n_channels=3, n_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    transform = PairedTransform(size=(size, size))
    image_tensor, _ = transform(image, image)  # dummy mask
    image_input = image_tensor.unsqueeze(0).to(device)
    
    # Predict
    pred_logits = model(image_input)
    pred_mask = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()
    
    # Hiển thị
    img_vis = np.array(image.resize((size, size))) / 255.0
    if mask_path is not None:
        mask_gt = np.array(Image.open(mask_path).resize((size, size)))
        gt_colored = colorize_mask(mask_gt, num_classes)
    else:
        gt_colored = np.zeros_like(img_vis)
    pred_colored = colorize_mask(pred_mask, num_classes)
    
    # Vẽ 3 ảnh cạnh nhau
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    axes[0].imshow(img_vis)
    axes[0].set_title("Ảnh gốc")
    axes[0].axis("off")
    axes[1].imshow(gt_colored)
    axes[1].set_title("Mask GT")
    axes[1].axis("off")
    axes[2].imshow(pred_colored)
    axes[2].set_title("Mask Dự đoán")
    axes[2].axis("off")
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, os.path.basename(image_path).replace(".jpg", "_pred.png"))
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    
    print(f"💾 Lưu kết quả tại: {save_path}")

# ===============================
# Example usage
# ===============================
if __name__ == "__main__":
    checkpoint_path = "/kaggle/input/weight-lowlight/best_model.pth"
    image_path = "/kaggle/input/night-city-data/night_city/NightCity-label/NightCity-label/images/val/Chicago_0004.jpg"
    mask_path = "/kaggle/input/night-city-data/night_city/NightCity-label/NightCity-label/label/val/Chicago_0004_labelIds.png"
    
    visualize_single_image(checkpoint_path, image_path, mask_path, num_classes=18, size=256)
