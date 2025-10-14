import os
import torch
import cv2
import numpy as np
from PIL import Image
from model.unet import UNet
from lowlight_dataset import PairedTransform

# ===============================
# Tô màu mask segmentation
# ===============================
def colorize_mask(mask, num_classes):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    cmap = (np.array(plt.cm.tab20.colors[:num_classes]) * 255).astype(np.uint8)
    for c in range(num_classes):
        color_mask[mask == c] = cmap[c]
    color_mask[mask == 255] = [0, 0, 0]  # ignore_index
    return color_mask

# ===============================
# Predict 1 frame
# ===============================
@torch.no_grad()
def predict_frame(model, frame, num_classes=18, size=256, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = PairedTransform(size=(size, size))
    image_tensor, _ = transform(image, image)
    image_input = image_tensor.unsqueeze(0).to(device)
    
    pred_logits = model(image_input)
    pred_mask = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()
    pred_colored = colorize_mask(pred_mask, num_classes)
    
    # Resize mask về size gốc của frame
    pred_colored = cv2.resize(pred_colored, (frame.shape[1], frame.shape[0]))
    return pred_colored

# ===============================
# Predict video
# ===============================
def predict_video(checkpoint_path, input_video_path, output_video_path,
                  num_classes=18, size=256, device=None):
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
    
    # Mở video
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🔹 Tổng số frame: {frame_count}, FPS: {fps}, Resolution: {width}x{height}")
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict mask
        mask_colored = predict_frame(model, frame, num_classes, size, device)
        
        # Overlay mask lên frame
        overlay = cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)
        
        # Ghi frame vào video output
        out.write(overlay)
        
        if (i+1) % 10 == 0:
            print(f"⏳ Đã xử lý {i+1}/{frame_count} frames")
    
    cap.release()
    out.release()
    print(f"✅ Hoàn tất! Video dự đoán lưu tại: {output_video_path}")
