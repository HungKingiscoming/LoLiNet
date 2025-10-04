import torch
import numpy as np

IGNORE_INDEX = 255

def batch_multi_class_metrics(pred_logits, target_mask, num_classes, smooth=1e-6):
    """
    Tính mIoU, mDice, và mAcc (Mean Class Recall) cho batch segmentation đa lớp
    bằng phép toán tensor.
    """
    # 1. Dự đoán nhãn (B, H, W)
    pred_mask = pred_logits.argmax(dim=1)

    # 2. Tạo mask các pixel hợp lệ
    valid_mask = target_mask != IGNORE_INDEX  # (B, H, W)

    # 3. One-hot encoding prediction và target
    # target_mask.clamp(0, num_classes-1) đảm bảo các giá trị ignore (255)
    # được giới hạn trong khoảng [0, C-1] cho one-hot, nhưng sẽ bị valid_mask loại bỏ sau.
    pred_onehot = torch.nn.functional.one_hot(pred_mask, num_classes=num_classes)  # (B, H, W, C)
    target_onehot = torch.nn.functional.one_hot(
        target_mask.clamp(0, num_classes - 1), 
        num_classes=num_classes
    )  # (B, H, W, C)
    
    # Chuyển đổi về (B, C, H, W)
    pred_onehot = pred_onehot.permute(0, 3, 1, 2).float()
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()

    # 4. Chỉ giữ pixel hợp lệ
    valid_mask_expanded = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)
    
    # Áp dụng valid_mask. Các pixel IGNORE_INDEX sẽ bị zero-out
    pred_onehot = pred_onehot * valid_mask_expanded
    target_onehot = target_onehot * valid_mask_expanded

    # 5. Tính intersection, target_sum, và pred_sum cho từng lớp (C,)
    intersection = (pred_onehot * target_onehot).sum(dim=(0, 2, 3))
    target_sum = target_onehot.sum(dim=(0, 2, 3))
    pred_sum = pred_onehot.sum(dim=(0, 2, 3))
    
    # 6. Tính IoU và Dice cho từng lớp
    union = target_sum + pred_sum
    iou = (intersection + smooth) / (union - intersection + smooth)
    dice = (2 * intersection + smooth) / (union + smooth)

    # 7. Tính Mean Class Accuracy (Recall)
    # Tránh chia cho 0. Các lớp có target_sum=0 sẽ bị lọc ở bước 8.
    acc = intersection / (target_sum + 1e-6)

    # 8. Lọc lớp không có pixel hợp lệ (target_sum > 0)
    valid_classes = target_sum > 0
    
    # Nếu không có lớp hợp lệ nào trong batch (trường hợp hiếm)
    if not valid_classes.any():
        return {'mIoU': 0.0, 'mDice': 0.0, 'mAcc': 0.0}

    mIoU = iou[valid_classes].mean().item()
    mDice = dice[valid_classes].mean().item()
    mAcc = acc[valid_classes].mean().item() # mAcc là trung bình của Recall

    return {'mIoU': mIoU, 'mDice': mDice, 'mAcc': mAcc}

def overall_pixel_accuracy(prediction_logits, target_mask):
    """
    Tính độ chính xác pixel tổng thể, LOẠI BỎ các pixel có nhãn IGNORE_INDEX (255).
    
    Args:
        prediction_logits (torch.Tensor): Logits đầu ra từ mô hình (B, C, H, W).
        target_mask (torch.Tensor): Nhãn mask (B, H, W).

    Returns:
        float: Độ chính xác tổng thể (0.0 đến 1.0).
    """
    # 1. Lấy dự đoán lớp (index) (B, H, W)
    pred_mask = torch.argmax(prediction_logits, dim=1)
    
    # 2. Tạo mặt nạ boolean nơi target_mask KHÔNG PHẢI là IGNORE_INDEX
    valid_pixels_mask = (target_mask != IGNORE_INDEX)
    
    # 3. Tổng số pixel hợp lệ
    total_valid_pixels = valid_pixels_mask.sum().item()
    
    if total_valid_pixels == 0:
        return 1.0 # Trả về 1.0 nếu không có pixel hợp lệ (tránh chia cho 0)

    # 4. So sánh dự đoán và nhãn CHỈ tại các vị trí hợp lệ
    correct_predictions = (pred_mask == target_mask)
    
    # Lọc kết quả đúng chỉ trên các pixel hợp lệ
    total_correct_valid = (correct_predictions & valid_pixels_mask).sum().item()
    
    return total_correct_valid / total_valid_pixels
