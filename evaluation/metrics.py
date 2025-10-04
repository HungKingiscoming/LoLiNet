import torch
import numpy as np

IGNORE_INDEX = 255

def batch_multi_class_metrics(pred_logits, target_mask, num_classes, smooth=1e-6):
    """
    Tính mIoU, mDice, mAcc cho batch segmentation đa lớp.
    
    Args:
        pred_logits: Tensor (B, C, H, W) - logits model output
        target_mask: Tensor (B, H, W) - ground truth với giá trị 0..num_classes-1 hoặc IGNORE_INDEX
        num_classes: int - số lớp (không tính IGNORE_INDEX)
        smooth: float - hệ số làm mượt tránh chia 0
    
    Returns:
        dict: {'mIoU': float, 'mDice': float, 'mAcc': float}
    """
    # 1. Dự đoán nhãn (B, H, W)
    pred_mask = pred_logits.argmax(dim=1)

    # 2. Tạo mask các pixel hợp lệ
    valid_mask = target_mask != IGNORE_INDEX  # (B, H, W)

    # 3. One-hot encoding prediction và target
    # Chuyển sang (B, C, H, W) nhị phân
    pred_onehot = torch.nn.functional.one_hot(pred_mask, num_classes=num_classes)  # (B, H, W, C)
    target_onehot = torch.nn.functional.one_hot(target_mask.clamp(0, num_classes-1), num_classes=num_classes)
    
    pred_onehot = pred_onehot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

    # 4. Chỉ giữ pixel hợp lệ
    valid_mask = valid_mask.unsqueeze(1)  # (B, 1, H, W)
    pred_onehot = pred_onehot * valid_mask
    target_onehot = target_onehot * valid_mask

    # 5. Tính intersection và union cho từng lớp
    intersection = (pred_onehot * target_onehot).sum(dim=(0, 2, 3))  # (C,)
    union = pred_onehot.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))
    iou = (intersection + smooth) / (union - intersection + smooth)
    dice = (2 * intersection + smooth) / (union + smooth)

    # 6. Pixel Accuracy cho từng lớp
    acc = (pred_onehot * target_onehot + (1 - pred_onehot) * (1 - target_onehot)).sum(dim=(0,2,3)) \
          / valid_mask.sum(dim=(0,2,3).keepdim(False) + 1e-6)

    # 7. Lọc lớp không có pixel hợp lệ
    valid_classes = target_onehot.sum(dim=(0,2,3)) > 0
    mIoU = iou[valid_classes].mean().item()
    mDice = dice[valid_classes].mean().item()
    mAcc = acc[valid_classes].mean().item()

    return {'mIoU': mIoU, 'mDice': mDice, 'mAcc': mAcc}

