import torch
import torch.nn.functional as F
import numpy as np

# =================================================================
# 1. HÀM TÍNH TOÁN CƠ BẢN CHO MỘT LỚP (PER-CLASS METRICS)
# =================================================================

def dice_coeff(pred, target, smooth=1e-6):
    """
    Tính hệ số Dice (Dice Coefficient) cho một cặp dự đoán và nhãn.
    Args:
        pred (torch.Tensor): Tensor nhị phân (0 hoặc 1) của dự đoán, kích thước (H, W).
        target (torch.Tensor): Tensor nhị phân (0 hoặc 1) của nhãn, kích thước (H, W).
        smooth (float): Hằng số làm mượt để tránh chia cho 0.
    Returns:
        float: Hệ số Dice.
    """
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def jaccard_iou(pred, target, smooth=1e-6):
    """
    Tính chỉ số Jaccard/IoU (Intersection over Union) cho một cặp dự đoán và nhãn.
    Args:
        pred (torch.Tensor): Tensor nhị phân (0 hoặc 1) của dự đoán, kích thước (H, W).
        target (torch.Tensor): Tensor nhị phân (0 hoặc 1) của nhãn, kích thước (H, W).
    Returns:
        float: Chỉ số IoU.
    """
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def pixel_accuracy(pred, target):
    """
    Tính độ chính xác pixel (Pixel Accuracy) cho một cặp dự đoán và nhãn.
    Args:
        pred (torch.Tensor): Tensor nhị phân (0 hoặc 1) của dự đoán.
        target (torch.Tensor): Tensor nhị phân (0 hoặc 1) của nhãn.
    Returns:
        float: Độ chính xác.
    """
    correct_pixels = (pred == target).sum()
    total_pixels = target.numel()
    acc = correct_pixels.float() / total_pixels
    return acc.item()


# =================================================================
# 2. HÀM TÍNH TOÁN CHO MULTI-CLASS (mIoU, mDice, mAcc)
# =================================================================

def calculate_multi_class_metrics(prediction_logits, target_mask, num_classes):
    """
    Tính mIoU, mDice, và mAcc cho bài toán Segmentation đa lớp.

    Args:
        prediction_logits (torch.Tensor): Logits đầu ra từ mô hình, kích thước (B, C, H, W).
        target_mask (torch.Tensor): Nhãn mask, kích thước (B, H, W) và dtype=torch.long.
        num_classes (int): Tổng số lớp.

    Returns:
        dict: Chứa mIoU, mDice, mAcc.
    """
    # Lấy dự đoán lớp (index) từ logits
    # pred_mask có kích thước (B, H, W)
    pred_mask = torch.argmax(prediction_logits, dim=1) 
    
    # Đảm bảo target_mask cùng kích thước và dtype là long (đã được PairedTransform xử lý)
    if target_mask.dtype != torch.long:
         target_mask = target_mask.long()

    # Khởi tạo danh sách lưu trữ
    iou_scores = []
    dice_scores = []
    acc_scores = []
    
    # Bỏ qua lớp nền (background) hoặc lớp bỏ qua (ignore_index=255) nếu cần,
    # nhưng ở đây ta tính toán cho tất cả các lớp (0 đến num_classes - 1)
    
    for class_id in range(num_classes):
        # 1. Tạo mask nhị phân cho lớp hiện tại
        pred_class = (pred_mask == class_id)
        target_class = (target_mask == class_id)
        
        # Bỏ qua nếu không có pixel nào thuộc lớp này trong cả dự đoán và nhãn
        if target_class.sum() == 0 and pred_class.sum() == 0:
            continue
            
        # 2. Tính toán Metric trên mask nhị phân
        iou = jaccard_iou(pred_class, target_class)
        dice = dice_coeff(pred_class, target_class)
        # Note: Accuracy đa phần được tính chung, nhưng ta tính Acc cho từng lớp
        # để tính mAcc (tính toán này là chính xác cho từng lớp)
        acc = pixel_accuracy(pred_class, target_class)
        
        iou_scores.append(iou)
        dice_scores.append(dice)
        acc_scores.append(acc)

    # 3. Tính Mean Metrics
    # Nếu không có lớp nào được tìm thấy, trả về 0
    if not iou_scores:
        return {'mIoU': 0.0, 'mDice': 0.0, 'mAcc': 0.0}
    
    mIoU = np.mean(iou_scores)
    mDice = np.mean(dice_scores)
    mAcc = np.mean(acc_scores)

    return {'mIoU': mIoU, 'mDice': mDice, 'mAcc': mAcc}

# =================================================================
# 3. HÀM TÍNH TOÁN TỔNG THỂ (OVERALL ACCURACY)
# =================================================================

def overall_pixel_accuracy(prediction_logits, target_mask):
    """
    Tính độ chính xác pixel tổng thể trên toàn bộ mask (không phân biệt lớp).
    """
    pred_mask = torch.argmax(prediction_logits, dim=1)
    
    # So sánh từng pixel
    correct_pixels = (pred_mask == target_mask).sum()
    total_pixels = target_mask.numel()
    
    return (correct_pixels.float() / total_pixels).item()
