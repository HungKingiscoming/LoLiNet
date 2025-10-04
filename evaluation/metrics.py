import torch
import torch.nn.functional as F
import numpy as np

# Hằng số chung
IGNORE_INDEX = 255 # Nhãn được ánh xạ để bỏ qua

# =================================================================
# 1. HÀM TÍNH TOÁN CƠ BẢN CHO MỘT LỚP (PER-CLASS METRICS)
# =================================================================

def dice_coeff(pred, target, smooth=1e-6):
    """
    Tính hệ số Dice (Dice Coefficient) cho một cặp dự đoán và nhãn nhị phân.
    """
    # Làm phẳng (flatten) tensor
    pred = pred.float().flatten()
    target = target.float().flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def jaccard_iou(pred, target, smooth=1e-6):
    """
    Tính chỉ số Jaccard/IoU (Intersection over Union) cho một cặp dự đoán và nhãn nhị phân.
    """
    # Làm phẳng (flatten) tensor
    pred = pred.float().flatten()
    target = target.float().flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def pixel_accuracy(pred, target):
    """
    Tính độ chính xác pixel (Pixel Accuracy) cho một cặp dự đoán và nhãn nhị phân.
    """
    correct_pixels = (pred == target).sum()
    total_pixels = target.numel()
    
    if total_pixels == 0:
        return 1.0 # Nếu không có pixel nào thuộc lớp này, coi là chính xác 100%
        
    acc = correct_pixels.float() / total_pixels
    return acc.item()


# =================================================================
# 2. HÀM TÍNH TOÁN CHO MULTI-CLASS (mIoU, mDice, mAcc)
# =================================================================

def calculate_multi_class_metrics(prediction_logits, target_mask, num_classes):
    """
    Tính mIoU, mDice, và mAcc cho bài toán Segmentation đa lớp, 
    chỉ tính toán trên các pixel có nhãn hợp lệ (0 đến num_classes - 1).
    """
    # 1. Dự đoán và lọc nhãn không hợp lệ
    # pred_mask có kích thước (B, H, W)
    pred_mask = torch.argmax(prediction_logits, dim=1) 
    
    # 2. Loại bỏ các pixel IGNORE_INDEX (255)
    valid_pixels_mask = (target_mask != IGNORE_INDEX)
    
    # Khởi tạo danh sách lưu trữ
    iou_scores = []
    dice_scores = []
    acc_scores = []
    
    # Ta chỉ tính toán cho các lớp từ 0 đến num_classes - 1 (tức là 0 đến 19)
    for class_id in range(num_classes):
        # Tạo mask nhị phân cho lớp hiện tại, chỉ sử dụng các pixel hợp lệ
        
        # Tạo mask nhị phân cho toàn bộ batch/ảnh
        pred_class_full = (pred_mask == class_id)
        target_class_full = (target_mask == class_id)
        
        # Chỉ giữ lại các pixel hợp lệ (KHÔNG phải 255)
        # Bằng cách sử dụng phép AND, ta loại bỏ các pixel 255 khỏi cả target và prediction
        pred_class = pred_class_full & valid_pixels_mask
        target_class = target_class_full & valid_pixels_mask
        
        # Nếu không có pixel nào thuộc lớp này trong các pixel hợp lệ
        if target_class.sum() == 0 and pred_class.sum() == 0:
            continue
            
        # Tính toán Metric trên mask nhị phân (chỉ chứa 0/1, không còn 255)
        iou = jaccard_iou(pred_class, target_class)
        dice = dice_coeff(pred_class, target_class)
        acc = pixel_accuracy(pred_class, target_class)
        
        iou_scores.append(iou)
        dice_scores.append(dice)
        acc_scores.append(acc)

    # 3. Tính Mean Metrics
    if not iou_scores:
        # Trường hợp hiếm: không có lớp có ý nghĩa nào được tìm thấy trong batch
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
    Tính độ chính xác pixel tổng thể trên toàn bộ mask, 
    LOẠI BỎ các pixel có nhãn IGNORE_INDEX (255).
    """
    pred_mask = torch.argmax(prediction_logits, dim=1)
    
    # Tạo mặt nạ boolean nơi target_mask KHÔNG PHẢI là IGNORE_INDEX
    valid_pixels_mask = (target_mask != IGNORE_INDEX)
    
    # 1. Tổng số pixel hợp lệ
    total_valid_pixels = valid_pixels_mask.sum().item()
    
    if total_valid_pixels == 0:
        return 1.0 # Tránh chia cho 0 nếu không có pixel nào để tính

    # 2. So sánh dự đoán và nhãn CHỈ tại các vị trí hợp lệ
    correct_predictions = (pred_mask == target_mask)
    
    # Lọc kết quả đúng chỉ trên các pixel hợp lệ
    total_correct_valid = (correct_predictions & valid_pixels_mask).sum().item()
    
    return total_correct_valid / total_valid_pixels
