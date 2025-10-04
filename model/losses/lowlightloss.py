import torch.nn as nn
import torch

class LowLightLoss(nn.Module):
    def __init__(self, lambda_dsl=1.0):
        super(LowLightLoss, self).__init__()
        self.lambda_dsl = lambda_dsl
        # Task Loss: Sử dụng L1 Loss cho nhiệm vụ Enhancement
        self.task_loss_func = nn.L1Loss() 
        # DSL Loss: Sử dụng L1 Loss giữa các đặc trưng
        self.dsl_loss_func = nn.L1Loss() 

    def forward(self, model, noisy_image, gt_clean_image, reduction='mean'):
        # 1. NOISY Forward Pass (Dùng cho Task Loss và DSL Loss)
        # dec_outs: list[Tensor] từ Decoder, enc_outs: list[Tensor] từ Encoder
        noisy_dec_outs, noisy_enc_outs = model(noisy_image)
        
        # 2. GT Forward Pass (CHỈ DÙNG CHO DSL Loss)
        with torch.no_grad(): # Tắt gradient cho GT Pass để tránh ảnh hưởng ngược
            _, gt_enc_outs = model(gt_clean_image)
            
        # 3. Tính Task Loss
        # Giả định đầu ra cuối cùng là phần tử đầu tiên của dec_outs (output full size)
        final_output = noisy_dec_outs[-1] 
        L_Task = self.task_loss_func(final_output, gt_clean_image)

        # 4. Tính DSL Loss
        L_DSL = 0.0
        # Tính DSL trên tất cả các tầng Encoder (trừ tầng cuối cùng/bottleneck)
        # LIS thường dùng 4 tầng đầu tiên (0, 1, 2, 3)
        for i in range(len(noisy_enc_outs) - 1): 
            # So sánh đặc trưng nhiễu và sạch ở tầng i
            L_DSL += self.dsl_loss_func(noisy_enc_outs[i], gt_enc_outs[i])
        
        # 5. Total Loss
        L_Total = L_Task + self.lambda_dsl * L_DSL

        # Trả về các giá trị loss để theo dõi
        return L_Total, L_Task, L_DSL