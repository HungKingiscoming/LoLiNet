import torch
import torch.nn as nn
import torch.nn.functional as F
from model.custom_block import AdaDConv, LLPFConv 


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 - ĐÃ SỬA ĐỔI VỚI SCB RESIDUAL"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        # 1. Luồng Trích xuất Đặc trưng Tiêu chuẩn (Standard Conv)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. Luồng Lọc Thông Thấp (LLPF Residual - SCB Principle)
        # Tạo kết nối tàn dư cho lớp làm mượt để ổn định đặc trưng
        self.llpf_res = nn.Sequential(
            LLPFConv(channels=in_channels, stride=1, padding=1),
            # Conv 1x1 để điều chỉnh kênh từ in_channels -> out_channels
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), 
            nn.BatchNorm2d(out_channels)
        )
        
        # 3. Trọng số học được để kiểm soát sự đóng góp của luồng làm mượt
        self.gamma = nn.Parameter(torch.zeros(1)) 

    def forward(self, x):
        # Luồng chính (Standard Conv)
        x_conv = self.conv_block(x)
        
        # Luồng làm mượt (LLPF Residual)
        x_smooth = self.llpf_res(x)
        
        # Tổng hợp: Luồng chính + (Trọng số học được * Luồng làm mượt)
        return x_conv + self.gamma * x_smooth

# ===============================================================
# C. DOWN/UP/OUT CONV - Thay thế MaxPool bằng AdaDConv
# ===============================================================

class Down(nn.Module):
    """Giảm mẫu bằng AdaDConv sau đó DoubleConv (đã sửa đổi)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = AdaDConv(in_channels=in_channels, stride=2) 
        # DoubleConv đã được sửa đổi để sử dụng SCB
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        # 1. Giảm mẫu thích ứng (Adaptive Downsampling)
        x = self.down(x) 
        # 2. Trích xuất đặc trưng và làm mượt (Smooth Double Conv)
        return self.conv(x)


class Up(nn.Module):
    """Upscaling sau đó DoubleConv (đã sửa đổi)"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
