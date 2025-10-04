import torch
import torch.nn as nn
import torch.nn.functional as F

# Đảm bảo các lớp AdaDConv và LLPFConv đã được import/định nghĩa ở đây
# from your_file import AdaDConv, LLPFConv 

# Dùng lại LLPFConv bạn đã cung cấp:
class LLPFConv(nn.Module):
    """ Learnable Low Pass Filter, Smooth-oriented convolution """
    def __init__(self, channels=3, stride=1, padding=1):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        # Khởi tạo kernel Gaussian/Blur
        kernel = [
            [1/16., 1/8., 1/16.],
            [1/8., 1/4., 1/8.],
            [1/16., 1/8., 1/16.],
        ]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=True)
    
    def forward(self, x):
        if x.dim == 3:
            x.unsqueeze(0)
        # Chuẩn hóa (softmax) để kernel là low-pass filter
        normalized_weight = self.weight.reshape(self.channels, 1, -1).softmax(-1).reshape(self.channels, 1, 3, 3)
        x = F.conv2d(x, normalized_weight, 
                     padding=self.padding, groups=self.channels, stride=self.stride)
        return x


class AdaDConv(nn.Module):
    """
    Adaptive-weighted downsampling
    Sử dụng code bạn cung cấp cho cả __init__ và forward.
    """
    def __init__(self, in_channels, kernel_size=3, stride=2, groups=1, use_channel=True, use_nin=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = (kernel_size-1) // 2
        self.stride = stride
        self.in_channels = in_channels
        self.groups = groups
        self.use_channel = use_channel

        if use_nin:
            mid_channel = min((kernel_size ** 2 // 2), 4)
            self.weight_net = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=groups * mid_channel , stride=stride,
                          kernel_size=kernel_size, bias=False, padding=self.pad, groups=groups),
                nn.BatchNorm2d(self.groups * mid_channel), 
                nn.ReLU(True),
                nn.Conv2d(in_channels=groups * mid_channel, out_channels=groups * kernel_size ** 2, stride=1,
                          kernel_size=1, bias=False, padding=0, groups=groups),
                nn.BatchNorm2d(self.groups * kernel_size ** 2), 
            )

        else:
            self.weight_net = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=groups * kernel_size ** 2, stride=stride,
                          kernel_size=kernel_size, bias=False, padding=self.pad, groups=groups),
                nn.BatchNorm2d(self.groups * kernel_size ** 2), 
            )

        if use_channel:
            self.channel_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), 
                nn.Conv2d(in_channels=in_channels, out_channels= in_channels // 4, kernel_size=1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(in_channels=in_channels // 4, out_channels = in_channels, kernel_size=1, bias=False),
            )

    def forward(self, x):
        # Đây là logic tính toán Adaptive Weighted Downsampling CHÍNH XÁC
        b, c, h, w = x.shape
        oh = (h - 1) // self.stride + 1
        ow = (w - 1) // self.stride + 1
        weight = self.weight_net(x) 
        weight = weight.reshape(b, self.groups, 1, self.kernel_size ** 2, oh, ow) 
        weight = weight.repeat(1, 1, c // self.groups, 1, 1, 1)

        if self.use_channel:
            tmp = self.channel_net(x).reshape(b, self.groups, c // self.groups, 1, 1, 1)
            weight = weight * tmp
            
        weight = weight.permute(0, 1, 2, 4, 5, 3).softmax(dim=-1)
        weight = weight.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)

        pad_x = F.pad(x, pad=[self.pad] * 4, mode='reflect')
        pad_x = pad_x.unfold(2, self.kernel_size,self.stride).unfold(3, self.kernel_size, self.stride)
        pad_x = pad_x.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)
        
        res = weight * pad_x
        res = res.sum(dim=(-1, -2)).reshape(b, c, oh, ow)
        return res



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 - Đã sửa đổi với nguyên tắc SCB. """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        # 1. Luồng Trích xuất Đặc trưng Tiêu chuẩn
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. Luồng Lọc Thông Thấp (LLPF Residual - Tích hợp SCB)
        # - LLPFConv: Lọc thông thấp học được (depth-wise, C_in -> C_in)
        # - Conv 1x1: Điều chỉnh kênh (C_in -> C_out) để khớp với đầu ra của self.conv_block
        self.llpf_res = nn.Sequential(
            LLPFConv(channels=in_channels, stride=1, padding=1),
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
        
        # Tổng hợp: Luồng chính + (Trọng số * Luồng làm mượt)
        return x_conv + self.gamma * x_smooth


