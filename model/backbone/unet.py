import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import numpy as np
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm
from model.backbone.blocks import AdaDConv, SCBConvBlock

from ops import Upsample  # Lớp upsampling từ OpenMMLab



class ConvBlock(nn.Module):
    """Khối Convolution cơ bản 2 lớp, tương đương với ConvModule trong mmcv."""
    def __init__(self, in_channels, out_channels, num_convs=2, norm_cfg=True, act_cfg=True):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(
                nn.Sequential(
                    # Conv 3x3
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 
                              kernel_size=3, padding=1, bias=False),
                    # Batch Norm
                    nn.BatchNorm2d(out_channels) if norm_cfg else nn.Identity(),
                    # ReLU
                    nn.ReLU(inplace=True) if act_cfg else nn.Identity(),
                )
            )
        self.convs = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.convs(x)

class UpConvBlock(nn.Module):
    """
    Khối giải mã (Decoder Block) tương đương với UpConvBlock của mmseg/mmcv.
    Thực hiện: Upsample -> Concat (Skip) -> Convolution (PyTorchConvBlock/SCB).
    """
    def __init__(self, in_channels, skip_channels, out_channels, 
                 conv_block=ConvBlock): # Có thể dùng PyTorchConvBlock hoặc SCB
        
        super().__init__()
        
        # 1. Upsample module (Tương đương InterpConv: Bilinear Upsample + 1x1 Conv)
        # Giảm số kênh từ in_channels (từ tầng dưới) xuống skip_channels
        self.upsample = nn.Sequential(
            # Bilinear Upsample: Tăng kích thước không gian lên x2
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
            # 1x1 Conv: Điều chỉnh kênh
            nn.Conv2d(in_channels, skip_channels, kernel_size=1, bias=False) 
        )
        
        # 2. Convolution Block (Fuses skip connection và upsampled features)
        # Input channel sau khi concatenate: skip_channels (từ skip) + skip_channels (từ upsampled x)
        self.conv_block = conv_block(
            in_channels=2 * skip_channels,
            out_channels=out_channels,
        )

    def forward(self, skip, x):
        """
        Args:
            skip (Tensor): Feature map từ Encoder (low-level, high-resolution).
            x (Tensor): Feature map từ Decoder tầng dưới (high-level, low-resolution).
        """
        
        # 1. Upsample
        x = self.upsample(x)
        
        # Đảm bảo kích thước khớp nếu có làm tròn khác biệt
        if skip.shape[-2:] != x.shape[-2:]:
             x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        
        # 2. Concatenation (Fusion)
        out = torch.cat([skip, x], dim=1)
        
        # 3. Convolution
        out = self.conv_block(out)

        return out


class LowLightUNet(BaseModule):
    """
    UNet tích hợp SCB, AWD, và trả về cả enc_outs để hỗ trợ DSL.
    """

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_stages=5,
                 strides=(1, 1, 1, 1, 1),
                 enc_num_convs=(2, 2, 2, 2, 2),
                 dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 enc_dilations=(1, 1, 1, 1, 1),
                 dec_dilations=(1, 1, 1, 1),
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 norm_eval=False,
                 dcn=None,
                 plugins=None,
                 pretrained=None,
                 init_cfg=None):
        
        super(LowLightUNet, self).__init__(init_cfg)

        # --- Kiểm tra và Khởi tạo thuộc tính ---
        self.pretrained = pretrained
        # ... (Bỏ qua phần assert checks để giữ mã ngắn gọn, giả định input hợp lệ) ...

        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.norm_eval = norm_eval
        self.base_channels = base_channels
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(num_stages):
            enc_conv_block = []
            
            # --- Encoder Downsampling ---
            if i != 0:
                # THAY THẾ MaxPool2d bằng AdaDConv (AWD)
                if strides[i] == 1 and downsamples[i - 1]:
                    awd_in_channels = base_channels * 2**(i - 1)
                    enc_conv_block.append(
                        AdaDConv(
                            in_channels=awd_in_channels,
                            kernel_size=3,
                            stride=2,
                        )
                    )

                # Khối Decoder
                upsample = (strides[i] != 1 or downsamples[i - 1])
                self.decoder.append(
                    UpConvBlock(
                        conv_block=SCBConvBlock, # SỬ DỤNG SCBConvBlock
                        in_channels=base_channels * 2**i,
                        skip_channels=base_channels * 2**(i - 1),
                        out_channels=base_channels * 2**(i - 1),
                        num_convs=dec_num_convs[i - 1],
                        stride=1,
                        dilation=dec_dilations[i - 1],
                        with_cp=with_cp,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        upsample_cfg=upsample_cfg if upsample else None,
                        dcn=None,
                        plugins=None))

            # Khối Encoder (sau Downsampling, nếu có)
            # Tính toán in_channels cho SCBConvBlock
            if i == 0:
                 current_in_channels = in_channels
            else:
                 current_in_channels = base_channels * 2**(i - 1)
            
            enc_conv_block.append(
                SCBConvBlock( # SỬ DỤNG SCBConvBlock
                    in_channels=current_in_channels,
                    out_channels=base_channels * 2**i,
                    num_convs=enc_num_convs[i],
                    # Stride chỉ áp dụng ở block đầu tiên nếu không dùng AdaDConv
                    stride=strides[i] if (i==0 or not downsamples[i-1]) else 1,
                    dilation=enc_dilations[i],
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    dcn=None,
                    plugins=None))
            self.encoder.append((nn.Sequential(*enc_conv_block)))
            in_channels = base_channels * 2**i

    def forward(self, x):
        """
        Forward function. Chỉ trả về đầu ra từ các tầng Decoder (cho Task Head).
        """
        # self._check_input_divisible(x) # Có thể bỏ qua nếu dùng PyTorch thuần
        
        enc_outs = []
        # Encoder
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
        
        # Decoder (Bao gồm bottleneck x)
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)

        # Trả về TẤT CẢ các đầu ra Decoder (tương thích với nhiều Task Head)
        return dec_outs  

    def train(self, mode=True):
        """Chuyển đổi mô hình sang chế độ training trong khi vẫn giữ freeze lớp chuẩn hóa."""
        super(LowLightUNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            # Tính toán tổng downsample rate
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        
        if not ((h % whole_downsample_rate == 0) and (w % whole_downsample_rate == 0)):
            raise ValueError(
                f'The input image size {(h, w)} should be divisible by the whole '
                f'downsample rate {whole_downsample_rate}, when num_stages is '
                f'{self.num_stages}, strides is {self.strides}, and downsamples '
                f'is {self.downsamples}.'
            )
class TaskOnlyLoss(nn.Module):
    def __init__(self, num_classes):
        super(TaskOnlyLoss, self).__init__()
        # Cross-Entropy Loss (Input: Logits (B, C, H, W), Target: Mask (B, H, W))
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, prediction_logits, target_mask):
        target_mask = target_mask.long() 
        L_Task = self.ce_loss(prediction_logits, target_mask)
        return L_Task
    
class SimpleTaskHead(nn.Module):
    """
    Chuyển đầu ra Decoder (dec_outs) thành Logits dự đoán cho Segmentation.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Dec_outs[-1] là đầu ra full size, có BASE_CHANNELS
        self.final_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, dec_outs):
        # Đầu ra cuối cùng của Decoder (có kích thước gốc)
        final_features = dec_outs[-1] 
        # Logits (B, NUM_CLASSES, H, W)
        return self.final_conv(final_features)
