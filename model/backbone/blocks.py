import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import numpy as np
from mmcv.cnn import (UPSAMPLE_LAYERS, ConvModule, build_activation_layer,
                      build_norm_layer)
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm

# Giả định các module sau đã được import:
# from ops import Upsample
# from ..builder import BACKBONES
# from ..utils import UpConvBlock 

# -------------------------- LLPFConv (Smooth Component) --------------------------
class LLPFConv(nn.Module):
    """Learnable Low Pass Filter (Thành phần của SCB)."""
    def __init__(self, channels, stride=1, padding=1):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        # Khởi tạo kernel tương tự Gaussian kernel
        kernel = torch.FloatTensor([[1/16., 1/8., 1/16.], [1/8., 1/4., 1/8.], [1/16., 1/8., 1/16.]]).unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=True)
 
    def forward(self, x):
        weight_softmax = self.weight.reshape(self.channels, 1, -1).softmax(-1).reshape(self.channels, 1, 3, 3)
        x = F.conv2d(x, weight_softmax, padding=self.padding, groups=self.channels, stride=self.stride)
        return x

# -------------------------- SCB Conv Module --------------------------
class SCBConvModule(nn.Module):
    """SCB(x) = StandardConv(x) + Proj(LLPF(x))"""
    def __init__(self, in_channels, out_channels, **kwargs):
        super(SCBConvModule, self).__init__()
        
        conv_cfg = kwargs.get('conv_cfg')
        norm_cfg = kwargs.get('norm_cfg')
        act_cfg = kwargs.get('act_cfg')
        stride = kwargs.get('stride', 1)
        dilation = kwargs.get('dilation', 1)
        
        self.standard_branch = ConvModule(
            in_channels, out_channels, kernel_size=3, stride=stride, dilation=dilation,
            padding=dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        self.llpf_filter = LLPFConv(channels=in_channels, stride=stride, padding=dilation)
        
        self.llpf_proj = ConvModule(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None) 

    def forward(self, x):
        smooth_out = self.llpf_filter(x)
        smooth_out = self.llpf_proj(smooth_out)
        return self.standard_branch(x) + smooth_out

# -------------------------- SCBConvBlock (Thay thế BasicConvBlock) --------------------------
class SCBConvBlock(nn.Module):
    """SCB Convolutional Block sử dụng SCBConvModule."""
    def __init__(self,
                 in_channels, out_channels, num_convs=2, stride=1, dilation=1,
                 with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'),
                 dcn=None, plugins=None):
        super(SCBConvBlock, self).__init__()
        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(
                SCBConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out

# -------------------------- AdaDConv (AWD - Thay thế MaxPool2d) --------------------------
class AdaDConv(nn.Module):
    """Adaptive-weighted Downsampling (AWD)."""
    def __init__(self, in_channels, kernel_size=3, stride=2, groups=1, use_channel=True, use_nin=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2
        self.stride = stride
        self.in_channels = in_channels
        self.groups = groups
        self.use_channel = use_channel

        if use_nin:
             pass 
        else:
            self.weight_net = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=groups * kernel_size ** 2, stride=stride,
                          kernel_size=kernel_size, bias=False, padding=self.pad, groups=groups),
                nn.BatchNorm2d(self.groups * kernel_size ** 2), 
            )

        if use_channel:
            self.channel_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), 
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(in_channels=in_channels // 4, out_channels=in_channels, kernel_size=1, bias=False),
            )

    def forward(self, x):
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
        pad_x = pad_x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        pad_x = pad_x.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)
        
        res = weight * pad_x
        res = res.sum(dim=(-1, -2)).reshape(b, c, oh, ow)
        return res