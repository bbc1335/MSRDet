# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .conv import (Conv, DWConv, GhostConv, LightConv, RepConv, autopad, CBAM, ECA, CoordAtt, ELA,
                   SequentialPolarizedSelfAttention, ChannelAttention, SpatialAttention)
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "AConv",
    "RF",
    "DCE",
    'TEM',
    "InjectionMultiSum_Auto_pool",
    "Fusion_2in",
    "ALF",
    "HTEM",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "GatedSPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fk",
    "DFCA",
    "LCSPELAN4",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class GatedSPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        self.epsilon = 1e-4
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        weight = F.silu(self.w)
        weight = weight / (torch.sum(weight, dim=0) + self.epsilon)
        y.extend(weight[_] * self.m(y[-1]) for _ in range(3))
        return self.cv2((torch.cat(y, 1)))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class AConv(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class Bottleneck1(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, c_, 1, 1)
        self.cv1 = Conv(c_, c_, k[0], 1, g=g)
        self.cv2 = Conv(c_, c2, k[1], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(self.cv0(x))) if self.add else self.cv2(self.cv1(x))


class Bottleneck2(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, c_, 1, 1)
        self.cv1 = Conv(c_, c_, k[0], 1, g=g)
        self.cv2 = Conv(c_, c2, k[1], 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(self.cv0(x)) + self.cv3(self.cv0(x))) if self.add else self.cv2(self.cv1(x))


class C2fk(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, kl=3, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck1(self.c, self.c, shortcut, self.c, k=((kl, kl), (1, 1)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class DFCA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c2):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.gate_fn = nn.Sigmoid()

        self.short_conv = nn.Sequential(
            nn.Conv2d(c2, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.Conv2d(c2, c2, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.Conv2d(c2, c2, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=c2, bias=False),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        """Forward pass through C2f layer."""
        res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
        out = x * F.interpolate(self.gate_fn(res), size=(x.shape[-2], x.shape[-1]), mode='nearest')

        return out


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# class RF(nn.Module):
#     # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
#     # GitHub: https://github.com/ruinmessi/RFBNet
#     def __init__(self, in_channel, out_channel):
#         super(RF, self).__init__()
#         self.act = nn.SiLU(True)
#
#         self.branch0 = nn.Sequential(
#             Conv(in_channel, out_channel, 1, act=False),
#         )
#         self.branch1 = nn.Sequential(
#             Conv(in_channel, out_channel, 1, act=False),
#             Conv(out_channel, out_channel, k=(1, 3), p=(0, 1), act=False),
#             Conv(out_channel, out_channel, k=(3, 1), p=(1, 0), act=False),
#             Conv(out_channel, out_channel, 3, p=3, d=3, act=False)
#         )
#         self.branch2 = nn.Sequential(
#             Conv(in_channel, out_channel, 1, act=False),
#             Conv(out_channel, out_channel, k=(1, 5), p=(0, 2), act=False),
#             Conv(out_channel, out_channel, k=(5, 1), p=(2, 0), act=False),
#             Conv(out_channel, out_channel, 3, p=5, d=5, act=False)
#         )
#         self.branch3 = nn.Sequential(
#             Conv(in_channel, out_channel, 1, act=False),
#             Conv(out_channel, out_channel, k=(1, 7), p=(0, 3), act=False),
#             Conv(out_channel, out_channel, k=(7, 1), p=(3, 0), act=False),
#             Conv(out_channel, out_channel, 3, p=7, d=7, act=False)
#         )
#
#         self.conv_cat = Conv(4*out_channel, out_channel, 3, p=1, act=False)
#         self.conv_res = Conv(in_channel, out_channel, 1, act=False)
#
#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#         x3 = self.branch3(x)
#
#         x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))
#
#         x = self.act(x_cat + self.conv_res(x))
#         return x


class RF(nn.Module):
    # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
    # GitHub: https://github.com/ruinmessi/RFBNet
    def __init__(self, in_channel, out_channel):
        super(RF, self).__init__()
        self.act = nn.SiLU(True)

        self.branch0 = nn.Sequential(
            Conv(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            Conv(in_channel, out_channel, 1),
            Conv(out_channel, out_channel, k=(1, 3), p=(0, 1)),
            Conv(out_channel, out_channel, k=(3, 1), p=(1, 0)),
            Conv(out_channel, out_channel, 3, p=3, d=3)
        )
        self.branch2 = nn.Sequential(
            Conv(in_channel, out_channel, 1),
            Conv(out_channel, out_channel, k=(1, 5), p=(0, 2)),
            Conv(out_channel, out_channel, k=(5, 1), p=(2, 0)),
            Conv(out_channel, out_channel, 3, p=5, d=5)
        )
        self.branch3 = nn.Sequential(
            Conv(in_channel, out_channel, 1),
            Conv(out_channel, out_channel, k=(1, 7), p=(0, 3)),
            Conv(out_channel, out_channel, k=(7, 1), p=(3, 0)),
            Conv(out_channel, out_channel, 3, p=7, d=7)
        )

        self.conv_cat = Conv(4 * out_channel, out_channel, 3, p=1)
        self.conv_res = Conv(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))

        x = self.act(x_cat + self.conv_res(x))
        return x


# class DCE(nn.Module):
#     # Revised from: Exploring Dense Context for Salient Object Detection, 2022, TCSVT
#     def __init__(self, in_channel, out_channel):
#         super(DCE, self).__init__()
#         self.branch0 = nn.Sequential(
#             Conv(in_channel, out_channel // 4, 1),
#         )
#         self.branch1_0 = Conv(in_channel, out_channel // 4, 1)
#         self.branch1_1 = nn.Sequential(
#             Conv(out_channel // 4, out_channel // 4, k=(1, 3), p=(0, 1)),
#             Conv(out_channel // 4, out_channel // 4, k=(3, 1), p=(1, 0))
#         )
#         self.branch1_2 = nn.Sequential(
#             Conv(out_channel // 4, out_channel // 4, k=(3, 1), p=(1, 0)),
#             Conv(out_channel // 4, out_channel // 4, k=(1, 3), p=(0, 1))
#         )
#         self.branch1_3 = nn.Sequential(
#             Conv(out_channel // 2, out_channel // 4, 1),
#             Conv(out_channel // 4, out_channel // 4, 3, p=3, d=3)
#         )
#
#         self.branch2_0 = Conv(in_channel, out_channel // 4, 1)
#         self.branch2_1 = nn.Sequential(
#             Conv(out_channel // 2, out_channel // 2, k=(1, 5), p=(0, 2)),
#             Conv(out_channel // 2, out_channel // 2, k=(5, 1), p=(2, 0))
#         )
#         self.branch2_2 = nn.Sequential(
#             Conv(out_channel // 2, out_channel // 2, k=(5, 1), p=(2, 0)),
#             Conv(out_channel // 2, out_channel // 2, k=(1, 5), p=(0, 2))
#         )
#         self.branch2_3 = nn.Sequential(
#             Conv(out_channel, out_channel // 4, 1),
#             Conv(out_channel // 4, out_channel // 4, 3, p=5, d=5)
#         )
#
#         self.branch3_0 = Conv(in_channel, out_channel // 4, 1)
#         self.branch3_1 = nn.Sequential(
#             Conv(out_channel // 2, out_channel // 2, k=(1, 7), p=(0, 3)),
#             Conv(out_channel // 2, out_channel // 2, k=(7, 1), p=(3, 0))
#         )
#         self.branch3_2 = nn.Sequential(
#             Conv(out_channel // 2, out_channel // 2, k=(7, 1), p=(3, 0)),
#             Conv(out_channel // 2, out_channel // 2, k=(1, 7), p=(0, 3))
#         )
#         self.branch3_3 = nn.Sequential(
#             Conv(out_channel, out_channel // 4, 1),
#             Conv(out_channel // 4, out_channel // 4, 3, p=7, d=7)
#         )
#
#         self.branch4_0 = Conv(in_channel, out_channel // 4, 1)
#         self.branch4_1 = nn.Sequential(
#             Conv(out_channel // 2, out_channel // 2, k=(1, 9), p=(0, 4)),
#             Conv(out_channel // 2, out_channel // 2, k=(9, 1), p=(4, 0))
#         )
#         self.branch4_2 = nn.Sequential(
#             Conv(out_channel // 2, out_channel // 2, k=(9, 1), p=(4, 0)),
#             Conv(out_channel // 2, out_channel // 2, k=(1, 9), p=(0, 4))
#         )
#         self.branch4_3 = nn.Sequential(
#             Conv(out_channel, out_channel // 4, 1),
#             Conv(out_channel // 4, out_channel // 4, 3, p=9, d=9)
#         )
#         self.conv_cat = Conv(out_channel//4*5, out_channel, 3, p=1)
#
#     def forward(self, x):
#         x0 = self.branch0(x)
#
#         x1_0 = self.branch1_0(x)
#         x1_1 = self.branch1_1(x1_0)
#         x1_2 = self.branch1_2(x1_0)
#         x1_3 = torch.cat((x1_1, x1_2), dim=1)
#         x1 = self.branch1_3(x1_3)
#
#         x2_0 = self.branch2_0(x)
#         x2_0 = torch.cat((x2_0, x1), dim=1)
#         x2_1 = self.branch2_1(x2_0)
#         x2_2 = self.branch2_2(x2_0)
#         x2_3 = torch.cat((x2_1, x2_2), dim=1)
#         x2 = self.branch2_3(x2_3)
#
#         x3_0 = self.branch3_0(x)
#         x3_0 = torch.cat((x3_0, x2), dim=1)
#         x3_1 = self.branch3_1(x3_0)
#         x3_2 = self.branch3_2(x3_0)
#         x3_3 = torch.cat((x3_1, x3_2), dim=1)
#         x3 = self.branch3_3(x3_3)
#
#         x4_0 = self.branch4_0(x)
#         x4_0 = torch.cat((x4_0, x3), dim=1)
#         x4_1 = self.branch4_1(x4_0)
#         x4_2 = self.branch4_2(x4_0)
#         x4_3 = torch.cat((x4_1, x4_2), dim=1)
#         x4 = self.branch4_3(x4_3)
#
#         x_cat = torch.cat((x0, x1, x2, x3, x4), dim=1)
#         x_out = self.conv_cat(x_cat)
#         return x_out


class TEM(nn.Module):
    # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
    # GitHub: https://github.com/ruinmessi/RFBNet
    def __init__(self, in_channel, out_channel):
        super(TEM, self).__init__()
        self.act = nn.SiLU(True)

        self.branch0 = nn.Sequential(
            Conv(in_channel, out_channel // 4, 1),
        )
        self.branch1 = nn.Sequential(
            Conv(in_channel, out_channel // 4, 1, act=False),
            Conv(out_channel // 4, out_channel // 4, k=(1, 3), p=(0, 1)),
            Conv(out_channel // 4, out_channel // 4, k=(3, 1), p=(1, 0)),
            Conv(out_channel // 4, out_channel // 4, 3, p=3, d=3)
        )
        self.branch2 = nn.Sequential(
            Conv(in_channel, out_channel // 4, 1),
            Conv(out_channel // 4, out_channel // 4, k=(1, 5), p=(0, 2)),
            Conv(out_channel // 4, out_channel // 4, k=(5, 1), p=(2, 0)),
            Conv(out_channel // 4, out_channel // 4, 3, p=5, d=5)
        )
        self.branch3 = nn.Sequential(
            Conv(in_channel, out_channel // 4, 1),
            Conv(out_channel // 4, out_channel // 4, k=(1, 7), p=(0, 3)),
            Conv(out_channel // 4, out_channel // 4, k=(7, 1), p=(3, 0)),
            Conv(out_channel // 4, out_channel // 4, 3, p=7, d=7)
        )

        # self.conv_cat = Conv(4*out_channel, out_channel, 3, p=1)
        # self.conv_res = Conv(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat((x0, x1, x2, x3), dim=1)

        return x_cat


class DCEBranch(nn.Module):
    def __init__(self, c1, c2, i, cat=False, shortcut=False, groupconv=False):
        super(DCEBranch, self).__init__()
        self.cat = cat
        self.sc = shortcut
        self.gc = groupconv

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(c2, c2, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

        self.branch1_0 = Conv(c1, c2, 1)
        self.branch1_1 = nn.Sequential(
            Conv(c2 * (2 if self.cat else 1), c2 * (2 if self.cat else 1), k=(1, i), p=(0, i // 2)),
            Conv(c2 * (2 if self.cat else 1), c2 * (2 if self.cat else 1), k=(i, 1), p=(i // 2, 0))
        )
        if self.gc:
            self.branch1_2 = nn.Sequential(
                nn.MaxPool2d(i, 1, (i // 2, i // 2))
            )
        else:
            self.branch1_2 = nn.Sequential(
                Conv(c2 * (2 if self.cat else 1), c2 * (2 if self.cat else 1), k=(i, 1), p=(i // 2, 0)),
                Conv(c2 * (2 if self.cat else 1), c2 * (2 if self.cat else 1), k=(1, i), p=(0, i // 2))
            )
        if self.sc:
            self.branch1_3 = Conv(c2 * (6 if self.cat else 4), c2, 1)
            self.branch1_4 = Conv(c2, c2, 3, p=i, d=i)
        else:
            self.branch1_3 = Conv(c2 * (4 if self.cat else 2), c2, 1)
            self.branch1_4 = Conv(c2, c2, 3, p=i, d=i)

    def forward(self, x):
        if self.cat is False:
            x1_0 = self.branch1_0(x)
            x1_1 = self.branch1_1(x1_0)
            x1_2 = self.branch1_2(x1_0)
        else:
            x1_0 = self.branch1_0(x[1])
            x1_0 = torch.cat((x[0], x1_0), dim=1)
            x1_1 = self.branch1_1(x1_0)
            x1_2 = self.branch1_2(x1_0)
        x1_3 = torch.cat(((x1_1, x1_2) if not self.sc else (x1_0, x1_1, x1_2)), dim=1)
        x1_4 = self.branch1_3(x1_3)
        x1 = self.branch1_4(x1_4)
        return x1


class DCE(nn.Module):
    # Revised from: Exploring Dense Context for Salient Object Detection, 2022, TCSVT
    def __init__(self, in_channel, out_channel, cat=True, sc=False, gc=False):
        super(DCE, self).__init__()
        self.branch0 = Conv(in_channel, out_channel // 4, 1)
        self.branch1 = DCEBranch(in_channel, out_channel // 4, 3, groupconv=gc)
        self.branch2 = DCEBranch(in_channel, out_channel // 4, 5, cat=cat, shortcut=sc, groupconv=gc)
        self.branch3 = DCEBranch(in_channel, out_channel // 4, 7, cat=cat, shortcut=sc, groupconv=gc)
        self.branch4 = DCEBranch(in_channel, out_channel // 4, 9, cat=cat, shortcut=sc, groupconv=gc)
        self.conv_cat = Conv(out_channel // 4 * 5, out_channel, 3, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2([x1, x])
        x3 = self.branch3([x2, x])
        x4 = self.branch4([x3, x])

        x_cat = torch.cat((x0, x1, x2, x3, x4), dim=1)
        x_out = self.conv_cat(x_cat)
        return x_out


class HTEMBranch(nn.Module):
    def __init__(self, c1, c2, i):
        super(HTEMBranch, self).__init__()
        self.branch1_0 = Conv(c1, c2, 1)
        self.branch1_1 = nn.Sequential(
            Conv(c2, c2, k=(1, i), p=(0, i // 2)),
            Conv(c2, c2, k=(i, 1), p=(i // 2, 0))
        )
        self.branch1_2 = nn.Sequential(
            Conv(c2, c2, k=(i, 1), p=(i // 2, 0)),
            Conv(c2, c2, k=(1, i), p=(0, i // 2))
        )
        self.branch1_3 = Conv(c2 * 2, c1, 1)
        self.branch1_4 = Conv(c1, c1, 3, p=i, d=i)

    def forward(self, x):
        x1_0 = self.branch1_0(x[0] + x[1])
        x1_1 = self.branch1_1(x1_0)
        x1_2 = self.branch1_2(x1_0)
        x1_3 = self.branch1_3(torch.cat((x1_1, x1_2), dim=1))
        x1 = self.branch1_4(x1_3)
        return x1


# class MSDCEBranch(nn.Module):
#     def __init__(self, c1, c2, i):
#         super(MSDCEBranch, self).__init__()
#         self.gate_fn = nn.Sigmoid()
#
#         self.branch1_0 = Conv(c1, c2, 1)
#         self.branch1_1 = nn.Sequential(
#             Conv(c2, c2, k=(1, i), p=(0, i // 2)),
#             Conv(c2, c2, k=(i, 1), p=(i // 2, 0))
#         )
#         self.branch1_2 = nn.Sequential(
#             nn.AvgPool2d(2, 2),
#             nn.Conv2d(c2, c2, (i+2, 1), 1, ((i+2) // 2, 0), groups=c2, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.Conv2d(c2, c2, (1, i+2), 1, (0, (i+2) // 2), groups=c2, bias=False),
#             nn.BatchNorm2d(c2),
#         )
#         self.branch1_4 = Conv(c2, c2, 3, p=i, d=i)
#
#     def forward(self, x):
#         B, C, H, W = x[1].shape
#         x1_0 = self.branch1_0(torch.cat(x, dim=1))
#         x1_1 = self.branch1_1(x1_0)
#         x1_2 = self.branch1_2(x1_0)
#         x1_3 = x1_1 * F.interpolate(self.gate_fn(x1_2), size=(H, W), mode='nearest')
#         x1 = self.branch1_4(x1_3)
#         return x1
#
#
# class MS_DCE(nn.Module):
#     # Revised from: Exploring Dense Context for Salient Object Detection, 2022, TCSVT
#     def __init__(self, in_channel, out_channel):
#         super(MS_DCE, self).__init__()
#         self.branch0 = Conv(in_channel // 2, in_channel, 1)
#
#         self.branch1 = MSDCEBranch(in_channel * 3 // 2, in_channel, 3)
#
#         self.branch2 = MSDCEBranch(in_channel * 3 // 2, in_channel, 5)
#
#         self.branch3 = MSDCEBranch(in_channel * 3 // 2, in_channel, 7)
#
#         self.branch4 = MSDCEBranch(in_channel * 3 // 2, in_channel, 9)
#
#         self.conv_cat = Conv(in_channel * 5, out_channel, 3, 1)
#
#     def forward(self, x):
#         x = list(torch.chunk(x, 2, 1))
#         x0 = self.branch0(x[0])
#         x1 = self.branch1([x0, x[1]])
#         x2 = self.branch2([x1, x[1]])
#         x3 = self.branch3([x2, x[1]])
#         x4 = self.branch4([x3, x[1]])
#
#         x_cat = torch.cat([x0, x1, x2, x3, x4], dim=1)
#         x_out = self.conv_cat(x_cat)
#         return x_out


class HTEM(nn.Module):
    # Revised from: Exploring Dense Context for Salient Object Detection, 2022, TCSVT
    def __init__(self, in_channel, out_channel):
        super(HTEM, self).__init__()
        self.branch0 = Conv(in_channel // 2, in_channel // 2, 1)

        self.branch1 = HTEMBranch(in_channel // 2, in_channel, 3)

        self.branch2 = HTEMBranch(in_channel // 2, in_channel, 5)

        self.branch3 = HTEMBranch(in_channel // 2, in_channel, 7)

        self.branch4 = HTEMBranch(in_channel // 2, in_channel, 9)

        self.conv_cat = Conv(in_channel * 5 // 2, out_channel, 3, 1)

    def forward(self, x):
        x = list(torch.chunk(x, 2, 1))
        x0 = self.branch0(x[0])
        x1 = self.branch1([x0, x[1]])
        x2 = self.branch2([x1, x[1]])
        x3 = self.branch3([x2, x[1]])
        x4 = self.branch4([x3, x[1]])

        x_cat = torch.cat((x0, x1, x2, x3, x4), dim=1)
        x_out = self.conv_cat(x_cat)
        return x_out


class InjectionMultiSum_Auto_pool(nn.Module):
    def __init__(self, c1: int, c2: int, global_inp=None) -> None:
        super().__init__()

        if not global_inp:
            global_inp = c1

        self.local_embedding = Conv(c1, c2, 1, act=False)
        self.global_embedding = Conv(global_inp, c2, 1, act=False)
        self.global_act = Conv(global_inp, c2, 1, act=False)
        # self.act = h_sigmoid()
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        x_g: global features
        x_l: local features
        """
        x_l, x_g = x
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        global_feat = self.global_embedding(x_g)

        out = local_feat * global_act + global_feat

        return out


class Fusion_2in(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.downmsample = nn.functional.adaptive_avg_pool2d
        self.local_embedding = Conv(c1[0] + c1[1], c2, 1)
        self.global_embedding = Conv(c1[2], c2, 1)
        self.global_act = Conv(c1[2], c2, 1)

    def forward(self, x):
        B, C, H, W = x[1].shape
        outputSize = (H // 2, W // 2)
        x0 = torch.cat((x[0], x[1]), dim=1)
        x_l = self.local_embedding(x0)
        x_g = self.global_embedding(x[2])
        x_g_act = self.global_act(x[2])

        x_l = self.downmsample(x_l, outputSize)
        out = x_l * x_g_act + x_g
        return out


class ALF(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.downsample = nn.AvgPool2d(2, 2)
        self.local_embedding = Conv(c1[1], c1[1], 1, 1)
        self.cat_conv = Conv(c1[0]+c1[1], c2, 1)

    def forward(self, x):
        x0 = self.downsample(x[0])
        x1 = self.local_embedding(x[1])
        x_cat = torch.cat((x0, x1), dim=1)
        out = self.cat_conv(x_cat)
        return out


# class Fusion_2in(nn.Module):
#     def __init__(self, c1, c2):
#         super().__init__()
#         self.downmsample = nn.functional.adaptive_avg_pool2d
#         self.cv1 = Conv(c1, c2, 1)
#
#     def forward(self, x):
#         B, C, H, W = x[1].shape
#         outputSize = (H // 2, W // 2)
#         x0 = self.downmsample(x[0], outputSize)
#         x1 = self.downmsample(x[1], outputSize)
#         x2 = torch.cat((x0, x1), dim=1)
#         x3 = self.cv1(x2)
#         return x3


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc ** 0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k ** 2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc ** 0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Contrastive Head for YOLO-World compute the region-text scores according to the similarity between image and text
    features.
    """

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        self.bias = nn.Parameter(torch.zeros([]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcut option, groups and expansion
        ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Rep CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class LCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1, k=5):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(C2f(c3 // 2, c4, n, shortcut=True), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(C2fk(c4, c4, n, shortcut=True, kl=k), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class Silence(nn.Module):
    """Silence."""

    def __init__(self):
        """Initializes the Silence module."""
        super(Silence, self).__init__()

    def forward(self, x):
        """Forward pass through Silence layer."""
        return x


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out
