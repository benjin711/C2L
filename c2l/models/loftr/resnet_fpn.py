from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from c2l.models.loftr.c2ltregressor1 import FeatureWithMask


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class ResNetFPN(nn.Module):
    """
    ResNet+FPN as in LoFTR paper
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        input_dim: int,
        initial_dim: int,
        block_dims: List[int],
        num_down: int,
        num_up: int
    ):
        """
        Args:
            input_dim (int): input dimension
            initial_dim (int): initial dimension
            block_dims (List[int]): list of dimensions for each block
            num_down (int): number of downsampling blocks excluding the 
                initial different downsampling block, the bottleneck is 
                of resolution 1/(2^num_down)
            num_up (int): number of upsampling blocks, the output is of 
                resolution 1/(2^(num_down - num_up))
        """

        super().__init__()
        # Config
        block = BasicBlock

        # Initial Conv 1/1 -> 1/2
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_dim, initial_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(initial_dim),
            nn.ReLU(inplace=True),
            self._make_layer(block, initial_dim, block_dims[0], stride=1)
        )

        self.down = nn.ModuleList()
        self.propagate = nn.ModuleList()
        self.up = nn.ModuleList()

        for i in range(num_down):
            self.down.append(
                self._make_layer(block, block_dims[i], block_dims[i + 1], stride=2)
            )

            if i >= num_down - num_up:
                self.propagate.append(
                    conv1x1(block_dims[i], block_dims[i + 1]),
                )
                self.up.append(nn.Sequential(
                    conv3x3(block_dims[i + 1], block_dims[i + 1]),
                    nn.BatchNorm2d(block_dims[i + 1]),
                    nn.LeakyReLU(),
                    conv3x3(block_dims[i + 1], block_dims[i]),
                ))

        self.up = self.up[::-1]

        # Bottleneck
        self.bottleneck = conv1x1(block_dims[num_down], block_dims[num_down])

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_c, out_c, stride=1):
        layer1 = block(in_c, out_c, stride=stride)
        layer2 = block(out_c, out_c, stride=1)
        return nn.Sequential(layer1, layer2)

    def _create_feature(self, x: torch.Tensor, x_mask: torch.Tensor) -> FeatureWithMask:
        """
        Args:
            x (torch.Tensor): feature tensor [N, C, H', W']
            x_mask (torch.Tensor): feature mask [N, H, W]
        Returns:
            out (FeatureWithMask): [N, C, H', W'], [N, H', W'] feature tensor and resized mask
        """
        if x_mask is not None:
            x_mask = x_mask.to(dtype=torch.uint8)
            x_mask = F.interpolate(x_mask[:, None, :, :], size=x.shape[-2:], mode='nearest')
            x_mask = x_mask.squeeze(1).to(dtype=torch.bool)

        return FeatureWithMask(
            feat=x,
            mask=x_mask,
        )

    def forward(
        self,
        img: FeatureWithMask,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            img (FeatureWithMask): image feature with mask [N, C, H, W]
        Returns:
            out (List[FeatureWithMask]): list of features with their masks
                at different resolutions
        """
        feat, mask = img.feat, img.mask
        feat = self.init_conv(feat)
        prop_out = []

        offset = len(self.down) - len(self.up)
        for i, down in enumerate(self.down):
            if i >= offset:
                prop_out.append(self.propagate[i - offset](feat))

            feat = down(feat)

        feat = self.bottleneck(feat)
        out = [self._create_feature(feat, mask)]

        prop_out.reverse()

        for i, up in enumerate(self.up):
            feat = F.interpolate(feat, scale_factor=2., mode='bilinear', align_corners=True)
            feat = up(feat + prop_out[i])

        out.append(self._create_feature(feat, mask))
        return out
