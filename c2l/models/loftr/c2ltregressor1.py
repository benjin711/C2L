from dataclasses import dataclass
from typing import Dict

import torch
from einops.einops import rearrange
from torch import nn


@dataclass
class FeatureWithMask:
    feat: torch.Tensor
    mask: torch.Tensor


class C2LTRegressor1(nn.Module):

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        img_backbone: nn.Module,
        pcl_backbone: nn.Module,
        pos_encoding: nn.Module,
        loftr_coarse: nn.Module,
        transf_decoder: nn.Module,
    ):
        """
        Args:
            img_backbone (nn.Module): image backbone
            pcl_backbone (nn.Module): point cloud backbone
            pos_encoding (nn.Module): positional encoding
            loftr_coarse (nn.Module): LoFTR Coarse
            transf_decoder (nn.Module): transformation decoder
        """
        super().__init__()

        self.img_backbone = img_backbone
        self.pcl_backbone = pcl_backbone
        self.pos_encoding = pos_encoding
        self.loftr_coarse = loftr_coarse
        self.transf_decoder = transf_decoder

    def forward(self, data: Dict):
        """ 
        Update:
            data (dict): {
                'img': (torch.Tensor): (N, 3, H, W)
                'pcl': (torch.Tensor): (N, S, C)
                'mask_img'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask_pcl'(optional) : (torch.Tensor): (N, S)
            }
        """
        # 1. Backbones
        img = FeatureWithMask(
            feat=data['img'],
            mask=data['mask_img'] if 'mask_img' in data else None,
        )

        coarse_img, _ = self.img_backbone(img)  # [N, C, H, W]
        coarse_img.feat = rearrange(self.pos_encoding(coarse_img.feat),
                                    'n c h w -> n (h w) c')  # [N, L, C]
        if coarse_img.mask is not None:
            coarse_img.mask = rearrange(coarse_img.mask, 'n h w -> n (h w)')

        pcl = FeatureWithMask(
            feat=data['pcl'],
            mask=data['mask_pcl'] if 'mask_pcl' in data else None,
        )
        pcl = self.pcl_backbone(pcl)  # [N, S, C]

        # 2. LoFTR Coarse
        coarse_img, pcl = self.loftr_coarse(coarse_img, pcl)

        # 3. Transformation Decoder
        twu = self.transf_decoder(coarse_img, pcl)
        data.update({
            "trans": twu.trans,
            "trans_unc": twu.trans_unc,
            "rot": twu.rot,
            "rot_unc": twu.rot_unc,
        })

        return data
