from typing import Dict

from einops.einops import rearrange
from torch import nn


class LoFTR(nn.Module):

    def __init__(
        self,
        img_backbone: nn.Module,
        pcl_backbone: nn.Module,
        pos_encoding: nn.Module,
        loftr_coarse: nn.Module,
    ):
        super().__init__()

        self.backbone = img_backbone
        self.pcl_backbone = pcl_backbone
        self.pos_encoding = pos_encoding
        self.loftr_coarse = loftr_coarse

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
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        cfeat_img, ffeat_img = self.backbone(data['img'])  # pylint: disable=unused-variable
        cfeat_img = rearrange(self.pos_encoding(cfeat_img), 'n c h w -> n (h w) c')

        feat_pcl = self.pcl_backbone(data['pcl'])

        # 2. LoFTR Coarse
        mask_img = data['mask_img'].flatten(-2) if 'mask_img' in data else None
        mask_pcl = data['mask_pcl'] if 'mask_pcl' in data else None
        cfeat_img, feat_pcl = self.loftr_coarse(cfeat_img, feat_pcl, mask_img, mask_pcl)
