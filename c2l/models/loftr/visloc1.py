from typing import Dict

from einops.einops import rearrange
from torch import nn


class VisLoc1(nn.Module):

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
        data.update({
            'bs': data['img'].size(0),
            'img_shape': data['img'].shape[1:],
            'pcl_shape': data['pcl'].shape[1:],
        })

        img = data['img']
        img_mask = data['mask_img'] if 'mask_img' in data else None

        (cfeat_img, cfeat_img_mask), _ = self.img_backbone(img, img_mask)  # [N, C, H, W]
        cfeat_img = rearrange(self.pos_encoding(cfeat_img), 'n c h w -> n (h w) c')  # [N, L, C]
        cfeat_img_mask = rearrange(cfeat_img_mask, 'n h w -> n (h w)')

        pcl = data['pcl']
        pcl_mask = data['mask_pcl'] if 'mask_pcl' in data else None
        feat_pcl, feat_pcl_mask = self.pcl_backbone(pcl, pcl_mask)  # [N, S, C]

        # 2. LoFTR Coarse
        cfeat_img, feat_pcl = self.loftr_coarse(cfeat_img, feat_pcl, cfeat_img_mask, feat_pcl_mask)

        # 3. Transformation Decoder
        trans, trans_unc, rot, rot_unc = self.transf_decoder(
            cfeat_img, feat_pcl, cfeat_img_mask, feat_pcl_mask)
        data.update({
            "trans": trans,
            "trans_unc": trans_unc,
            "rot": rot,
            "rot_unc": rot_unc,
        })

        return data
