import torch
from torch import nn

from c2l.models.loftr.encoder_layer import LoFTREncoderLayer


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, dim: int, nhead: int, nlayers: int):
        """
        Args:
            dim (int): dimension of query, key, and value
            nhead (int): number of attention heads of a single layer
            nlayers (int): number of layers
        """
        super().__init__()

        self.layers = nn.ModuleDict({
            'img': nn.ModuleList(),
            'pcl': nn.ModuleList(),
        })

        for _ in range(nlayers):
            self.layers['img'].append(LoFTREncoderLayer(dim, nhead))
            self.layers['pcl'].append(LoFTREncoderLayer(dim, nhead))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        feat_img: torch.Tensor,
        feat_pcl: torch.Tensor,
        mask_img: torch.Tensor = None,
        mask_pcl: torch.Tensor = None
    ):
        """
        Args:
            feat_img (torch.Tensor): [N, L, C]
            feat_pcl (torch.Tensor): [N, S, C]
            mask_img (torch.Tensor): [N, L] (optional)
            mask_pcl (torch.Tensor): [N, S] (optional)

        Returns:
            feat_img (torch.Tensor): [N, L, C]
            feat_pcl (torch.Tensor): [N, S, C]
        """
        next_layer_type = {'self': 'cross', 'cross': 'self'}
        layer_type = 'self'

        for img_layer, pcl_layer in zip(self.layers['img'], self.layers['pcl']):

            if layer_type == 'self':
                feat_img = img_layer(x=feat_img, source=feat_img,
                                     x_mask=mask_img, source_mask=mask_img)
                feat_pcl = pcl_layer(x=feat_pcl, source=feat_pcl,
                                     x_mask=mask_pcl, source_mask=mask_pcl)
            else:
                feat_img = img_layer(x=feat_img, source=feat_pcl,
                                     x_mask=mask_img, source_mask=mask_pcl)
                feat_pcl = pcl_layer(x=feat_pcl, source=feat_img,
                                     x_mask=mask_pcl, source_mask=mask_img)

            layer_type = next_layer_type[layer_type]

        return feat_img, feat_pcl
