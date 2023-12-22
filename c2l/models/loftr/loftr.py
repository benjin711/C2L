from torch import nn

from c2l.models.loftr.c2ltregressor1 import FeatureWithMask
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
        feat_img: FeatureWithMask,
        feat_pcl: FeatureWithMask,
    ):
        """
        Args:
            feat_img (FeatureWithMask): image feature [N, L, C] with mask (optional) [N, L]
            feat_pcl (FeatureWithMask): point cloud feature [N, S, C] with mask (optional) [N, S]

        Returns:
            feat_img (FeatureWithMask): image feature [N, L, C] with mask (optional) [N, L]
            feat_pcl (FeatureWithMask): point cloud feature [N, S, C] with mask (optional) [N, S]
        """
        next_layer_type = {'self': 'cross', 'cross': 'self'}
        layer_type = 'self'

        mask_img, feat_img = feat_img.mask, feat_img.feat
        mask_pcl, feat_pcl = feat_pcl.mask, feat_pcl.feat

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

        return (
            FeatureWithMask(feat=feat_img, mask=mask_img),
            FeatureWithMask(feat=feat_pcl, mask=mask_pcl)
        )
