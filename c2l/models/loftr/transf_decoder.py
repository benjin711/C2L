from dataclasses import dataclass

import torch
from torch import nn

from c2l.models.loftr.encoder_layer import LoFTREncoderLayer
from c2l.models.loftr.visloc1 import FeatureWithMask


@dataclass
class TransformationWithUncertainty:
    trans: torch.Tensor  # [N, 3]
    trans_unc: torch.Tensor  # [N, 3]
    rot: torch.Tensor  # [N, 4]
    rot_unc: torch.Tensor  # [N, 1]


class TransformationDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        nhead: int,
        nlayers: int,
        heads: nn.ModuleDict,
    ):
        """
        Args:
            dim (int): input dimension of query, key, value
            nhead (int): number of heads
            nlayers (int): number of cross attention layers
            heads: dictionary of heads for translation and rotation
        """
        super().__init__()
        self.layers = nn.ModuleList([
            LoFTREncoderLayer(dim, nhead) for _ in range(nlayers)
        ])

        self.heads = nn.ModuleDict({
            "trans": heads["trans"],
            "rot": heads["rot"],
        })

        # Learnable translation and rotation queries
        self.transf_queries = nn.Parameter(torch.zeros(2, dim))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        img_feat: FeatureWithMask,
        pcl_feat: FeatureWithMask,
    ) -> TransformationWithUncertainty:
        """
        Args:
            img_feat (FeatureWithMask): image feature [N, L, C] with mask (optional) [N, L]
            pcl_feat (FeatureWithMask): point cloud feature [N, S, C] with mask (optional) [N, S]
        Returns:
            (TransformationWithUncertainty): transformation with uncertainty
        """
        N = img_feat.feat.size(0)

        q = self.transf_queries.unsqueeze(0).repeat(N, 1, 1)  # [N, 2, C]
        if img_feat.mask is not None:
            img_feat.feat = img_feat.feat * img_feat.mask[:, :, None]
        if pcl_feat.mask is not None:
            pcl_feat.feat = pcl_feat.feat * pcl_feat.mask[:, :, None]

        feat = {0: img_feat.feat, 1: pcl_feat.feat}

        for i, layer in enumerate(self.layers):
            q = layer(q, feat[i % 2])  # [N, 2, C]

        trans_out = self.heads["trans"](q[:, 0])  # [N, 6]
        rot_out = self.heads["rot"](q[:, 1])  # [N, 5]

        return TransformationWithUncertainty(
            trans=trans_out[:, :3],
            trans_unc=torch.sigmoid(trans_out[:, 3:]),
            rot=rot_out[:, :4] / rot_out[:, :4].norm(dim=-1, keepdim=True),
            rot_unc=torch.sigmoid(rot_out[:, 4:]),
        )
