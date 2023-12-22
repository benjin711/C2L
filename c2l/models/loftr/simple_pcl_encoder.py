from torch import nn

from c2l.models.loftr.c2ltregressor1 import FeatureWithMask


class SimplePCLEncoder(nn.Module):
    """ 
    Simple PCL Encoder. Wrapper for nn.Conv2d that takes a pcl and a mask as input.
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, pcl: FeatureWithMask) -> FeatureWithMask:
        """ 
        Args:
            pcl (FeatureWithMask): point cloud with mask
        Returns:
            FeatureWithMask: encoded point cloud with mask
        """
        pcl_feat = pcl.feat.permute(0, 2, 1).unsqueeze(-1)
        pcl_feat = self.conv2d(pcl_feat)
        pcl_feat = pcl_feat.squeeze(-1).permute(0, 2, 1)
        return FeatureWithMask(
            feat=pcl_feat,
            mask=pcl.mask,
        )
