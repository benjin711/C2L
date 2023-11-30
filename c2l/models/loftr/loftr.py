from torch import nn


class LoFTR(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        pos_encoding: nn.Module
    ):
        super().__init__()

        self.backbone = backbone
        self.pos_encoding = pos_encoding

    def forward(self, x):
        return x
