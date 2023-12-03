from typing import Tuple

import torch
from torch import nn


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalizes to 2-dimensional images.
    First proposed by "End-to-End Object Detection with Transformers". This is a 
    modified version taken from the LoFTR implementation.
    https://github.com/zju3dv/LoFTR/blob/master/src/loftr/utils/position_encoding.py
    """

    def __init__(self, dim: int, max_shape: Tuple[int, int] = (256, 256)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        temp = torch.tensor(10000.0)  # As in the end-to-end detection transformer paper

        pos_encoding = torch.zeros((dim, *max_shape))

        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)

        div_term = torch.pow(temp, torch.arange(0, dim//2, 2, dtype=torch.float32) / (dim//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]

        pos_encoding[0::4, :, :] = torch.sin(x_position / div_term)
        pos_encoding[1::4, :, :] = torch.cos(x_position / div_term)
        pos_encoding[2::4, :, :] = torch.sin(y_position / div_term)
        pos_encoding[3::4, :, :] = torch.cos(y_position / div_term)

        self.register_buffer('pos_encoding', pos_encoding.unsqueeze(0),
                             persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pos_encoding[:, :, :x.size(2), :x.size(3)]
