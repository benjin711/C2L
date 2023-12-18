import torch
from torch import nn

from c2l.models.loftr.linear_attention import LinearAttention


class LoFTREncoderLayer(nn.Module):
    def __init__(
        self,
        orig_dim: int,
        nhead: int
    ):
        """
        Args:
            orig_dim (int): original dimension of input
            nhead (int): number of attention heads
        """
        super().__init__()

        self.dim = orig_dim // nhead
        self.nhead = nhead

        # multi-head attention
        self.linear = nn.ModuleDict({
            "q_proj": nn.Linear(orig_dim, orig_dim, bias=False),
            "k_proj": nn.Linear(orig_dim, orig_dim, bias=False),
            "v_proj": nn.Linear(orig_dim, orig_dim, bias=False),
        })

        self.attention = LinearAttention()
        self.merge = nn.Sequential(
            nn.Linear(orig_dim, orig_dim, bias=False),
            nn.LayerNorm(orig_dim)
        )

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(orig_dim*2, orig_dim*2, bias=False),
            nn.ReLU(True),
            nn.Linear(orig_dim*2, orig_dim, bias=False),
            nn.LayerNorm(orig_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        x_mask: torch.Tensor = None,
        source_mask: torch.Tensor = None
    ):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.linear["q_proj"](query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.linear["k_proj"](key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.linear["v_proj"](value).view(bs, -1, self.nhead, self.dim)

        message = self.attention(query, key, value, q_mask=x_mask,
                                 kv_mask=source_mask)  # [N, L, (H, D)]

        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))

        return x + message
