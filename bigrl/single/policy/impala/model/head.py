import torch
from torch import nn


class PolicyHead(nn.Module):

    def __init__(self, head_cfg) -> None:
        super(PolicyHead, self).__init__()
        self.cfg = head_cfg

    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError
        return x


class ValueHead(nn.Module):

    def __init__(self, head_cfg) -> None:
        super(ValueHead, self).__init__()
        self.cfg = head_cfg

    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError
        x = x.squeeze(-1)
        return x
