import torch
from torch import nn



class ValueHead(nn.Module):

    def __init__(self, head_cfg) -> None:
        super(ValueHead, self).__init__()
        self.cfg = head_cfg

    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError
        return x
