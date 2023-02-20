import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, encoder_cfg) -> None:
        super(Encoder, self).__init__()
        self.cfg = encoder_cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
