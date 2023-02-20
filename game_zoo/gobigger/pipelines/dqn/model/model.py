import os

import torch.nn as nn
from bigrl.core.utils.config_helper import read_config, deep_merge_dicts
from .encoder import Encoder
from .head import ValueHead

default_config = read_config(os.path.join(os.path.dirname(__file__), 'default_model_config.yaml'))


class Model(nn.Module):
    def __init__(self, cfg={}, **kwargs):
        super(Model, self).__init__()
        self.whole_cfg = deep_merge_dicts(default_config, cfg)
        self.encoder = Encoder(self.whole_cfg)
        self.q_head = ValueHead(self.whole_cfg)

    def forward(self, obs, ):
        embedding = self.encoder(obs)
        q_values = self.q_head(embedding)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return {'action': action, 'q_value': q_values}
