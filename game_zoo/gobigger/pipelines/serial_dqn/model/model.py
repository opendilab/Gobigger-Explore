import os

import torch.nn as nn
from bigrl.core.utils.config_helper import read_config, deep_merge_dicts
from .encoder import Encoder
from .head import ValueHead
from bigrl.core.torch_utils.initialization import ortho_init_weights
import numpy as np

default_config = read_config(os.path.join(os.path.dirname(__file__), 'default_model_config.yaml'))


class Model(nn.Module):
    def __init__(self, cfg={}, **kwargs):
        super(Model, self).__init__()
        self.whole_cfg = deep_merge_dicts(default_config, cfg)
        self.encoder = Encoder(self.whole_cfg)
        self.q_head = ValueHead(self.whole_cfg)
        self.ortho_init = self.whole_cfg.model.get('ortho_init', True)
        if self.ortho_init:
            ortho_init_weights(self.encoder,gain=np.sqrt(2))
            ortho_init_weights(self.q_head, gain=np.sqrt(2))
            ortho_init_weights(self.q_head.output_layer,gain=1)

    def forward(self, obs, ):
        embedding = self.encoder(obs)
        q_values = self.q_head(embedding)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return {'action': action, 'q_value': q_values}
