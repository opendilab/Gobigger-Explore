from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from bigrl.core.torch_utils.network import sequence_mask, ScatterConnection
from .network.encoder import SignBinaryEncoder, BinaryEncoder, OnehotEncoder, SignOnehotEncoder
from .network.nn_module import fc_block, conv2d_block, MLP
from .network.res_block import ResBlock
from .network.transformer import Transformer


class ValueEncoder(nn.Module):
    def __init__(self, cfg):
        super(ValueEncoder, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.value_encoder
        self.encode_modules = nn.ModuleDict()

        for k, item in self.cfg.modules.items():
            if item['arc'] == 'one_hot':
                self.encode_modules[k] = OnehotEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'binary':
                self.encode_modules[k] = BinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'sign_binary':
                self.encode_modules[k] = SignBinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'sign_one_hot':
                self.encode_modules[k] = SignOnehotEncoder(range=item['range'], )
            else:
                print(f'cant implement {k} for arc {item["arc"]}')
                raise NotImplementedError

        self.embedding_dim = self.cfg.embedding_dim
        self.encoder_cfg = self.cfg.encoder
        self.encode_layers = MLP(in_channels=self.encoder_cfg.input_dim,
                                 hidden_channels=self.encoder_cfg.hidden_dim,
                                 out_channels=self.embedding_dim,
                                 layer_num=self.encoder_cfg.layer_num,
                                 layer_fn=fc_block,
                                 activation=self.encoder_cfg.activation,
                                 norm_type=self.encoder_cfg.norm_type,
                                 use_dropout=False)
        # self.activation_type = self.cfg.activation

        self.transformer_cfg = self.cfg.transformer
        self.transformer = Transformer(
            n_heads=self.transformer_cfg.head_num,
            embedding_size=self.embedding_dim,
            ffn_size=self.transformer_cfg.ffn_size,
            n_layers=self.transformer_cfg.layer_num,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.0,
            activation=self.transformer_cfg.activation,
            variant=self.transformer_cfg.variant,
        )
        self.output_cfg = self.cfg.output
        self.output_fc = fc_block(self.embedding_dim,
                                  self.output_cfg.output_dim,
                                  norm_type=self.output_cfg.norm_type,
                                  activation=self.output_cfg.activation)

    def forward(self, x):
        embeddings = []
        player_num = x['player_num']
        mask = sequence_mask(player_num, max_len=x['view_x'].shape[1])
        for key, item in self.cfg.modules.items():
            assert key in x, f"{key} not implemented"
            x_input = x[key]
            embeddings.append(self.encode_modules[key](x_input))

        x = torch.cat(embeddings, dim=-1)
        x = self.encode_layers(x)
        x = self.transformer(x, mask=mask)
        team_info = self.output_fc(x.sum(dim=1) / player_num.unsqueeze(dim=-1))
        return team_info
