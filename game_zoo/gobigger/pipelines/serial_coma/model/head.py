import torch
import torch.nn as nn

from bigrl.core.torch_utils.network.nn_module import fc_block
from bigrl.core.torch_utils.network.res_block import ResFCBlock
from .network.encoder import OnehotEncoder
from .value_encoder import ValueEncoder


class PolicyHead(nn.Module):
    def __init__(self, cfg):
        super(PolicyHead, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.policy

        self.embedding_dim = self.cfg.embedding_dim
        self.project_cfg = self.cfg.project
        self.project = fc_block(in_channels=self.project_cfg.input_dim,
                                out_channels=self.embedding_dim,
                                activation= self.project_cfg.activation,
                                norm_type=self.project_cfg.norm_type)

        self.resnet_cfg = self.cfg.resnet
        blocks = [ResFCBlock(in_channels=self.embedding_dim,
                             activation=self.resnet_cfg.activation,
                             norm_type=self.resnet_cfg.norm_type)
                  for _ in range(self.resnet_cfg.res_num)]
        self.resnet = nn.Sequential(*blocks)

        self.direction_num = self.whole_cfg.agent.features.get('direction_num', 12)
        self.action_num = 2 * self.direction_num + 3
        self.output_layer = fc_block(in_channels=self.embedding_dim,
                                  out_channels=self.action_num,
                                  norm_type=None,
                                  activation=None)

    def forward(self, x, temperature=1):
        x = self.project(x)
        x = self.resnet(x)
        logit = self.output_layer(x)
        logit /= temperature
        return logit


class ValueHead(nn.Module):
    def __init__(self, cfg):
        super(ValueHead, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.value
        self.num_player = self.whole_cfg.env.player_num_per_team * self.whole_cfg.env.team_num

        self.embedding_dim = self.cfg.embedding_dim
        self.action_num_embed = self.whole_cfg.model.scalar_encoder.modules.last_action_type.num_embeddings
        self.action_encoder = OnehotEncoder(num_embeddings=self.action_num_embed)
        self.project_cfg = self.cfg.project
        self.project = fc_block(in_channels=self.project_cfg.input_dim + self.action_num_embed * self.num_player + 32 ,
                                out_channels=self.embedding_dim,
                                activation= self.project_cfg.activation,
                                norm_type=self.project_cfg.norm_type)

        self.resnet_cfg = self.cfg.resnet
        blocks = [ResFCBlock(in_channels=self.embedding_dim,
                             activation=self.resnet_cfg.activation,
                             norm_type=self.resnet_cfg.norm_type)
                  for _ in range(self.resnet_cfg.res_num)]
        self.resnet = nn.Sequential(*blocks)

        self.output_layer = fc_block(in_channels=self.embedding_dim,
                                     out_channels=self.action_num_embed,
                                     norm_type=None,
                                     activation=None)
        self.value_encoder = ValueEncoder(self.whole_cfg)
    
    def forward(self, x, actions, states, t=None):
        states = self.value_encoder(states)
        x = self._build_inputs(x, actions, states, t=t)
        x = self.project(x)
        x = self.resnet(x)
        x = self.output_layer(x)
        x = x.squeeze(1)
        return x

    def _build_inputs(self, x, actions, states, t=None):
        bs = actions.shape[0]
        max_t = actions.shape[1] if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = []

        # observation
        if t:
            inputs.append(x[:, 0]) # no hidden state, so only calculate one step
            inputs.append(states[:, 0])
        else:
            inputs.append(x[:, ts])
            inputs.append(states[:, ts])

        # actions (masked out by agent)
        actions = self.action_encoder(actions)
        actions = actions[:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.num_player, 1)
        agent_mask = (1 - torch.eye(self.num_player, device=x.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.action_num_embed).view(self.num_player, -1)
        inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))

        inputs = torch.cat([x.reshape(bs, max_t, self.num_player, -1) for x in inputs], dim=-1)
        return inputs
