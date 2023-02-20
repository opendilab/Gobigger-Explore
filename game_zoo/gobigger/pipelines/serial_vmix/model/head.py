import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.output_layer = fc_block(in_channels=self.embedding_dim,
                                  out_channels=1,
                                  norm_type=None,
                                  activation=None)
    def forward(self, x):
        x = self.project(x)
        x = self.resnet(x)
        x = self.output_layer(x)
        x = x.squeeze(1)
        return x


class QMixer(nn.Module):
    def __init__(self, cfg):
        super(QMixer, self).__init__()

        self.whole_cfg = cfg
        self.num_player = self.whole_cfg.env.player_num_per_team * self.whole_cfg.env.team_num
        self.state_dim = self.whole_cfg.model.value_encoder.output.output_dim

        self.embed_dim = 32
        self.abs = True
        
        hypernet_embed = 64
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim * self.num_player, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.num_player))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim * self.num_player, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim * self.num_player, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim * self.num_player, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))
        

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.num_player * self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.num_player)
        # First layer
        w1 = self.hyper_w_1(states).abs() if self.abs else self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.num_player, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        
        # Second layer
        w_final = self.hyper_w_final(states).abs() if self.abs else self.hyper_w_final(states)
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        
        return q_tot

    def k(self, states):
        bs = states.size(0)
        w1 = torch.abs(self.hyper_w_1(states))
        w_final = torch.abs(self.hyper_w_final(states))
        w1 = w1.view(-1, self.num_player, self.embed_dim)
        w_final = w_final.view(-1, self.embed_dim, 1)
        k = torch.bmm(w1,w_final).view(bs, -1, self.num_player)
        k = k / torch.sum(k, dim=2, keepdim=True)
        return k

    def b(self, states):
        bs = states.size(0)
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim)
        v = self.V(states).view(-1, 1, 1)
        b = torch.bmm(b1, w_final) + v
        return b
