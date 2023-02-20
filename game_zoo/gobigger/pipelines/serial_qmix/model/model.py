from gc import collect
import os

import torch.nn as nn
from bigrl.core.utils.config_helper import read_config, deep_merge_dicts
from .encoder import Encoder
from .head import ValueHead
import torch
from .mixer import Mixer

default_config = read_config(os.path.join(os.path.dirname(__file__), 'default_model_config.yaml'))


class Model(nn.Module):
    def __init__(self, cfg={}, **kwargs):
        super(Model, self).__init__()
        self.whole_cfg = deep_merge_dicts(default_config, cfg)
        self.encoder = Encoder(self.whole_cfg)
        self.q_head = ValueHead(self.whole_cfg)
        
        # qmix
        self.mixer = Mixer(self.whole_cfg)
        self._global_state_encoder = nn.Identity()
        
        self.action_num = self.whole_cfg.agent.action_num
        self.player_num = self.whole_cfg.env.player_num_per_team
        self.team_num = self.whole_cfg.env.team_num

    def forward(self, obs):
        embedding = self.encoder(obs)
        batch_size = embedding.shape[0] // self.player_num
        agent_q = self.q_head(embedding)
        agent_q = agent_q.reshape(batch_size, self.player_num, self.action_num)
        action = agent_q.argmax(dim=-1)
        agent_q_act = torch.gather(agent_q, dim=-1, index=action.unsqueeze(-1))
        agent_q_act = agent_q_act.squeeze(-1)
        
        #TODO, need optim
        global_state = embedding.reshape(batch_size, -1)
        global_state_embedding = self._global_state_encoder(global_state)
        total_q = self.mixer(agent_q_act, global_state_embedding)
        return {'action': action.reshape(-1), 'total_q': total_q, 'logit': agent_q}
    

    def forward_collect(self, obs):
        embedding = self.encoder(obs)
        batch_size = embedding.shape[0] // self.player_num  // self.team_num
        agent_q = self.q_head(embedding)
        agent_q = agent_q.reshape(batch_size*self.team_num, self.player_num, self.action_num)
        action = agent_q.argmax(dim=-1)
        agent_q_act = torch.gather(agent_q, dim=-1, index=action.unsqueeze(-1))
        agent_q_act = agent_q_act.squeeze(-1)  # rollout_nstep, bs, agent_num
        
        #TODO, need optim
        global_state = embedding.reshape(batch_size*self.team_num, -1)
        global_state_embedding = self._global_state_encoder(global_state)
        total_q = self.mixer(agent_q_act, global_state_embedding) # rollout_nstep, batch_size*self.team_num
        return {'action': action.reshape(-1), 'total_q': total_q, 'logit': agent_q}
                