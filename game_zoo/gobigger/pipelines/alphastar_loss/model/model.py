import os

import torch
import torch.nn as nn
import numpy as np
from bigrl.core.torch_utils.initialization import ortho_init_weights

from bigrl.core.utils.config_helper import read_config, deep_merge_dicts
from .encoder import Encoder
from .head import PolicyHead, ValueHead

default_config = read_config(os.path.join(os.path.dirname(__file__), 'default_model_config.yaml'))


class Model(nn.Module):
    def __init__(self, cfg={}, use_value_network=False):
        super(Model, self).__init__()
        self.whole_cfg = deep_merge_dicts(default_config, cfg)
        self.model_cfg = self.whole_cfg.model
        self.use_value_network = use_value_network
        self.encoder = Encoder(self.whole_cfg)
        self.policy_head = PolicyHead(self.whole_cfg)
        self.ortho_init = self.whole_cfg.model.get('ortho_init', False)
        if self.ortho_init:
            ortho_init_weights(self.encoder,gain=np.sqrt(2))
            ortho_init_weights(self.policy_head, gain=np.sqrt(2))
            ortho_init_weights(self.policy_head.output_layer,gain=0.01)
        if self.use_value_network:
            self.only_update_value = False
            self.value_networks = nn.ModuleDict()
            self.value_head_init_gains = self.whole_cfg.model.get('value_head_init_gains', {})
            for k in self.whole_cfg.agent.enable_baselines:
                self.value_networks[k] = ValueHead(self.whole_cfg)
                if self.ortho_init:
                    ortho_init_weights(self.value_networks[k],gain=np.sqrt(2))
                    ortho_init_weights(self.value_networks[k].output_layer,gain=self.value_head_init_gains.get(k,1))

        self.temperature = self.whole_cfg.agent.get('temperature', 1)


    # used in rl_train actor
    def forward(self, obs):
        action_mask = obs.pop('action_mask',None)
        embedding = self.encoder(obs, )
        logit = self.policy_head(embedding, temperature=self.temperature)
        if action_mask is not None:
            logit.masked_fill_(mask=action_mask,value=-1e9)


        dist = torch.distributions.Categorical(logits=logit)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return {'logit': logit, 'action_logp': action_log_prob, 'action': action, }

    # used in rl_eval actor
    def compute_action(self, obs, ):
        action_mask = obs.pop('action_mask',None)
        embedding = self.encoder(obs, )
        logit = self.policy_head(embedding, temperature=self.temperature)
        if action_mask is not None:
            logit.masked_fill_(mask=action_mask,value=-1e9)
        dist = torch.distributions.Categorical(logits=logit)
        action = dist.sample()
        return {'action': action, 'logit': logit}

    def sl_train(self, input_data):
        obs = input_data['obs']
        x = self.encoder(obs)
        logit = self.policy_head(x)
        return {'logit': logit, 'action': input_data['action']}

    def rl_train(self, input_data):
        obs = input_data['obs']
        obs = flatten_data(obs)
        action_mask = obs.pop('action_mask',None)
        embedding = self.encoder(obs)
        logit = self.policy_head(embedding, temperature=self.temperature)
        if action_mask is not None:
            logit.masked_fill_(mask=action_mask,value=-1e9)
        if self.only_update_value:
            critic_input = embedding.detach()
        else:
            critic_input = embedding

        values = {}
        for k, v in self.value_networks.items():
            values[k] = v(critic_input)
        return {'logit': logit,
                'action_logp': input_data['action_logp'],
                'value': values,
                'action': input_data['action'],
                'reward': input_data['reward'],
                'done': input_data['done'],
                'flatten_obs': obs,
                }

    # used in rl teacher model
    def teacher_forward(self, input_data):
        obs = input_data['flatten_obs']
        action_mask = obs.get('action_mask',None)
        embedding = self.encoder(obs, )
        logit = self.policy_head(embedding, temperature=self.temperature)
        if action_mask is not None:
            logit.masked_fill_(mask=action_mask,value=-1e9)
        input_data['teacher_logit'] = logit
        return input_data

def flatten_data(data):
    if isinstance(data, dict):
        return {k: flatten_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return torch.flatten(data, start_dim=0, end_dim=1)
