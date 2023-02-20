import os
from typing import Any, Dict

import torch
import torch.nn as nn

from bigrl.core.torch_utils.detach import detach_grad
from bigrl.core.utils.config_helper import read_config, deep_merge_dicts
from .encoder import Encoder
from .head import PolicyHead, ValueHead

default_config = read_config(os.path.join(os.path.dirname(__file__), 'default_model_config.yaml'))


class Model(nn.Module):
    def __init__(self, cfg={}, use_value_network=False):
        super(Model, self).__init__()
        self.whole_cfg = deep_merge_dicts(default_config, cfg)
        self.model_cfg = self.whole_cfg.model
        self.encoder = Encoder(self.model_cfg.encoder)
        self.policy_head = PolicyHead(self.model_cfg.policy_head)
        self.use_value_network = use_value_network

        if self.use_value_network:
            self.value_head = ValueHead(self.model_cfg.value_head)
            self.only_update_value = False

    def forward(self, obs, temperature=0):
        embedding = self.encoder(obs)
        logit = self.policy_head(embedding)
        if temperature == 0:
            action = logit.argmax(dim=-1)
        else:
            logit = logit.div(temperature)
            dist = torch.distributions.Categorical(logits=logit)
            action = dist.sample()
        return {'action': action, 'logit': logit}

    def compute_logp_action(self, obs, **kwargs, ):
        embedding = self.encoder(obs)
        logit = self.policy_head(embedding)
        dist = torch.distributions.Categorical(logits=logit)
        action = dist.sample()
        action_log_probs = dist.log_prob(action)
        log_action_probs = action_log_probs
        return {'action': action,
                'action_logp': log_action_probs,
                'logit': logit, }

    def rl_train(self, inputs: dict, **kwargs) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - inputs (:obj:`dict`): Dict type data
        ArgumentsKeys:
            - obs shape     :math:`(T+1, B)`, where T is timestep, B is batch size
            - action_logp: behaviour logits, :math:`(T, B,action_size)`
            - action: behaviour actions, :math:`(T, B)`
            - reward: shape math:`(T, B)`
            - done:shape math:`(T, B)`
        Returns:
            - metric_dict (:obj:`Dict[str, Any]`):
              Including current total_loss, policy_gradient_loss, critic_loss and entropy_loss
        """

        obs = inputs['obs']
        # flat obs
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = torch.flatten(obs[k], start_dim=0, end_dim=1)  # ((T+1) * B)
        elif isinstance(obs, torch.Tensor):
            obs = torch.flatten(obs, start_dim=0, end_dim=1)

        embedding = self.encoder(obs, )
        logits = self.policy_head(embedding)
        critic_input = embedding
        if self.only_update_value:
            critic_input = detach_grad(critic_input)
        values = self.value_head(critic_input)

        outputs = {
            'value': values,
            'logit': logits,
            'action': inputs['action'],
            'action_logp': inputs['action_logp'],
            'reward': inputs['reward'],
            'done': inputs['done'],
        }
        return outputs
