import os
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from bigrl.core.torch_utils.initialization import ortho_init_weights
from bigrl.core.torch_utils.detach import detach_grad
from bigrl.core.utils.config_helper import read_config, deep_merge_dicts
from .encoder import Encoder
from .head import PolicyHead, ValueHead

default_config = read_config(os.path.join(os.path.dirname(__file__), 'default_model_config.yaml'))


class Model(nn.Module):
    def __init__(self, cfg={}, **kwargs):
        super(Model, self).__init__()
        self.whole_cfg = deep_merge_dicts(default_config, cfg)
        self.encoder = Encoder(self.whole_cfg)
        self.policy_head = PolicyHead(self.whole_cfg)
        self.value_head = ValueHead(self.whole_cfg)
        self.only_update_value = False
        self.ortho_init = self.whole_cfg.model.get('ortho_init', True)
        if self.ortho_init:
            ortho_init_weights(self.encoder,gain=np.sqrt(2))
            ortho_init_weights(self.policy_head, gain=np.sqrt(2))
            ortho_init_weights(self.value_head,gain=np.sqrt(2))
            ortho_init_weights(self.policy_head.output_layer,gain=0.01)
            ortho_init_weights(self.value_head.output_layer,gain=1)

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

    def compute_value(self, obs, ):
        embedding = self.encoder(obs)
        value = self.value_head(embedding)
        return {'value': value}

    def compute_logp_action(self, obs, **kwargs, ):
        embedding = self.encoder(obs)
        logit = self.policy_head(embedding)
        dist = torch.distributions.Categorical(logits=logit)
        action = dist.sample()
        action_log_probs = dist.log_prob(action)
        log_action_probs = action_log_probs
        value = self.value_head(embedding)
        return {'action': action,
                'action_logp': log_action_probs,
                'logit': logit,
                'value': value,
                }

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
            # 'reward': inputs['reward'],
            # 'done': inputs['done'],
            'old_value': inputs['old_value'],
            'advantage': inputs['advantage'],
            'return': inputs['return'],
        }
        return outputs

