import os
import copy
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from bigrl.core.torch_utils.initialization import ortho_init_weights
from bigrl.core.torch_utils.detach import detach_grad
from bigrl.core.utils.config_helper import read_config, deep_merge_dicts
from .encoder import Encoder
from .head import PolicyHead, ValueHead, QMixer
from .value_encoder import ValueEncoder

default_config = read_config(os.path.join(os.path.dirname(__file__), 'default_model_config.yaml'))


class Model(nn.Module):
    def __init__(self, cfg={}, **kwargs):
        super(Model, self).__init__()
        self.whole_cfg = deep_merge_dicts(default_config, cfg)
        self.encoder = Encoder(self.whole_cfg)
        self.policy_head = PolicyHead(self.whole_cfg)
        self.only_update_value = False
        self.ortho_init = self.whole_cfg.model.get('ortho_init', True)
        if self.ortho_init:
            ortho_init_weights(self.encoder,gain=np.sqrt(2))
            ortho_init_weights(self.policy_head, gain=np.sqrt(2))
            ortho_init_weights(self.policy_head.output_layer,gain=0.01)

        self.use_value_feature = self.whole_cfg.agent.get('use_value_feature', True)
        if self.use_value_feature:
            self.value_head = ValueHead(self.whole_cfg)
            self.mixer = QMixer(self.whole_cfg)
            self.value_encoder = ValueEncoder(self.whole_cfg)
            ortho_init_weights(self.value_head, gain=np.sqrt(2))
            ortho_init_weights(self.value_head.output_layer, gain=1)
            ortho_init_weights(self.mixer, gain=np.sqrt(2))

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
    
    def compute_value(self, agent_qs, obs):
        B, player_num = obs.batch_size, obs.player_num
        obs = flatten_data(flatten_data(get_data_from_indices(obs['obs'], [slice(0, None), slice(None)])))
        value_embedding = self.value_encoder(obs['value_info'])
        value = self.mixer(agent_qs, value_embedding)
        return value

    def rl_forward(self, obs):
        B, max_t, player_num = obs.batch_size, obs.max_seq_length, obs.player_num
        obs = flatten_data(flatten_data(get_data_from_indices(obs['obs'], [slice(0, None), slice(None)])))
        embedding = self.encoder(obs)
        logit = self.policy_head(embedding)
        q_val = self.value_head(embedding)
        logit = nn.functional.softmax(logit, dim=-1)
        return logit.view(B, max_t, player_num, -1), q_val.view(B, max_t, player_num, -1)

    def _update_targets(self):
        self.target_value_head.load_state_dict(self.mixer.state_dict())

    def compute_logp_action(self, obs, **kwargs, ):
        embedding = self.encoder(obs)
        logit = self.policy_head(embedding)
        dist = torch.distributions.Categorical(logits=logit)
        action = dist.sample()
        action_log_probs = dist.log_prob(action)
        log_action_probs = action_log_probs
        value = self.mixer(embedding)
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
        values = self.mixer(critic_input, inputs['action'])

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

def flatten_data(data):
    if isinstance(data, dict):
        return {k: flatten_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return torch.flatten(data, start_dim=0, end_dim=1)


def get_data_from_indices(data, indices):
    if isinstance(data, dict):
        return {k: get_data_from_indices(v, indices) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data[indices]
