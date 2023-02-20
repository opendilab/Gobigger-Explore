import torch
import torch.nn.functional as F
from bigrl.core.rl_utils.vtrace_util import vtrace_from_importance_weights
from bigrl.core.utils.config_helper import read_config, deep_merge_dicts
import os.path as osp
from .utils import RunningMeanStd
from bigrl.core.rl_utils.td_lambda import generalized_lambda_returns
import numpy as np
from collections import defaultdict
from functools import partial
default_rl_loss_config = read_config(osp.join(osp.dirname(__file__), "default_rl_loss_config.yaml"))


class ReinforcementLoss:
    def __init__(self, cfg: dict, ) -> None:
        self.whole_cfg = deep_merge_dicts(default_rl_loss_config, cfg)
        # loss parameters
        self.loss_parameters = self.whole_cfg.learner.get('loss_parameters', {})
        self.gammas = self.loss_parameters.get('gammas', {})
        self.lambda_ = self.loss_parameters.get('lambda', 0.8)
        self.clip_rho_threshold = self.loss_parameters.get('clip_rho_threshold', 1)
        self.clip_pg_rho_threshold = self.loss_parameters.get('clip_pg_rho_threshold', 1)
        self.fake_reward_norm = self.loss_parameters.get('fake_reward_normalization', True)
        self.score_reward_norm = self.loss_parameters.get('score_reward_normalization', False)
        self.dist_reward_norm = self.loss_parameters.get('dist_reward_normalization', True)
        self.ret_rms_dict = defaultdict(partial(RunningMeanStd))
        self._eps = 1e-8
        # loss weights
        self.loss_weights = self.whole_cfg.learner.get('loss_weights', {})
        self.policy_weights = self.loss_weights.policies
        self.value_weights = self.loss_weights.values
        self.entropy_weight = self.loss_weights.get('entropy', 0.01)
        self.kl_weight = self.loss_weights.get('kl',1)
        self.use_teacher = self.whole_cfg.agent.get('use_teacher', True)
        self.only_update_value = False



    def compute_loss(self, inputs):
        # take data from inputs
        actions = inputs['action']
        behaviour_action_log_probs = inputs['action_logp']
        logits = inputs['logit']
        rewards_dict = inputs['reward']
        values_dict = inputs['value']
        if self.use_teacher:
            teacher_logits = inputs['teacher_logit']

        unroll_len = actions.shape[0]
        batch_size = actions.shape[1]


        # get dist for target policy
        ## reshape logits and values
        logits = logits.reshape(unroll_len + 1, batch_size, -1)  # ((T+1), B,-1)
        target_policy_logits = logits[:-1]  # ((T), B,-1)
        pi_target = torch.distributions.Categorical(logits=target_policy_logits)
        target_action_log_probs = pi_target.log_prob(actions)

        # Entropy regulariser.
        entropy = pi_target.entropy()
        entropy_loss = - torch.mean(entropy)

        with torch.no_grad():
            log_rhos = target_action_log_probs - behaviour_action_log_probs

        total_critic_loss = 0
        total_policy_gradient_loss = 0
        loss_info_dict = defaultdict(float)

        for k,values in values_dict.items():
            reward_norm_flag = (k == 'score' and self.score_reward_norm) or (k!= 'score' and 'dist' not in k and self.fake_reward_norm) or ('dist' in k and self.dist_reward_norm)
            rewards = rewards_dict[k]
            discounts = (1 - inputs['done'].float()) * self.gammas[k]
            values = values.reshape(unroll_len + 1, batch_size)  # ((T+1), B)

            if reward_norm_flag:
                unnormalized_values = values * np.sqrt(self.ret_rms_dict[k].var + self._eps)
            else:
                unnormalized_values = values
            target_values, bootstrap_value = unnormalized_values[:-1], unnormalized_values[-1]  # ((T), B) ,(B)
            # Make sure no gradients backpropagated through the returned values.
            with torch.no_grad():
                vtrace_returns = vtrace_from_importance_weights(log_rhos, discounts, rewards, target_values,
                                                                bootstrap_value,
                                                                clip_rho_threshold=self.clip_rho_threshold,
                                                                clip_pg_rho_threshold=self.clip_pg_rho_threshold)

            # Policy-gradient loss.
            policy_gradient_loss = -torch.mean((target_action_log_probs * vtrace_returns.pg_advantages))

            # Critic loss.
            with torch.no_grad():
                unnormalized_returns = generalized_lambda_returns(rewards=rewards,
                                                                  pcontinues=discounts,
                                                                  state_values=target_values,
                                                                  bootstrap_value=bootstrap_value,
                                                                  lambda_=self.lambda_, )
            if reward_norm_flag:
                returns = unnormalized_returns/ np.sqrt(self.ret_rms_dict[k].var + self._eps)
                self.ret_rms_dict[k].update(unnormalized_returns.cpu().numpy())
            else:
                returns = unnormalized_returns
            critic_loss = 0.5 * torch.pow(returns - values[:-1], 2).mean()
            loss_info_dict[f'critic/{k}'] = critic_loss.item()
            loss_info_dict[f'policy/{k}'] = policy_gradient_loss.item()
            loss_info_dict[f'value/{k}'] = values.mean().item()


            total_critic_loss += critic_loss * self.value_weights[k]
            total_policy_gradient_loss += policy_gradient_loss * self.policy_weights[k]

        for k,val in rewards_dict.items():
            loss_info_dict[f'reward/{k}'] = val.mean().item()

        # kl loss
        if self.use_teacher:
            # get dist for teacher
            teacher_logits = teacher_logits.reshape(unroll_len + 1, batch_size, -1)  # ((T+1), B,-1)
            teacher_logits = teacher_logits[:-1]  # ((T), B,-1)
            pi_teacher = torch.distributions.Categorical(logits=teacher_logits)
            kl_loss = F.kl_div(input=pi_target.logits,target=pi_teacher.probs)
            loss_info_dict['kl'] = kl_loss.item()
        else:
            kl_loss = 0

        # Combine weighted sum of actor & critic losses.
        if self.only_update_value:
            total_loss = total_critic_loss
        else:
            total_loss = total_policy_gradient_loss + total_critic_loss + self.entropy_weight * entropy_loss + self.kl_weight * kl_loss
        loss_info_dict.update({
            'total_loss': total_loss.item(),
            'pg_loss': total_policy_gradient_loss.item(),
            'critic_loss': total_critic_loss.item(),
            'entropy': -entropy_loss.item(),
        })
        return total_loss, loss_info_dict
