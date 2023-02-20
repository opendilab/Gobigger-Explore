import torch

from bigrl.core.rl_utils.td_lambda import generalized_lambda_returns
from bigrl.core.rl_utils.vtrace_util import vtrace_from_importance_weights
from bigrl.core.rl_utils.running_mean_std import RunningMeanStd
import numpy as np


class ReinforcementLoss:
    def __init__(self, cfg: dict, ) -> None:
        self.whole_cfg = cfg

        # loss parameters
        self.loss_parameters = self.whole_cfg.learner.get('loss_parameters',{})
        self.gamma = self.loss_parameters.get('gamma', 0.99)
        self.lambda_ = self.loss_parameters.get('lambda', 0.8)
        self.clip_rho_threshold = self.loss_parameters.get('clip_rho_threshold', 1)
        self.clip_pg_rho_threshold = self.loss_parameters.get('clip_pg_rho_threshold', 1)
        self.reward_normalization = self.loss_parameters.get('reward_normalization', False)
        if self.reward_normalization:
            self.ret_rms= RunningMeanStd()
            self._eps = 1e-8

        # loss weights
        self.loss_weights = self.whole_cfg.learner.get('loss_weights',{})
        self.policy_weight = self.loss_weights.get('policy', 1)
        self.value_weight = self.loss_weights.get('value', 0.5)
        self.entropy_weight = self.loss_weights.get('entropy', 0.01)

        self.only_update_value = False

    def compute_loss(self, inputs):
        # take data from inputs
        actions = inputs['action']
        behaviour_action_log_probs = inputs['action_logp']
        rewards = inputs['reward']
        logits = inputs['logit']
        values = inputs['value']
        discounts = (1 - inputs['done'].float()) * self.gamma

        # behaviour_policy_logits = inputs['logit']
        # pi_behaviour = torch.distributions.Categorical(logits=behaviour_policy_logits)

        unroll_len = rewards.shape[0]
        batch_size = rewards.shape[1]

        # reshape logits and values
        logits = logits.reshape(unroll_len + 1, batch_size, -1)  # ((T+1), B,-1)
        target_policy_logits = logits[:-1]  # ((T), B,-1)
        values = values.reshape(unroll_len + 1, batch_size)  # ((T+1), B)

        if self.reward_normalization:
            unnormalized_values = values * np.sqrt(self.ret_rms.var + self._eps)
        else:
            unnormalized_values = values
        target_values, bootstrap_value = unnormalized_values[:-1], unnormalized_values[-1]

        # get dist for behaviour policy and target policy
        pi_target = torch.distributions.Categorical(logits=target_policy_logits)
        target_action_log_probs = pi_target.log_prob(actions)
        entropy = pi_target.entropy()

        with torch.no_grad():
            log_rhos = target_action_log_probs - behaviour_action_log_probs

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
        if self.reward_normalization:
            returns = unnormalized_returns/ np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns.cpu().numpy())
        else:
            returns = unnormalized_returns
        critic_loss = 0.5 * torch.pow(returns - values[:-1], 2).mean()

        # Entropy regulariser.
        entropy_loss = - torch.mean(entropy)

        # Combine weighted sum of actor & critic losses.
        if self.only_update_value:
            total_loss = self.value_weight * critic_loss
        else:
            total_loss = self.policy_weight * policy_gradient_loss + self.value_weight * critic_loss + self.entropy_weight * entropy_loss

        loss_info_dict = {
            'total_loss': total_loss.item(),
            'pg_loss': policy_gradient_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': -entropy_loss.item(),
            'reward': rewards.mean().item(),
            'value': values.mean().item(),
        }
        return total_loss, loss_info_dict
