import torch

from bigrl.core.rl_utils.running_mean_std import RunningMeanStd
import torch.nn.functional as F

class ReinforcementLoss:
    def __init__(self, cfg: dict, ) -> None:
        self.whole_cfg = cfg

        # loss parameters
        self.loss_parameters = self.whole_cfg.agent.get('loss_parameters', {})
        self.gamma = self.loss_parameters.get('gamma', 0.99)
        self.gae_lambda = self.loss_parameters.get('gae_lambda', 0.95)
        self.reward_normalization = self.loss_parameters.get('reward_normalization', False)
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8
        self.advantage_normalization = self.loss_parameters.get('advantage_normalization', False)

        # loss weights
        self.loss_weights = self.whole_cfg.agent.get('loss_weights', {})
        self.policy_weight = self.loss_weights.get('policy', 1)
        self.value_weight = self.loss_weights.get('value', 0.5)
        self.entropy_weight = self.loss_weights.get('entropy', 0.01)

        self.only_update_value = False

    def compute_loss(self, inputs):
        # take data from inputs
        actions = inputs['action']
        # rewards = inputs['reward']
        target_policy_logits = inputs['logit']
        values = inputs['value']
        returns = inputs['return']
        advantages = inputs['advantage']

        if self.advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # get dist for behaviour policy and target policy
        pi_target = torch.distributions.Categorical(logits=target_policy_logits)
        target_action_log_probs = pi_target.log_prob(actions)
        entropy = pi_target.entropy()

        # Policy-gradient loss.
        policy_gradient_loss = - (target_action_log_probs * advantages).mean()

        # Critic loss.
        critic_loss = F.mse_loss(returns, values, )

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
            'value': values.mean().item(),
            'return': returns.mean().item(),
            'advantage': advantages.mean().item(),
        }
        return total_loss, loss_info_dict
