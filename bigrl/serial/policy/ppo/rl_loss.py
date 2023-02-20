import torch

from bigrl.core.rl_utils.running_mean_std import RunningMeanStd
from bigrl.core.torch_utils.scheduler import get_schedule_fn


class ReinforcementLoss:
    def __init__(self, cfg: dict, ) -> None:
        self.whole_cfg = cfg

        # loss parameters
        self.loss_parameters = self.whole_cfg.agent.get('loss_parameters', {})
        self.gamma = self.loss_parameters.get('gamma', 0.99)
        clip_range = self.loss_parameters.get('clip_range', 0.2)
        self.clip_range_schedule = get_schedule_fn(clip_range)
        clip_range_vf = self.loss_parameters.get('clip_range_vf', -1)
        self.clip_range_vf = -1
        self.clip_range_vf_schedule = get_schedule_fn(clip_range_vf) if (clip_range_vf and clip_range_vf != -1) else None
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
        behaviour_action_log_probs = inputs['action_logp']
        # rewards = inputs['reward']
        target_policy_logits = inputs['logit']
        values = inputs['value']
        old_values = inputs['old_value']
        returns = inputs['return']
        advantages = inputs['advantage']
        if self.clip_range_vf <= 0:
            # No clipping
            values_pred = values
        else:
            # Clip the different between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = old_values + torch.clamp(values - old_values, -self.clip_range_vf, self.clip_range_vf)

        if self.advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # get dist for behaviour policy and target policy
        pi_target = torch.distributions.Categorical(logits=target_policy_logits)
        target_action_log_probs = pi_target.log_prob(actions)
        entropy = pi_target.entropy()

        ratios = torch.exp(target_action_log_probs - behaviour_action_log_probs)

        # Policy-gradient loss.
        back1 = ratios * advantages
        back2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_gradient_loss = - torch.min(back1, back2).mean()

        # Critic loss.
        critic_loss = 0.5 * torch.pow(returns - values_pred, 2).mean()

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

    def set_progress(self, progress_remaining):
        self._current_progress_remaining = progress_remaining
        self.clip_range = self.clip_range_schedule(progress_remaining)
        if self.clip_range_vf_schedule:
            self.clip_range_vf = self.clip_range_vf_schedule(progress_remaining)
