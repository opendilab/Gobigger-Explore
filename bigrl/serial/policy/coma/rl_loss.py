import torch
from torch.distributions import Categorical
from bigrl.core.rl_utils.running_mean_std import RunningMeanStd
from bigrl.core.torch_utils.scheduler import get_schedule_fn
from .utils import RunningMeanStd
import numpy as np
from collections import defaultdict
from functools import partial

class ReinforcementLoss:
    def __init__(self, cfg: dict, ) -> None:
        self.whole_cfg = cfg

        # loss parameters
        self.loss_parameters = self.whole_cfg.agent.get('loss_parameters', {})
        self.gamma = self.loss_parameters.get('gamma', 0.99)
        self.td_lambda = self.loss_parameters.get('td_lambda', 0.95)
        self.reward_normalization = self.loss_parameters.get('reward_normalization', False)
        self.ret_rms = defaultdict(partial(RunningMeanStd))
        self._eps = 1e-8
        self.advantage_normalization = self.loss_parameters.get('advantage_normalization', False)
        self.critic_training_steps = 0 
        self.last_target_update_step = 0

        # loss weights
        self.loss_weights = self.whole_cfg.agent.get('loss_weights', {})
        self.policy_weight = self.loss_weights.get('policy', 1)
        self.value_weight = self.loss_weights.get('value', 0.5)
        self.entropy_weight = self.loss_weights.get('entropy', 0.01)

        self.only_update_value = False

    def compute_loss(self, inputs, agent):
        if (agent.player_num * agent.team_num) >= 4:
            step = 2
            if (agent.player_num * agent.team_num) >= 12:
                step = 1
            loss_info_dict = {
                'total_loss': 0.,
                'coma_loss': 0.,
                'critic_loss': 0.,
                'entropy': 0.,
                # 'value': values.mean().item(),
                'rewards': 0.,
                'advantage': 0.,
                'gradient': 0.,
                'critic_gradient': 0.,
            }
            back = False
            total_loss = 0.
            for i in range(0, inputs.batch_size, step):
                if i + step == inputs.batch_size:
                    back = True
                loss, info_dict = self._compute_loss(inputs[i: i + step], agent, back=back)
                for k, v in info_dict.items():
                    loss_info_dict[k] += v
            for k, v in loss_info_dict.items():
                if k != 'gradient' or k != 'critic_gradient':
                    loss_info_dict[k] /= (inputs.batch_size // step)
            total_loss /= (inputs.batch_size // step)
            return total_loss, loss_info_dict
        else:
            return self._compute_loss(inputs, agent)

    def _compute_loss(self, inputs, agent, back=True):
        # take data from inputs
        bs = inputs.batch_size
        max_t = inputs.max_seq_length
        actions = inputs['action']
        obs = inputs['obs']
        rewards = inputs['reward'][:, :-1].squeeze(-1)
        terminated = inputs["terminated"][:, :-1].float()
        mask = inputs["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        critic_mask = mask.clone()
        mask = mask.repeat(1, 1, (agent.player_num * agent.team_num)).view(-1)

        q_vals, critic_loss, critic_gradient = self._train_critic(inputs, rewards, terminated, actions, critic_mask, bs, max_t, agent, back)

        actions = actions[:,:-1]

        action_logits = agent.model.rl_forward(inputs)[:, :-1]  # Concat over time
        # Calculated baseline
        q_vals = q_vals.reshape(-1, agent.action_num)
        pi = action_logits.reshape(-1, agent.action_num)
        
        baseline = (pi * q_vals).sum(-1).detach()

        # Calculate policy grad with mask
        q_taken = torch.gather(q_vals, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken = torch.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = torch.log(pi_taken)

        advantages = (q_taken - baseline).detach()

        coma_loss = - ((advantages * log_pi_taken) * mask).sum() / mask.sum()
        dist_entropy = Categorical(pi).entropy().view(-1)
        dist_entropy[mask == 0] = 0 # fill nan
        entropy_loss = (dist_entropy * mask).sum() / mask.sum()

        # Optimise agents
        total_loss = coma_loss - self.entropy_weight * entropy_loss
        total_loss.backward()
        if back:
            gradient = agent.grad_clip.apply(agent.model.parameters())
            agent.optimizer.step()
            agent.optimizer.zero_grad()
        else:
            gradient = 0.

        if (self.critic_training_steps - self.last_target_update_step) / self.whole_cfg.learner.target_update_interval >= 1.0:
            agent._update_targets()
            self.last_target_update_step = self.critic_training_steps

        loss_info_dict = {
            'total_loss': total_loss.item(),
            'coma_loss': coma_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': -entropy_loss.item(),
            # 'value': values.mean().item(),
            'rewards': rewards.mean().item(),
            'advantage': advantages.mean().item(),
            'gradient': gradient,
            'critic_gradient': critic_gradient
        }
        return total_loss, loss_info_dict

    def _train_critic(self, batch, rewards, terminated, actions, mask, bs, max_t, agent, back=True):
        # Optimise critic
        target_q_vals = agent.target_model.compute_value(batch, actions)[:, :]
        targets_taken = torch.gather(target_q_vals, dim=3, index=actions).squeeze(3)
        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, targets_taken, (agent.player_num * agent.team_num), self.gamma, self.td_lambda).detach()
        
        if self.reward_normalization:
            self.ret_rms['score'].update(targets.cpu().reshape(-1).numpy())
            target_q_vals = target_q_vals * np.sqrt(self.ret_rms['score'].var + self._eps)
            targets = targets / np.sqrt(self.ret_rms['score'].var + self._eps)

        mask_t = mask.expand(-1, max_t - 1, (agent.player_num * agent.team_num))

        q_t = agent.model.compute_value(batch, actions)
        q_vals = q_t.view(bs, max_t, (agent.player_num * agent.team_num), agent.action_num)[:, :-1]
        q_taken = torch.gather(q_t, dim=3, index=actions[:, :-1]).squeeze(3).squeeze(1)

        if self.reward_normalization:
            q_taken = q_taken * np.sqrt(self.ret_rms['score'].var + self._eps)
        td_error = (q_taken - targets)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask_t

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask_t.sum() * 0.001
        loss.backward()
        if back:
            gradient = agent.critic_grad_clip.apply(agent.model.parameters())
            agent.critic_optimizer.step()
            agent.critic_optimizer.zero_grad()
        else:
            gradient = 0.
        self.critic_training_steps += 1
        
        return q_vals, loss, gradient
        # return q_vals, loss


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - torch.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]