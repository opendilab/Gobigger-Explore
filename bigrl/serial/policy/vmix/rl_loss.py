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
        self.reward_normalization = self.loss_parameters.get('reward_normalization', True)
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
        if (agent.player_num * agent.team_num) > 4:
            loss_info_dict = {
                'total_loss': 0.,
                'pg_loss': 0.,
                'vf_loss': 0.,
                'entropy': 0.,
                # 'value': values.mean().item(),
                'rewards': 0.,
                'advantage': 0.,
                'gradient': 0.,
                # 'critic_gradient': critic_gradient
            }
            back = False
            total_loss = 0.
            for i in range(0, inputs.batch_size, 2):
                if i + 2 == inputs.batch_size:
                    back = True
                loss, info_dict = self._compute_loss(inputs[i: i + 2], agent, back=back)
                for k, v in info_dict.items():
                    loss_info_dict[k] += v
            for k, v in loss_info_dict.items():
                if k != 'gradient':
                    loss_info_dict[k] /= (inputs.batch_size // 2)
            total_loss /= (inputs.batch_size // 2)
            return total_loss, loss_info_dict
        else:
            return self._compute_loss(inputs, agent)

    def _compute_loss(self, inputs, agent, back=True):
        # take data from inputs
        bs = inputs.batch_size
        max_t = inputs.max_seq_length
        rewards = inputs['reward'][:, :-1].squeeze(-1)
        actions = inputs['action']
        terminated = inputs["terminated"][:, :-1].float()
        mask = inputs["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        obs = inputs['obs']

        critic_mask = mask.clone()
        mask = mask.repeat(1, 1, (agent.player_num * agent.team_num)).view(-1)

        advantages, td_error, targets_taken, log_pi_taken, entropy = self._calculate_advs(inputs, rewards, terminated, actions, critic_mask, bs, max_t, agent)

        pg_loss = - ((advantages.detach() * log_pi_taken) * mask).sum() / mask.sum()
        vf_loss = ((td_error ** 2) * mask).sum() / mask.sum()
        entropy[mask == 0] = 0
        entropy_loss = (entropy * mask).sum() / mask.sum()

        total_loss = pg_loss + 0.1 * vf_loss - self.entropy_weight * entropy_loss

        # Optimise agents
        total_loss.backward()
        
        if back:
            gradient = agent.grad_clip.apply(agent.model.parameters())
            agent.optimizer.step()
            agent.optimizer.zero_grad()

        loss_info_dict = {
            'total_loss': total_loss.item(),
            'pg_loss': pg_loss.item(),
            'vf_loss': vf_loss.item(),
            'entropy': -entropy_loss.item(),
            # 'value': values.mean().item(),
            'rewards': rewards.mean().item(),
            'advantage': advantages.mean().item(),
            'gradient': gradient if back else 0,
            # 'critic_gradient': critic_gradient
        }
        return total_loss, loss_info_dict

    def _calculate_advs(self, batch, rewards, terminated, actions, mask, bs, max_t, agent):
        mac_out, q_outs = agent.model.rl_forward(batch)

        # Mask out unavailable actions, renormalise (as in action selection)
        # mac_out[avail_actions == 0] = 0
        # mac_out = mac_out/(mac_out.sum(dim=-1, keepdim=True) + 1e-5)

        # Calculated baseline
        pi = mac_out[:, :-1]  #[bs, t, n_agents, n_actions]
        pi_taken = torch.gather(pi, dim=-1, index=actions[:, :-1]).squeeze(-1)    #[bs, t, n_agents]
        action_mask = mask.repeat(1, 1, (agent.player_num * agent.team_num))
        pi_taken[action_mask == 0] = 1.0
        log_pi_taken = torch.log(pi_taken).reshape(-1)

        # Calculate entropy
        entropy = Categorical(pi).entropy().reshape(-1)  #[bs, t, n_agents, 1]

        # Calculate q targets
        targets_taken = q_outs.squeeze(-1)   #[bs, t, n_agents]
        targets_taken = agent.model.compute_value(targets_taken, batch) #[bs, t, 1]

        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, targets_taken, (agent.player_num * agent.team_num), self.gamma, self.td_lambda)

        advantages = targets - targets_taken[:, :-1]
        advantages = advantages.unsqueeze(2).reshape(-1)

        td_error = targets_taken[:, :-1] - targets.detach()
        td_error = td_error.unsqueeze(2).reshape(-1)


        return advantages, td_error, targets_taken[:, :-1].unsqueeze(2).repeat(1, 1, (agent.player_num * agent.team_num), 1).reshape(-1), log_pi_taken, entropy


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape[:-1], n_agents)
    ret[:, -1] = target_qs[:, -1] * (1 - torch.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]