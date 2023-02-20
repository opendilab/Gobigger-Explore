import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate

from bigrl.core.torch_utils.collate_fn import default_collate_with_dim
from bigrl.core.torch_utils.data_helper import to_device
from bigrl.serial.policy.dqn.recurr_agent import BaseAgent

from bigrl.serial.policy.dqn.utils import polyak_update
from .env_status import EnvManagerStatus
from .features import Features


class Agent(BaseAgent):
    def __init__(self, cfg=None, ):
        super(Agent, self).__init__(cfg)
        self.player_num = self.whole_cfg.env.player_num_per_team
        self.team_num = self.whole_cfg.env.team_num
        self.features = Features(self.whole_cfg)
        self.colllect_env_num = self.whole_cfg.collect.env_num
        self.evaluate_env_num = self.whole_cfg.evaluate.env_num
        self.direction_num = self.whole_cfg.agent.features.get('direction_num', 12)
        self.collect_env_status = EnvManagerStatus(self.whole_cfg, self.colllect_env_num)
        self.evaluate_env_status = EnvManagerStatus(self.whole_cfg, self.evaluate_env_num)

    def _preprocess_obs(self, obs_list, env_status=None):
        '''
        Args:
            obs:
                original obs
        Returns:
            model input: Dict of logits, hidden states, action_log_probs, action_info
            value_feature[Optional]: Dict of global info
        '''
        env_player_obs = defaultdict(dict)
        for env_id, env_obs in enumerate(obs_list):
            for game_player_id in range(self.player_num * self.team_num):
                if env_status is None:
                    last_action_type = self.direction_num * 2
                else:
                    last_action_type = env_status[env_id].last_action_types[game_player_id]
                game_player_obs = self.features.transform_obs(env_obs, game_player_id=game_player_id, padding=True,
                                                              last_action_type=last_action_type)
                env_player_obs[env_id][game_player_id] = game_player_obs

        return env_player_obs

    def collate_obs(self, env_player_obs):
        processed_obs_list = []
        for env_id, env_obs in env_player_obs.items():
            for game_player_id, game_player_obs in env_obs.items():
                processed_obs_list.append(game_player_obs)
        obs_batch = default_collate_with_dim(processed_obs_list, device=self.device)
        return obs_batch

    def preprocess_obs(self, obs_list, env_status=None):
        env_player_obs = self._preprocess_obs(obs_list, env_status)
        obs_batch = self.collate_obs(env_player_obs)
        return obs_batch

    @torch.no_grad()
    def collect_episodes(self, env_manager, n_episodes):
        self.model.eval()
        self.evaluate_env_status.reset()
        env_num = len(env_manager)
        env_episode_count = [0 for _ in range(env_num)]
        collect_episode = 0
        cumulative_rewards = []
        info_dict = defaultdict(list)

        obs_dict = env_manager.reset()
        obs_list = [obs_dict[i] for i in range(env_num)]
        while np.min(env_episode_count) < n_episodes:
            obs_row = self.preprocess_obs(obs_list, self.evaluate_env_status)
            model_output = self.model.forward(obs_row, )
            actions, buffer_actions = self.transform_action(model_output, self.evaluate_env_status, deterministic=True)
            next_obs, env_rewards, env_dones, env_infos = env_manager.step(actions)
            # self.evaluate_env_status.update(next_obs, env_rewards, env_dones, env_infos)
            for env_id in range(env_num):
                if env_dones[env_id]:
                    env_info = env_infos[env_id]
                    collect_episode += 1
                    leaderboard = obs_list[env_id][0]['leaderboard']
                    if env_episode_count[env_id] < n_episodes:
                        cumulative_rewards.extend(leaderboard.values())
                        for game_player_id in range(self.player_num * self.team_num):
                            for k, v in env_info['eats'][game_player_id].items():
                                info_dict[k].append(v)
                    env_episode_count[env_id] += 1
                    obs_list[env_id] = env_manager.reset(env_id)[env_id]
                    if self.print_eval_result:
                        print(f"Eval Env{env_id} finish its episode, with leaderboard {leaderboard}, info: {env_info}")
                    # self.evaluate_env_status.reset(env_id)
                else:
                    obs_list[env_id] = next_obs[env_id]
        mean_reward = np.mean(cumulative_rewards)

        eval_info = {'rew_mean': np.mean(mean_reward),
                     'rew_min': np.min(cumulative_rewards),
                     'rew_max': np.max(cumulative_rewards),
                     'rew_std': np.std(cumulative_rewards), }
        for k, val in info_dict.items():
            eval_info[k] = np.mean(val)
        eval_text = '\n' + "=" * 4 + f'Evaluation_iter{self.last_iter.val}' + "=" * 4 + '\n'
        headers = ['Name', 'Value']
        table_data = [['num', n_episodes * env_num]]
        for key, val in eval_info.items():
            table_data.append([key, f'{val:.3f}'])
        table_text = tabulate(table_data, headers=headers, tablefmt='grid',
                              stralign='left', numalign='left')

        eval_text += table_text

        return mean_reward, eval_info, eval_text

    def transform_action(self, agent_outputs, env_status, deterministic):
        env_num = len(env_status)
        buffer_actions = defaultdict(list)
        if not deterministic and np.random.rand() < self.exploration_rate:
            actions = {}
            for env_id in range(env_num):
                actions[env_id] = {}
                for game_player_id in range(self.player_num * self.team_num):
                    action_idx = self.features.get_random_action()
                    buffer_actions[env_id].append(action_idx)
                    env_status[env_id].last_action_types[game_player_id] = action_idx
                    actions[env_id][game_player_id] = self.features.transform_action(action_idx)
        else:
            actions_list = agent_outputs['action'].cpu().numpy().tolist()
            actions = {}
            for env_id in range(env_num):
                actions[env_id] = {}
                for game_player_id in range(self.player_num * self.team_num):
                    action_idx = actions_list[env_id * (self.player_num * self.team_num) + game_player_id]
                    buffer_actions[env_id].append(action_idx)
                    env_status[env_id].last_action_types[game_player_id] = action_idx
                    actions[env_id][game_player_id] = self.features.transform_action(action_idx)
        return actions, buffer_actions

    @torch.no_grad()
    def collect_data(self, env_manager, ):
        if self.last_obs_list is None:
            # this means we haven't collect any train_data
            reset_obs_dict = env_manager.reset()
            last_obs_list = [reset_obs_dict[idx] for idx in range(env_manager.env_num)]
            self.last_obs_list = self._preprocess_obs(last_obs_list)
        self.model.eval()

        start_time = time.time()
        cumulative_rewards = []
        info_dict = defaultdict(list)
        collect_episode = 0
        env_num = len(env_manager)
        collect_buffer = {(env_id, game_player_id): {'obs': [],
                                                     'action': [],
                                                     'reward': [],
                                                     'done': [],
                                                     }
                          for env_id in range(env_num) for game_player_id in range(self.player_num * self.team_num)}
        next_obs_list = [None for _ in range(env_num)]
        for i in range(self.rollout_nstep):
            curr_obs_row = self.collate_obs(self.last_obs_list)
            model_outputs = self.model.forward(curr_obs_row, )
            actions, buffer_actions = self.transform_action(model_outputs, self.collect_env_status, deterministic=False)
            next_obs, env_rewards, env_dones, env_infos = env_manager.step(actions)
            env_rewards_list, env_done_list = self.collect_env_status.update(next_obs, env_rewards, env_dones,
                                                                             env_infos)
            for env_id in range(env_num):
                # if done, put result to result queue
                if env_dones[env_id]:
                    env_info = env_infos[env_id]
                    next_obs_list[env_id] = env_manager.reset(env_id)[env_id]
                    leaderboard = next_obs[env_id][0]['leaderboard']
                    cumulative_rewards.extend(leaderboard.values())
                    for game_player_id in range(self.player_num * self.team_num):
                        for k, v in env_info['eats'][game_player_id].items():
                            info_dict[k].append(v)
                    collect_episode += 1
                    if self.print_collect_result:
                        print(
                            f"Collect Env{env_id} finish its episode, with leaderboard: {leaderboard}, info: {env_info}")
                        self.collect_env_status.reset(env_id)
                else:
                    next_obs_list[env_id] = next_obs[env_id]

            self.last_obs_list = self._preprocess_obs(next_obs_list)
            for env_id in range(env_num):
                for game_player_id in range(self.player_num * self.team_num):
                    collect_buffer[(env_id, game_player_id)]['obs'].append(self.last_obs_list[env_id][game_player_id])
                    collect_buffer[(env_id, game_player_id)]['action'].append(buffer_actions[env_id][game_player_id])
                    collect_buffer[(env_id, game_player_id)]['reward'].append(env_rewards_list[env_id][game_player_id])
                    collect_buffer[(env_id, game_player_id)]['done'].append(env_done_list[env_id][game_player_id])

            self._n_calls += 1
            if self._n_calls % self.target_update_interval == 0:
                polyak_update(self.model.parameters(), self.target_model.parameters(), self.tau)
        collect_data = []
        for env_id in range(env_num):
            for game_player_id in range(self.player_num * self.team_num):
                collect_buffer[(env_id, game_player_id)]['obs'].append(self.last_obs_list[env_id][game_player_id])
                collect_data.append({
                    'obs': default_collate_with_dim(collect_buffer[(env_id, game_player_id)]['obs']),
                    'action': default_collate_with_dim(collect_buffer[(env_id, game_player_id)]['action']),
                    'reward': default_collate_with_dim(collect_buffer[(env_id, game_player_id)]['reward']).float(),
                    'done': default_collate_with_dim(collect_buffer[(env_id, game_player_id)]['done']),
                })
        collect_timestep = env_num * self.rollout_nstep
        collect_time_cost = time.time() - start_time

        self.total_collect_timesteps += collect_timestep
        self.total_collect_episode += collect_episode
        self._current_progress_remaining = max(1 - self.total_collect_timesteps / self.total_timesteps, 0)
        if self.exploration_type == 'frame':
            self.exploration_rate = self.exploration_schedule(self.total_collect_timesteps)
        else:
            self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        collect_info = {'collect/time': collect_time_cost,
                        'collect/rewards': cumulative_rewards,
                        'collect/timestep': collect_timestep,
                        'collect/velocity': collect_timestep / collect_time_cost,
                        'collect/episode': collect_episode,
                        'collect/eps': self.exploration_rate,
                        }

        return collect_data, collect_info

    def update(self, train_data):
        train_data = to_device(train_data, self.device)
        with self.timer:
            unroll_len, batch_size = train_data['reward'].shape[0], train_data['reward'].shape[1]
            flatten_obs_plus_one = flatten_data(train_data['obs'],start_dim=0,end_dim=1)
            with torch.no_grad():
                # Compute the next Q-values using the target network
                next_target_model_outputs = self.target_model(flatten_obs_plus_one)
                next_q_values_plus_one = next_target_model_outputs['q_value'].reshape(unroll_len+1, batch_size, -1)
                next_q_values = next_q_values_plus_one[1:]
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=-1)
                # Avoid potential broadcast issue
                # next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = train_data['reward'] + (1 - train_data['done'].float()) * self.gamma * next_q_values
            # Get current Q-values estimates
            target_model_outputs = self.model(flatten_obs_plus_one)
            current_q_values_plus_one = target_model_outputs['q_value'].reshape(unroll_len+1, batch_size, -1)
            current_q_values = current_q_values_plus_one[:-1]
            # Retrieve the q-values for the actions from the replay buffer

            current_q_values = torch.gather(current_q_values, dim=2,
                                            index=train_data['action'].unsqueeze(-1).long()).squeeze(-1)
            # Avoid potential broadcast issue

            # Compute Huber loss (less sensitive to outliers)
            total_loss = F.huber_loss(current_q_values, target_q_values)

        train_forward = self.timer.value
        with self.timer:
            self.optimizer.zero_grad()
            total_loss.backward()
            gradient = self.grad_clip.apply(self.model.parameters())
            self.optimizer.step()
            if self.lr_scheduler_type == 'Progress':
                self.lr_scheduler.set_progress(self._current_progress_remaining)
            self.lr_scheduler.step()
        train_background = self.timer.value
        loss_info_dict = {'total_loss': total_loss.item(),
                          'train_forward': train_forward,
                          'train_backward': train_background,
                          'gradient': gradient,
                          'lr': self.lr_scheduler.get_last_lr()[0],
                          'current_q':current_q_values.mean().item(),
                          'target_q':target_q_values.mean().item(),
                          }
        return loss_info_dict


def flatten_data(data,start_dim=0,end_dim=1):
    if isinstance(data, dict):
        return {k: flatten_data(v,start_dim=start_dim, end_dim=end_dim) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return torch.flatten(data, start_dim=start_dim, end_dim=end_dim)
