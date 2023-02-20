import os
import time
from collections import defaultdict

import numpy as np
import torch
from tabulate import tabulate
from functools import partial
from bigrl.core.torch_utils.data_helper import to_device
from bigrl.core.rl_utils.td_lambda import generalized_lambda_returns
from bigrl.core.torch_utils.collate_fn import default_collate_with_dim
from bigrl.serial.policy.coma.agent import BaseAgent
from bigrl.core.data.episode_buffer import EpisodeBatch, ReplayBuffer
from .env_status import EnvManagerStatus
from .features import Features


class Agent(BaseAgent):
    def __init__(self, cfg=None, ):
        super(Agent, self).__init__(cfg)
        self.player_num = self.whole_cfg.env.player_num_per_team
        self.team_num = self.whole_cfg.env.team_num
        self.features = Features(self.whole_cfg)
        self.collect_env_num = self.whole_cfg.collect.env_num
        self.evaluate_env_num = self.whole_cfg.evaluate.env_num
        self.direction_num = self.whole_cfg.agent.features.get('direction_num', 12)
        self.collect_env_status = EnvManagerStatus(self.whole_cfg, self.collect_env_num)
        self.evaluate_env_status = EnvManagerStatus(self.whole_cfg, self.evaluate_env_num)
        self.envs_not_terminated = []
        self.last_obs_dict = {}
        self.action_num = self.cfg.action_num
        self.buffer = ReplayBuffer(self.cfg.replay_buffer.max_buffer_size, self.player_num * self.team_num, self.rollout_nstep + 1, features=self.features)

    def preprocess_obs(self, obs_list, env_status=None, _eval=False):
        '''
        Args:
            obs:
                original obs
        Returns:
            model input: Dict of logits, hidden states, action_log_probs, action_info
            value_feature[Optional]: Dict of global info
        '''
        try:
            processed_obs_list = []
            it = zip(self.envs_not_terminated, obs_list) if not _eval else enumerate(obs_list)
            for env_id, env_obs in it:
                for game_player_id in range(self.player_num * self.team_num):
                    if env_status is None:
                        last_action_type = self.direction_num * 2
                    else:
                        last_action_type = env_status[env_id].last_action_types[game_player_id]
                    env_player_obs = self.features.transform_obs(env_obs, game_player_id=game_player_id, padding=True,
                                                                last_action_type=last_action_type)
                    processed_obs_list.append(env_player_obs)
            obs_batch = default_collate_with_dim(processed_obs_list, device=self.device)
            return obs_batch
        except Exception as e:
            print(obs_list, self.envs_not_terminated)
            raise e

    def train(self, train_data):
        self.model.train()

        self.buffer.insert_episode_batch(train_data)
        if self.total_collect_timesteps > 0 and self.total_collect_timesteps > self.learning_starts:
            log_record_list = []
            for _ in range(self.update_per_collect):
                train_sample = self.buffer.sample(self.batch_size)
                self.update(train_sample)
            # self.loss.set_progress(self._current_progress_remaining)
            # Increase update counter
            self.n_updates += self.update_per_collect
            log_info_dict = self.update(train_sample)
            log_record_list.append(log_info_dict)
            self.last_iter.add(1)
            return log_record_list
        else:
            return None

    def reset(self, team_num):
        self.features.team_num = team_num
        self.evaluate_env_status.reset()

    def step(self, obs):
        obs = self.preprocess_obs([obs], self.evaluate_env_status, _eval=True)
        with torch.no_grad():
            model_output = self.model(obs, temperature=1.)
        actions = self.transform_action(model_output, self.evaluate_env_status, _eval=True)
        return actions[0]

    def update(self, train_data):
        # with self.timer:
        #     with torch.enable_grad():
        #         model_output = self.model.rl_train(train_data['obs'])
        train_data.to(self.device)
        train_forward = self.timer.value
        with self.timer:
            total_loss, loss_info_dict = self.loss.compute_loss(train_data, self)
            # self.optimizer.zero_grad()
            # total_loss.backward()
            # gradient = self.grad_clip.apply(self.model.parameters())
            # self.optimizer.step()
            if self.lr_scheduler_type == 'Progress':
                self.lr_scheduler.set_progress(self._current_progress_remaining)
            self.lr_scheduler.step()
        train_background = self.timer.value
        loss_info_dict.update({'train_forward': train_forward,
                               'train_backward': train_background,
                               'lr': self.lr_scheduler.get_last_lr()[0]
                               })
        return loss_info_dict

    def _update_targets(self):
        self.target_model.load_state_dict(self.model.state_dict())

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
            obs_row = self.preprocess_obs(obs_list, self.evaluate_env_status, _eval=True)
            model_output = self.model(obs_row, temperature=1.)
            actions = self.transform_action(model_output, self.evaluate_env_status, _eval=True)
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

    def transform_action(self, agent_outputs, env_status, _eval=False):
        env_num = len(env_status)
        actions_list = agent_outputs['action'].cpu().numpy().tolist()
        actions = {}
        it = self.envs_not_terminated if not _eval else range(env_num)
        for env_id in it:
            actions[env_id] = {}
            for game_player_id in range(self.player_num * self.team_num):
                action_idx = actions_list[env_id * (self.player_num * self.team_num) + game_player_id]
                env_status[env_id].last_action_types[game_player_id] = action_idx
                actions[env_id][game_player_id] = self.features.transform_action(action_idx)
        return actions

    @torch.no_grad()
    def collect_data(self, env_manager):
        self.model.eval()

        env_num = len(env_manager)
        start_time = time.time()
        cumulative_rewards = []
        next_obs_dict = self.last_obs_dict
        info_dict = defaultdict(list)
        collect_episode = 0
        collect_timestep = 0

        for env_id in range(env_num):
            if env_id not in self.envs_not_terminated:
                next_obs_dict[env_id] = env_manager.reset(env_id)[env_id]
                self.envs_not_terminated.insert(env_id, env_id)
                self.collect_env_status.reset(env_id)
        curr_obs_row = self.preprocess_obs([next_obs_dict[k] for k in self.envs_not_terminated], self.collect_env_status)

        batch = EpisodeBatch(env_num, self.player_num * self.team_num, self.rollout_nstep + 1, features=self.features)
        for i in range(self.rollout_nstep):
            batch.update({'obs': curr_obs_row}, bs=self.envs_not_terminated, ts=i, mark_filled=True)
            agent_outputs = self.model(curr_obs_row, temperature=1.)
            actions = self.transform_action(agent_outputs, self.collect_env_status)
            next_obs, env_rewards, env_dones, env_infos = env_manager.step(actions, id=self.envs_not_terminated)
            step_reward_list, step_done_list = self.collect_env_status.update(next_obs, env_rewards, env_dones,
                                                                              env_infos, self.envs_not_terminated)
            step_rewards = torch.as_tensor(step_reward_list, device=self.device)

            _terminated = []
            for env_id in self.envs_not_terminated:
                # if done, put result to result queue
                if env_dones[env_id]:
                    env_info = env_infos[env_id]
                    leaderboard = next_obs[env_id][0]['leaderboard']
                    cumulative_rewards.extend(leaderboard.values())
                    for game_player_id in range(self.player_num * self.team_num):
                        for k, v in env_info['eats'][game_player_id].items():
                            info_dict[k].append(v)
                    collect_episode += 1
                    if self.print_collect_result:
                        print(
                            f"Collect Env{env_id} finish its episode, with leaderboard: {leaderboard}, info: {env_info}")
                    _terminated.append(env_id)
                else:
                    next_obs_dict[env_id] = next_obs[env_id]
            collect_timestep += len(self.envs_not_terminated)
            
            curr_obs_row = self.preprocess_obs([next_obs_dict[k] for k in self.envs_not_terminated], self.collect_env_status)
            batch.update({'action': agent_outputs['action'], 'reward': step_rewards, 
                            'terminated': [True if _ in _terminated else False for _ in self.envs_not_terminated]}, 
                            bs=self.envs_not_terminated, ts=i, mark_filled=False)
            for id in _terminated:
                self.envs_not_terminated.remove(id)
            if len(self.envs_not_terminated) == 0:
                break

        self.last_obs_dict = next_obs_dict
        collect_time_cost = time.time() - start_time
        collect_info = {'collect/time': collect_time_cost,
                        'collect/rewards': cumulative_rewards,
                        'collect/timestep': collect_timestep,
                        'collect/velocity': collect_timestep / collect_time_cost,
                        'collect/episode': collect_episode,
                        }
        collect_info.update(info_dict)
        self.total_collect_timesteps += collect_timestep
        self.total_collect_episode += collect_episode
        self._current_progress_remaining = max(1 - self.total_collect_timesteps / self.total_timesteps, 0)
        return batch, collect_info

    def save_checkpoint(self, checkpoint_dir='.'):
        checkpoint_path = os.path.join(checkpoint_dir,
                                       '{}_iteration_{}_envstep_{}.pth.tar'.format(self.whole_cfg.common.experiment_name,
                                                                        self.last_iter.val, self.total_collect_timesteps))
        self.checkpoint_helper.save(checkpoint_path,
                                    model=self.model,
                                    optimizer=self.optimizer,
                                    last_iter=self.last_iter,
                                    )
        return checkpoint_path
