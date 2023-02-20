import time
from collections import defaultdict

import numpy as np
import torch
from tabulate import tabulate

from bigrl.core.torch_utils.collate_fn import default_collate_with_dim
from bigrl.serial.policy.dqn.agent import BaseAgent

from bigrl.serial.policy.dqn.utils import polyak_update
from .env_status import EnvManagerStatus
from .features import Features
import os
from ..bot.agent import Agent as BotAgent
import copy


class Bot:
    def __init__(self, whole_cfg, env_num):
        self.whole_cfg = whole_cfg
        self.env_num = env_num
        self.player_num = self.whole_cfg.env.player_num_per_team
        self.team_num = self.whole_cfg.env.team_num
        self.bot = defaultdict(dict)
        for env_id in range(self.env_num):
            self.reset(env_id)

    def get_actions(self, raw_obs):
        actions = {}
        for env_id in range(self.env_num):
            actions[env_id] = {}
            for game_player_id, obs in raw_obs[env_id].items():
                action = self.bot[env_id][game_player_id].step(obs)[game_player_id]
                actions[env_id][game_player_id] = action
        return actions

    def reset(self, env_id):
        for env_id in range(self.env_num):
            for game_player_id in range(self.player_num * 1, self.player_num * self.team_num):
                cfg = copy.deepcopy(self.whole_cfg)
                cfg.agent.game_player_id = game_player_id
                cfg.agent.game_team_id = game_player_id // self.player_num
                cfg.agent.player_id = game_player_id # meaningless in serial pipeline
                cfg.agent.send_data = False          # meaningless in serial pipeline
                self.bot[env_id][game_player_id] = BotAgent(cfg)
        
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
        
        self.use_action_mask = self.whole_cfg.agent.get('use_action_mask', False)
        
        # only use for eval
        self.game_player_id = self.whole_cfg.agent.get('game_player_id', 0)
        self.game_team_id = self.whole_cfg.agent.get('game_team_id', 0)

    def _preprocess_obs(self, obs_list, env_status=None, eval_vsbot=False):
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
            game_player_num = self.player_num if eval_vsbot else self.player_num * self.team_num
            for game_player_id in range(game_player_num):
                if env_status is None:
                    last_action_type = self.direction_num * 2
                else:
                    last_action_type = env_status[env_id].last_action_types[game_player_id]
                if self.use_action_mask:
                    can_eject = env_obs[1][game_player_id]['can_eject']
                    can_split = env_obs[1][game_player_id]['can_split']
                    action_mask = self.features.generate_action_mask(can_eject=can_eject,can_split=can_split)
                else:
                    action_mask = self.features.generate_action_mask(can_eject=True,can_split=True)
                game_player_obs = self.features.transform_obs(env_obs, game_player_id=game_player_id, padding=True,
                                                              last_action_type=last_action_type)
                game_player_obs['action_mask'] = action_mask
                env_player_obs[env_id][game_player_id] = game_player_obs
        return env_player_obs

    def collate_obs(self, env_player_obs):
        processed_obs_list = []
        for env_id, env_obs in env_player_obs.items():
            for game_player_id, game_player_obs in env_obs.items():
                processed_obs_list.append(game_player_obs)
        obs_batch = default_collate_with_dim(processed_obs_list, device=self.device)
        return obs_batch

    def preprocess_obs(self, obs_list, env_status=None, eval_vsbot=False):
        env_player_obs = self._preprocess_obs(obs_list, env_status, eval_vsbot)
        obs_batch = self.collate_obs(env_player_obs)
        return obs_batch
    
    def preprocess_bot_obs(self, obs_list):
        # only team_0 is rl
        env_player_obs = defaultdict(dict)
        for env_id, env_obs in enumerate(obs_list):
            for game_player_id in range(self.player_num*1, self.player_num*self.team_num):
                env_player_obs[env_id][game_player_id] = env_obs
        return env_player_obs

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
                        print(f"Eval selfplay Env{env_id} finish its episode, with leaderboard {leaderboard}, info: {env_info}")
                    self.evaluate_env_status.reset(env_id)
                else:
                    obs_list[env_id] = next_obs[env_id]
        mean_reward = np.mean(cumulative_rewards)

        eval_info = {'rew_mean': np.mean(mean_reward),
                     'rew_min': np.min(cumulative_rewards),
                     'rew_max': np.max(cumulative_rewards),
                     'rew_std': np.std(cumulative_rewards), }
        for k, val in info_dict.items():
            eval_info[k] = np.mean(val)
        eval_text = '\n' + "=" * 4 + f'Evaluation_selfplay_iter{self.last_iter.val}_envstep{self.total_collect_timesteps}' + "=" * 4 + '\n'
        headers = ['Name', 'Value']
        table_data = [['num', n_episodes * env_num]]
        for key, val in eval_info.items():
            table_data.append([key, f'{val:.3f}'])
        table_text = tabulate(table_data, headers=headers, tablefmt='grid',
                              stralign='left', numalign='left')

        eval_text += table_text

        return mean_reward, eval_info, eval_text

    def transform_action(self, agent_outputs, env_status, deterministic, eval_vsbot=False):
        env_num = len(env_status)
        buffer_actions = defaultdict(list)
        if not deterministic and np.random.rand() < self.exploration_rate:
            actions = {}
            for env_id in range(env_num):
                actions[env_id] = {}
                game_player_num = self.player_num if eval_vsbot else self.player_num * self.team_num
                for game_player_id in range(game_player_num):
                    action_idx = self.features.get_random_action()
                    buffer_actions[env_id].append(action_idx)
                    env_status[env_id].last_action_types[game_player_id] = action_idx
                    actions[env_id][game_player_id] = self.features.transform_action(action_idx)
        else:
            actions_list = agent_outputs['action'].cpu().numpy().tolist()
            actions = {}
            for env_id in range(env_num):
                actions[env_id] = {}
                game_player_num = self.player_num if eval_vsbot else self.player_num * self.team_num
                for game_player_id in range(game_player_num):
                    action_idx = actions_list[env_id * (game_player_num) + game_player_id]
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

            self._n_calls += 1
            if self._n_calls % self.target_update_interval == 0:
                polyak_update(self.model.parameters(), self.target_model.parameters(), self.tau)

            next_preprocessed_obs_list = self._preprocess_obs(next_obs_list)

            for env_id in range(env_num):
                for game_player_id in range(self.player_num * self.team_num):
                    env_step_data = {
                        'obs': self.last_obs_list[env_id][game_player_id],
                        'action': buffer_actions[env_id][game_player_id],
                        'next_obs': next_preprocessed_obs_list[env_id][game_player_id],
                        'reward': env_rewards_list[env_id][game_player_id],
                        'done': env_done_list[env_id][game_player_id],
                    }
                    self.replay_buffer.push_data(env_step_data)

            self.last_obs_list = next_preprocessed_obs_list

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

        return {}, collect_info
    
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
    
    @torch.no_grad()
    def collect_episodes_vsbot(self, env_manager, n_episodes):
        self.model.eval()
        self.evaluate_env_status.reset()
        env_num = len(env_manager)
        self.bot = Bot(self.whole_cfg, env_num)
        env_episode_count = [0 for _ in range(env_num)]
        collect_episode = 0
        cumulative_rewards = []
        info_dict = defaultdict(list)

        obs_dict = env_manager.reset()
        obs_list = [obs_dict[i] for i in range(env_num)]
        while np.min(env_episode_count) < n_episodes:
            obs_row = self.preprocess_obs(obs_list, self.evaluate_env_status, eval_vsbot=True)
            model_output = self.model.forward(obs_row, )
            actions, buffer_actions = self.transform_action(model_output, self.evaluate_env_status, deterministic=True, eval_vsbot=True)
            # bot forward
            bot_obs_row = self.preprocess_bot_obs(obs_list)
            bot_actions = self.bot.get_actions(bot_obs_row)
            for env_id, v in bot_actions.items():
                actions[env_id].update(v)
            next_obs, env_rewards, env_dones, env_infos = env_manager.step(actions)
            # self.evaluate_env_status.update(next_obs, env_rewards, env_dones, env_infos)
            for env_id in range(env_num):
                if env_dones[env_id]:
                    env_info = env_infos[env_id]
                    collect_episode += 1
                    leaderboard = obs_list[env_id][0]['leaderboard']
                    if env_episode_count[env_id] < n_episodes:
                        cumulative_rewards.append(leaderboard[0])
                        for game_player_id in range(self.player_num):
                            for k, v in env_info['eats'][game_player_id].items():
                                info_dict[k].append(v)
                    env_episode_count[env_id] += 1
                    obs_list[env_id] = env_manager.reset(env_id)[env_id]
                    self.bot.reset(env_id)
                    if self.print_eval_result:
                        print(f"Eval vsbot Env{env_id} finish its episode, with leaderboard {leaderboard}, info: {env_info}")
                    self.evaluate_env_status.reset(env_id)
                else:
                    obs_list[env_id] = next_obs[env_id]
        mean_reward = np.mean(cumulative_rewards)

        eval_info = {'vsbot_rew_mean': np.mean(mean_reward),
                     'vsbot_rew_min': np.min(cumulative_rewards),
                     'vsbot_rew_max': np.max(cumulative_rewards),
                     'vsbot_rew_std': np.std(cumulative_rewards), }
        for k, val in info_dict.items():
            eval_info[k] = np.mean(val)
        eval_text = '\n' + "=" * 4 + f'Evaluation_vsbot_iter{self.last_iter.val}_envstep{self.total_collect_timesteps}' + "=" * 4 + '\n'
        headers = ['Name', 'Value']
        table_data = [['num', n_episodes * env_num]]
        for key, val in eval_info.items():
            table_data.append([key, f'{val:.3f}'])
        table_text = tabulate(table_data, headers=headers, tablefmt='grid',
                              stralign='left', numalign='left')

        eval_text += table_text
        return mean_reward, eval_info, eval_text

    ########## only for submission####################
    def reset(self):
        self.last_action_type = self.features.direction_num * 2

    def gen_action_mask(self, obs):
        if self.use_action_mask:
            can_eject = obs[1][self.game_player_id]['can_eject']
            can_split = obs[1][self.game_player_id]['can_split']
            action_mask = self.features.generate_action_mask(can_eject=can_eject,can_split=can_split)
        else:
            action_mask = self.features.generate_action_mask(can_eject=True,can_split=True)
        return action_mask
        
    def step(self, obs):
        """
        Overview:
            Agent.step() in submission
        Arguments:
            - obs
        Returns:
            - action
        """
        # action_mask
        action_mask = self.gen_action_mask(obs)
        
        # preprocess obs
        obs = self.features.transform_obs(obs, game_player_id=self.game_player_id,
                                          last_action_type=self.last_action_type)
        obs = default_collate_with_dim([obs])
        obs['action_mask'] = action_mask.unsqueeze(0)
        
        # policy 
        with torch.no_grad():
            model_output = self.model(obs)['action'].detach().numpy()
        
        # return action
        actions = {}
        actions[self.game_player_id] = self.features.transform_action(model_output[0])
        self.last_action_type = model_output[0].item()
        return actions
    ###################################################