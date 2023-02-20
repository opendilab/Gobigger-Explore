import time
from collections import defaultdict

import numpy as np
import torch
from tabulate import tabulate

from bigrl.core.rl_utils.td_lambda import generalized_lambda_returns
from bigrl.core.torch_utils.collate_fn import default_collate_with_dim
from bigrl.serial.policy.ppo.agent import BaseAgent
from ..bot.agent import Agent as BotAgent
from .env_status import EnvManagerStatus
from .features import Features
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

    def preprocess_obs(self, obs_list, env_status=None):
        '''
        Args:
            obs:
                original obs
        Returns:
            model input: Dict of logits, hidden states, action_log_probs, action_info
            value_feature[Optional]: Dict of global info
        '''
        processed_obs_list = []
        for env_id, env_obs in enumerate(obs_list):
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
            actions = self.transform_action(model_output, self.evaluate_env_status)
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
                    self.evaluate_env_status.reset(env_id)
                    if self.print_eval_result:
                        print(f"Eval selfplay Env{env_id} finish its episode, with leaderboard {leaderboard}, info: {env_info}")
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

    def transform_action(self, agent_outputs, env_status):
        env_num = len(env_status)
        actions_list = agent_outputs['action'].cpu().numpy().tolist()
        actions = {}
        for env_id in range(env_num):
            actions[env_id] = {}
            for game_player_id in range(self.player_num * self.team_num):
                action_idx = actions_list[env_id * (self.player_num * self.team_num) + game_player_id]
                env_status[env_id].last_action_types[game_player_id] = action_idx
                actions[env_id][game_player_id] = self.features.transform_action(action_idx)
        return actions

    @torch.no_grad()
    def collect_data(self, env_manager,):
        if self.last_obs_row is None:
            # this means we haven't collect any train_data
            reset_obs_dict = env_manager.reset()
            obs_list  = [reset_obs_dict[idx] for idx in range(env_manager.env_num)]
            self.last_obs_row = self.preprocess_obs(obs_list,self.collect_env_status)

        self.model.eval()

        start_time = time.time()
        cumulative_rewards = []
        info_dict = defaultdict(list)
        collect_episode = 0

        env_num = len(env_manager)
        obs_list = []
        action_list = []
        reward_list = []
        done_list = []
        value_list = []
        logp_list = []
        next_obs_list = [None for _ in range(env_num)]
        for i in range(self.rollout_nstep):
            curr_obs_row = self.last_obs_row
            agent_outputs = self.model.compute_logp_action(curr_obs_row, )
            actions = self.transform_action(agent_outputs, self.collect_env_status)
            next_obs, env_rewards, env_dones, env_infos = env_manager.step(actions)
            step_reward_list, step_done_list = self.collect_env_status.update(next_obs, env_rewards, env_dones,
                                                                              env_infos)
            step_rewards = torch.as_tensor(step_reward_list, device=self.device)
            step_dones = torch.as_tensor(step_done_list, device=self.device)
            obs_list.append(curr_obs_row)
            logp_list.append(agent_outputs['action_logp'])
            action_list.append(agent_outputs['action'])
            value_list.append(agent_outputs['value'])
            reward_list.append(step_rewards)
            done_list.append(step_dones)

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
            self.last_obs_row = self.preprocess_obs(next_obs_list,self.collect_env_status )

        bootstrap_value = self.model.compute_value(curr_obs_row)['value']
        state_values = default_collate_with_dim(value_list, device=self.device)
        old_values = state_values

        if self.loss.reward_normalization:
            state_values *= np.sqrt(self.loss.ret_rms.var + self.loss._eps)
            bootstrap_value *= np.sqrt(self.loss.ret_rms.var + self.loss._eps)

        done_tensor = torch.stack(done_list, dim=0)  # t, b
        reward_tensor = torch.stack(reward_list, dim=0).float()  # t, b
        discounts = (1 - done_tensor.float()) * self.loss.gamma  # t, b
        unnormalized_returns = generalized_lambda_returns(rewards=reward_tensor,
                                                          pcontinues=discounts,
                                                          state_values=state_values,
                                                          bootstrap_value=bootstrap_value,
                                                          lambda_=self.loss.gae_lambda)

        advantages = unnormalized_returns - state_values
        if self.loss.reward_normalization:
            returns = unnormalized_returns / np.sqrt(self.loss.ret_rms.var + self.loss._eps)
            self.loss.ret_rms.update(unnormalized_returns.cpu().numpy())
        else:
            returns = unnormalized_returns
        train_data = {}
        train_data['obs'] = default_collate_with_dim(obs_list, device=self.device)  # t, b, *
        train_data['action'] = default_collate_with_dim(action_list, device=self.device)  # t, b
        train_data['action_logp'] = default_collate_with_dim(logp_list, device=self.device)  # t, b
        train_data['return'] = returns
        train_data['advantage'] = advantages
        train_data['old_value'] = old_values

        collect_timestep = env_num * self.rollout_nstep
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
        return train_data, collect_info

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
            obs_row = self.preprocess_obs(obs_list, self.evaluate_env_status)
            model_output = self.model.forward(obs_row, )
            actions = self.transform_action(model_output, self.evaluate_env_status)
            # bot reward
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
                    self.evaluate_env_status.reset(env_id)
                    if self.print_eval_result:
                        print(f"Eval vsbot Env{env_id} finish its episode, with leaderboard {leaderboard}, info: {env_info}")
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