import copy
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate

from bigrl.core.torch_utils.collate_fn import default_collate_with_dim
from bigrl.core.torch_utils.data_helper import to_device
from bigrl.serial.policy.dqn.agent import BaseAgent
from bigrl.serial.policy.dqn.utils import polyak_update

from ..bot.agent import Agent as BotAgent
from .env_status import EnvManagerStatus
from .features import Features


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
        self.game_player_id = self.whole_cfg.agent.get('game_team_id',0)
        self.game_team_id = self.game_player_id // self.player_num

    def _preprocess_obs(self, obs_list, env_status=None, eval_vsbot=False):
        '''
        Args:
            obs:
                original obs
        Returns:
            model input: Dict of logits, hidden states, action_log_probs, action_info
            value_feature[Optional]: Dict of global info
        '''
        env_team_obs = defaultdict(dict)
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
                env_team_obs[env_id][game_player_id//self.player_num] = env_team_obs[env_id].get(game_player_id//self.player_num,[])+[game_player_obs]
        return env_team_obs

    def collate_team_obs(self, env_team_obs):
        processed_obs_list = []
        for env_id, env_obs in env_team_obs.items():
            for game_team_id, game_team_obs in env_obs.items():
                game_team_obs = stack(game_team_obs)
                processed_obs_list.append(game_team_obs)
        obs_batch = default_collate_with_dim(processed_obs_list, device=self.device)
        return obs_batch

    def collate_obs(self, env_player_obs):
        processed_obs_list = []
        for env_id, env_obs in env_player_obs.items():
            for game_player_id, game_player_obs in env_obs.items():
                processed_obs_list.append(game_player_obs)
        obs_batch = default_collate_with_dim(processed_obs_list, device=self.device)
        return obs_batch

    def preprocess_obs(self, obs_list, env_status=None, eval_vsbot=False):
        env_player_obs = self._preprocess_obs(obs_list, env_status, eval_vsbot)
        obs_batch = self.collate_team_obs(env_player_obs)
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
            obs_row = flatten_data(obs_row,start_dim=0,end_dim=1)
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
        buffer_actions = defaultdict(dict)
        if not deterministic and np.random.rand() < self.exploration_rate:
            actions = {}
            for env_id in range(env_num):
                actions[env_id] = {}
                game_player_num = self.player_num if eval_vsbot else self.player_num * self.team_num
                for game_player_id in range(game_player_num):
                    action_idx = self.features.get_random_action()
                    buffer_actions[env_id][game_player_id//self.player_num] = buffer_actions[env_id].get(game_player_id//self.player_num, []) + [action_idx]
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
                    buffer_actions[env_id][game_player_id//self.player_num] = buffer_actions[env_id].get(game_player_id//self.player_num, []) + [action_idx]
                    env_status[env_id].last_action_types[game_player_id] = action_idx
                    actions[env_id][game_player_id] = self.features.transform_action(action_idx)
        game_team_num = 1 if eval_vsbot else self.team_num
        for env_id in range(env_num):
            for team_id in range(game_team_num):
                buffer_actions[env_id][team_id] = torch.tensor(buffer_actions[env_id][team_id])
        return actions, buffer_actions

    @torch.no_grad()
    def collect_data(self, env_manager, ):
        if self.last_obs_list is None:
            # this means we haven't collect any train_data
            reset_obs_dict = env_manager.reset()
            last_obs_list = [reset_obs_dict[idx] for idx in range(env_manager.env_num)]
            self.last_obs_list = self._preprocess_obs(last_obs_list)
            # len(last_obs_list) = env_num
            # len(last_obs_list[0]) = team_num
            # len(last_obs_list[0][0]) = player_num
        self.model.eval()

        start_time = time.time()
        cumulative_rewards = []
        info_dict = defaultdict(list)
        collect_episode = 0
        env_num = len(env_manager)
        next_obs_list = [None for _ in range(env_num)]
        for i in range(self.rollout_nstep):
            curr_obs_row = self.collate_team_obs(self.last_obs_list)
            curr_obs_row = flatten_data(curr_obs_row,start_dim=0,end_dim=1) # [env_num*team_num, 2]
            model_outputs = self.model.forward_collect(curr_obs_row)
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

            env_rewards_list = self.collate_env_data_list(env_rewards_list, env_num)
            env_done_list = self.collate_env_data_list(env_done_list, env_num)
            self._n_calls += 1
            if self._n_calls % self.target_update_interval == 0:
                polyak_update(self.model.parameters(), self.target_model.parameters(), self.tau)

            next_preprocessed_obs_list = self._preprocess_obs(next_obs_list)

            for env_id in range(env_num):
                for team_id in range(self.team_num):
                    env_step_data = {
                        'obs': stack(self.last_obs_list[env_id][team_id]),
                        'action': buffer_actions[env_id][team_id],
                        'next_obs': stack(next_preprocessed_obs_list[env_id][team_id]),
                        'reward': env_rewards_list[env_id][team_id],
                        'done': env_done_list[env_id][team_id],
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

    def collate_env_data_list(self, player_data, env_num):
        team_data = defaultdict(dict)
        for env_id in range(env_num):
            for player_id in range(len(player_data[0])):
                team_data[env_id][player_id//self.player_num] = team_data[env_id].get(player_id//self.player_num, []) + [player_data[env_id][player_id]]
        for env_id in range(env_num):
            for team_id in range(self.team_num):
                team_data[env_id][team_id] = torch.tensor(team_data[env_id][team_id][0])
        return team_data
    
    def update(self, train_data):
        train_data = to_device(train_data, self.device)
        train_data['obs'] = flatten_data(train_data['obs'],start_dim=0,end_dim=1)
        train_data['next_obs'] = flatten_data(train_data['next_obs'],start_dim=0,end_dim=1)
        train_data['action'] = flatten_data(train_data['action'],start_dim=0,end_dim=1)
        with self.timer:
            with torch.no_grad():
                # Compute the next Q-values using the target network

                next_obs = train_data['next_obs']
                next_target_model_outputs = self.target_model(next_obs)
                # Follow greedy policy: use the one with the highest value
                target_total_q = next_target_model_outputs['total_q']
                #target_total_q, _ = target_total_q.max(dim=-1)
                target_v = self.gamma * (1 - train_data['done'].float()) * target_total_q + train_data['reward']
        
            # Get current Q-values estimates
            curr_obs = train_data['obs']
            target_model_outputs = self.model(curr_obs)
            total_q = target_model_outputs['total_q']
            
            # Compute Huber loss (less sensitive to outliers)
            # total_loss = F.huber_loss(total_q, target_v)
            total_loss = torch.nn.MSELoss(reduction='none')(total_q, target_v).mean()

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
                          'current_total_q': total_q.mean().item() / self.player_num,
                          'target_reward_total_q': target_v.mean().item() / self.player_num,
                          'target_total_q': target_total_q.mean().item() / self.player_num,
                          }
        return loss_info_dict
    
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
            obs_row = flatten_data(obs_row,start_dim=0,end_dim=1)
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
        self.last_action_type = {}
        for player_id in range(self.player_num*self.game_team_id, self.player_num*(self.game_team_id+1)):
            self.last_action_type[player_id] = self.features.direction_num * 2
        
    def step(self, obs):
        """
        Overview:
            Agent.step() in submission
        Arguments:
            - obs
        Returns:
            - action
        """
        
        # preprocess obs
        env_team_obs = []
        for player_id in range(self.player_num*self.game_team_id, self.player_num*(self.game_team_id+1)):
            game_player_obs = self.features.transform_obs(obs, game_player_id=player_id,
                                            last_action_type=self.last_action_type[player_id])
            env_team_obs.append(game_player_obs)
        env_team_obs = stack(env_team_obs)
        obs = default_collate_with_dim([env_team_obs], device=self.device)
        
        # policy 
        self.model_input = flatten_data(obs,start_dim=0,end_dim=1)
        with torch.no_grad():
            model_output = self.model(self.model_input)['action'].cpu().detach().numpy()
        
        actions = []
        for i in range(len(model_output)):
            actions.append(self.features.transform_action(model_output[i]))
        ret = {}
        for player_id, act in zip(range(self.player_num*self.game_team_id, self.player_num*(self.game_team_id+1)), actions):
            ret[player_id] = act
        for player_id, act in zip(range(self.player_num*self.game_team_id, self.player_num*(self.game_team_id+1)), model_output):
            self.last_action_type[player_id] = act.item() # TODO
        return ret
    #####################################
                

def flatten_data(data,start_dim=0,end_dim=1):
    if isinstance(data, dict):
        return {k: flatten_data(v,start_dim=start_dim,end_dim=end_dim) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return torch.flatten(data, start_dim=start_dim, end_dim=end_dim)

def stack(data):
    result = {}
    for k1 in data[0].keys():
        result[k1] = {}
        if isinstance(data[0][k1], dict):
            for k2 in data[0][k1].keys():
                result[k1][k2] = torch.stack([o[k1][k2] for o in data])
        else:
            result[k1] = torch.stack([o[k1] for o in data])
    return result