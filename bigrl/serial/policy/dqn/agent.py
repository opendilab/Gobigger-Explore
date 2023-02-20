import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate

from bigrl.core.data.buffer import ReplayBuffer
from bigrl.core.torch_utils.checkpoint_helper import CountVar, CheckpointHelper
from bigrl.core.torch_utils.collate_fn import default_collate_with_dim
from bigrl.core.torch_utils.data_helper import to_device
from bigrl.core.torch_utils.grad_clip import build_grad_clip
from bigrl.core.torch_utils.lr_scheduler import build_lr_scheduler
from bigrl.core.torch_utils.optimizer import build_optimizer

from bigrl.core.utils.time_helper import EasyTimer
from bigrl.serial.import_helper import import_pipeline_module
from .utils import polyak_update


class BaseAgent:
    def __init__(self, cfg=None, ):
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.agent
        self.exp_dir = os.path.join(os.getcwd(), 'experiments', self.whole_cfg.common.experiment_name, )
        self.env_name = self.whole_cfg.env.name
        self.pipeline = self.whole_cfg.agent.pipeline

        self.use_cuda = self.cfg.get('use_cuda', True) and torch.cuda.is_available()
        self.device = torch.cuda.current_device() if self.use_cuda else 'cpu'
        self.timer = EasyTimer(self.use_cuda)
        self.reset_stat()
        self.print_collect_result = self.cfg.get('print_collect_result', False)
        self.print_eval_result = self.cfg.get('print_eval_result', False)
        self.total_timesteps = float(self.whole_cfg.learner.n_timesteps)
        self.learning_starts = self.cfg.get('learning_starts', 10)
        # setup model
        self.setup_model()
        # following is only used when train agent

        self.action_space = list(range(self.cfg.action_num))

        # loss ralated
        self.setup_optimizer()
        self.setup_lr_scheduler()
        self.replay_buffer = ReplayBuffer(self.cfg.replay_buffer)
        self.update_per_collect = self.cfg.update_per_collect

        # checkpoint related
        self.checkpoint_helper = CheckpointHelper()
        self.load_checkpoint_path = self.cfg.get('load_checkpoint_path', '')
        self.load_checkpoint()
        self.rollout_nstep = self.cfg.rollout_nstep
        self.setup_epsilon_exploration()

        # For updating the target network with multiple envs:
        self.use_double = self.cfg.get('use_double', False)
        self.target_update_interval = self.cfg.target_update_interval
        self.tau = self.cfg.tau
        self.gamma = self.cfg.loss_parameters.gamma
        self._n_calls = 0

    def setup_epsilon_exploration(self):
        self.exploration_initial_eps = self.cfg.eps_greedy.exploration_initial_eps
        self.exploration_final_eps = self.cfg.eps_greedy.exploration_final_eps
        self.exploration_type = self.cfg.eps_greedy.get('type', 'frame') # fram/frac
        if self.exploration_type == 'frame':
            from bigrl.core.torch_utils.scheduler import get_linear_decay_step_fn
            self.exploration_exploration_frames = self.cfg.eps_greedy.exploration_frames
            self.exploration_schedule = get_linear_decay_step_fn(
                start=self.exploration_initial_eps,
                end=self.exploration_final_eps,
                decay_steps=self.exploration_exploration_frames,
            )
        elif self.exploration_type == 'frac':
            from bigrl.core.torch_utils.scheduler import get_linear_fn
            self.exploration_fraction = self.cfg.eps_greedy.exploration_fraction
            self.exploration_schedule = get_linear_fn(
                self.exploration_initial_eps,
                self.exploration_final_eps,
                self.exploration_fraction,
            )
        else:
            print(f'dont support type {self.exploration_type} for eps exploration')
            raise NotImplementedError
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0

    def reset_collect_buffer(self):
        self.collect_env_num = self.whole_cfg.collect.env_num

    def reset_stat(self):
        self.last_iter = CountVar(init_val=0)
        self.total_collect_timesteps = 0
        self.total_collect_episode = 0
        self.last_obs_list = None
        self.n_updates = 0

    def setup_model(self):
        self.ModelClass = import_pipeline_module(self.env_name, self.pipeline, 'Model', )
        self.model = self.ModelClass(self.whole_cfg, ).to(device=self.device)
        self.target_model = self.ModelClass(self.whole_cfg, ).to(device=self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def preprocess_obs(self, obs_list):
        '''
        Args:
            obs:
                original obs
        Returns:
            model input: Dict of logits, hidden states, action_log_probs, action_info
            value_feature[Optional]: Dict of global info
        '''

        obs_batch = default_collate_with_dim(obs_list, device=self.device)
        return obs_batch

    def train(self, *args, **kwargs):
        # self.replay_buffer.push_data(train_data)

        log_record_list = []
        if self.total_collect_timesteps > 0 and self.total_collect_timesteps > self.learning_starts:
            self.model.train()
            for _ in range(self.update_per_collect):
                train_sample = self.replay_buffer.sample()
                if train_sample is None:
                    break
                else:
                    train_data = default_collate_with_dim(train_sample, dim=0)
                log_info_dict = self.update(train_data)
                log_record_list.append(log_info_dict)

            # self.loss.set_progress(self._current_progress_remaining)
            # Increase update counter
            self.n_updates += self.update_per_collect
            self.last_iter.add(1)
        return log_record_list

    def update(self, train_data):
        train_data = to_device(train_data, self.device)
        with self.timer:
            with torch.no_grad():
                # Compute the next Q-values using the target network

                next_obs = train_data['next_obs']
                next_target_model_outputs = self.target_model(next_obs)
                # Follow greedy policy: use the one with the highest value
                next_q_values = next_target_model_outputs['q_value']
                if self.use_double:
                    next_model_outputs = self.model(next_obs)
                    next_greedy_actions = next_model_outputs['action']
                    next_q_values = torch.gather(next_q_values, dim=-1, index=next_greedy_actions.unsqueeze(-1).long())
                else:
                    next_q_values, _ = next_q_values.max(dim=-1)
                # Avoid potential broadcast issue
                # next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = train_data['reward'] + (1 - train_data['done'].float()) * self.gamma * next_q_values
            # Get current Q-values estimates
            curr_obs = train_data['obs']
            target_model_outputs = self.model(curr_obs)
            current_q_values = target_model_outputs['q_value']
            # Retrieve the q-values for the actions from the replay buffer

            current_q_values = torch.gather(current_q_values, dim=-1,
                                            index=train_data['action'].unsqueeze(-1).long()).squeeze(-1)
            # Avoid potential broadcast issue

            # Compute Huber loss (less sensitive to outliers)
            #total_loss = F.huber_loss(current_q_values, target_q_values)
            total_loss = torch.nn.MSELoss(reduction='none')(current_q_values, target_q_values).mean()
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

    def transform_action(self, model_output, deterministic):
        if not deterministic and np.random.rand() < self.exploration_rate:
            actions = np.array([np.random.choice(self.action_space) for _ in range(len(model_output['action']))])
        else:
            actions = model_output['action'].cpu().detach().numpy()
        return actions

    @torch.no_grad()
    def collect_episodes(self, env_manager, n_episodes):
        self.model.eval()

        env_num = len(env_manager)
        env_episode_count = [0 for _ in range(env_num)]
        collect_episode = 0
        reward_record = []
        obs_dict = env_manager.reset()
        obs_list = [obs_dict[i] for i in range(env_num)]
        while np.min(env_episode_count) < n_episodes:
            obs_row = self.preprocess_obs(obs_list)
            model_output = self.model.forward(obs_row, )
            actions = self.transform_action(model_output, deterministic=True)
            next_obs, env_rewards, env_dones, env_infos = env_manager.step(actions)
            for env_id in range(env_num):
                if env_dones[env_id]:
                    info = env_infos[env_id]
                    collect_episode += 1
                    if env_episode_count[env_id] < n_episodes:
                        reward_record.append(info['cumulative_rewards'])
                    env_episode_count[env_id] += 1
                    obs_list[env_id] = env_manager.reset(env_id)[env_id]
                    if self.print_eval_result:
                        print(
                            f"Eval Env{env_id} finish its episode, with cumulative_rewards are {info['cumulative_rewards']}")
                else:
                    obs_list[env_id] = next_obs[env_id]
        mean_reward = np.mean(reward_record)

        eval_info = {'rew_mean': np.mean(mean_reward),
                     'rew_min': np.min(reward_record),
                     'rew_max': np.max(reward_record),
                     'rew_std': np.std(reward_record),
                     }

        eval_text = '\n' + "=" * 4 + f'Evaluation_iter{self.last_iter.val}' + "=" * 4 + '\n'
        headers = ['Name', 'Value']
        table_data = [['num', n_episodes * env_num]]
        for key, val in eval_info.items():
            table_data.append([key, f'{val:.3f}'])
        table_text = tabulate(table_data, headers=headers, tablefmt='grid',
                              stralign='left', numalign='left')

        eval_text += table_text

        return mean_reward, eval_info, eval_text

    @torch.no_grad()
    def collect_data(self, env_manager, ):
        if self.last_obs_list is None:
            # this means we haven't collect any train_data
            reset_obs_dict = env_manager.reset()
            self.last_obs_list = [reset_obs_dict[idx] for idx in range(env_manager.env_num)]

        self.model.eval()

        start_time = time.time()
        cumulative_rewards = []
        collect_episode = 0
        env_num = len(env_manager)
        next_obs_list = [None for _ in range(env_num)]
        for i in range(self.rollout_nstep):
            curr_obs_row = self.preprocess_obs(self.last_obs_list)
            model_outputs = self.model.forward(curr_obs_row, )
            actions = self.transform_action(model_outputs, deterministic=False)
            next_obs, rewards, dones, infos = env_manager.step(actions)

            for env_id in range(env_num):
                # if done, put result to result queue
                if dones[env_id]:
                    next_obs_list[env_id] = env_manager.reset(env_id)[env_id]
                    cumulative_reward = infos[env_id]['cumulative_rewards']
                    cumulative_rewards.append(cumulative_reward)
                    collect_episode += 1
                    if self.print_collect_result:
                        print(
                            f"Collect Env{env_id} finish its episode, with cumulative_rewards are {cumulative_reward}")
                else:
                    next_obs_list[env_id] = next_obs[env_id]
                env_step_data = {
                    'obs': self.last_obs_list[env_id],
                    'action': actions[env_id],
                    'next_obs': next_obs_list[env_id],
                    'reward': rewards[env_id],
                    'done': dones[env_id],
                }
                self.replay_buffer.push_data(env_step_data)

            self._n_calls += 1
            if self._n_calls % self.target_update_interval == 0:
                polyak_update(self.model.parameters(), self.target_model.parameters(), self.tau)
            self.last_obs_list = next_obs_list

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

    def setup_optimizer(self) -> None:
        """
        Overview:
            Setup Learner's optimizer, lr_scheduler and grad_clip
        """
        self.optimizer = build_optimizer(cfg=self.cfg.optimizer, params=self.model.parameters())
        self.grad_clip = build_grad_clip(self.cfg.grad_clip)

    def setup_lr_scheduler(self):
        lr_scheduler_cfg = self.cfg.get('lr_scheduler', {})
        self.lr_scheduler_type = lr_scheduler_cfg.get('type', None)
        self.lr_scheduler = build_lr_scheduler(cfg=lr_scheduler_cfg, optimizer=self.optimizer)

    def save_checkpoint(self, checkpoint_dir='.'):
        checkpoint_path = os.path.join(checkpoint_dir,
                                       '{}_iteration_{}.pth.tar'.format(self.whole_cfg.common.experiment_name,
                                                                        self.last_iter.val))
        self.checkpoint_helper.save(checkpoint_path,
                                    model=self.model,
                                    optimizer=self.optimizer,
                                    last_iter=self.last_iter,
                                    )
        return checkpoint_path

    def load_checkpoint(self, ):
        if self.load_checkpoint_path == '' or not os.path.exists(self.load_checkpoint_path):
            print(f"Can't load from path:{self.load_checkpoint_path}")
            return False
        self.checkpoint_helper.load(load_path=self.load_checkpoint_path, model=self.model,
                                    optimizer=self.optimizer,
                                    last_iter=self.last_iter, )
        self.lr_scheduler.last_epoch = self.last_iter.val
        print(f'load checkpoint from {self.load_checkpoint_path}')


def flatten_data(data,start_dim=0,end_dim=1):
    if isinstance(data, dict):
        return {k: flatten_data(v,start_dim=start_dim, end_dim=end_dim) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return torch.flatten(data, start_dim=start_dim, end_dim=end_dim)
