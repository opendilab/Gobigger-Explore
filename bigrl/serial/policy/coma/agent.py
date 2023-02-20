import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate

from bigrl.core.torch_utils.checkpoint_helper import CountVar, CheckpointHelper
from bigrl.core.torch_utils.collate_fn import default_collate_with_dim
from bigrl.core.torch_utils.data_helper import to_device
from bigrl.core.torch_utils.grad_clip import build_grad_clip
from bigrl.core.torch_utils.lr_scheduler import build_lr_scheduler
from bigrl.core.torch_utils.optimizer import build_optimizer
from .rl_loss import ReinforcementLoss
from bigrl.core.utils.time_helper import EasyTimer
from bigrl.serial.import_helper import import_pipeline_module


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
        self.use_value_feature = self.whole_cfg.agent.get('use_value_feature', True)

        # setup model
        self.setup_model()
        # following is only used when train agent

        self.action_space = list(range(self.cfg.action_num))

        # loss ralated
        self.loss = ReinforcementLoss(self.whole_cfg)
        self.setup_optimizer()
        self.setup_lr_scheduler()
        self.update_per_collect = self.cfg.update_per_collect

        # checkpoint related
        self.checkpoint_helper = CheckpointHelper()
        self.load_checkpoint_path = self.cfg.get('load_checkpoint_path', '')
        self.load_checkpoint()
        self.rollout_nstep = self.cfg.rollout_nstep

        # For updating the target network with multiple envs:
        self.use_double = self.cfg.get('use_double', False)
        self.target_update_interval = self.cfg.target_update_interval
        self.gamma = self.cfg.loss_parameters.gamma
        self.batch_size = self.cfg.replay_buffer.batch_size

        self._n_calls = 0

    def reset_stat(self):
        self.last_iter = CountVar(init_val=0)
        self.total_collect_timesteps = 0
        self.total_collect_episode = 0
        self.n_updates = 0

    def setup_model(self):
        self.ModelClass = import_pipeline_module(self.env_name, self.pipeline, 'Model', )
        self.model = self.ModelClass(self.whole_cfg,).to(device=self.device)
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

    def train(self,train_data):
        self.model.train()

        flatten_train_data = flatten_data(train_data)
        log_record_list = []
        for i in range(self.cfg.update_per_collect):
            # Learner will train ``update_per_collect`` times in one iteration.
            train_data_size = flatten_train_data['action'].shape[0]
            indices = list(range(train_data_size))
            np.random.shuffle(indices)
            for start_idx in range(0, train_data_size, self.batch_size):
                train_sample = get_data_from_indices(flatten_train_data,
                                                     indices[start_idx:start_idx + self.batch_size])
                log_info_dict = self.update(train_sample)
                log_record_list.append(log_info_dict)
        self.last_iter.add(1)
        return log_record_list

    def update(self, train_data):
        with self.timer:
            with torch.enable_grad():
                model_output = self.model.rl_train(train_data)
            total_loss, loss_info_dict = self.loss.compute_loss(model_output)
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
        loss_info_dict.update({'train_forward': train_forward,
                               'train_backward': train_background,
                               'gradient': gradient,
                               'lr': self.lr_scheduler.get_last_lr()[0],
                               'clip_range': self.loss.clip_range,
                               'clip_range_vf': self.loss.clip_range_vf
                               })
        return loss_info_dict

    @torch.no_grad()
    def collect_episodes(self, env_manager, n_episodes):
        self.model.eval()
        env_num = len(env_manager)
        env_episode_count = [0 for _ in range(env_num)]
        collect_episode = 0
        cumulative_rewards = []
        obs_dict = env_manager.reset()
        obs_list = [obs_dict[i] for i in range(env_num)]
        while np.min(env_episode_count) < n_episodes:
            obs_row = self.preprocess_obs(obs_list)
            model_output = self.model.forward(obs_row, temperature=0)
            actions = model_output['action'].cpu().detach().numpy()
            next_obs, env_rewards, env_dones, env_infos = env_manager.step(actions)
            for env_id in range(env_num):
                if env_dones[env_id]:
                    env_info = env_infos[env_id]
                    collect_episode += 1
                    if env_episode_count[env_id] < n_episodes:
                        cumulative_rewards.append(env_info['cumulative_rewards'])
                    env_episode_count[env_id] += 1
                    obs_list[env_id] = env_manager.reset(env_id)[env_id]
                    if self.print_eval_result:
                        print(f"Eval Env{env_id} finish its episode, with cumulative_rewards are {env_info['cumulative_rewards']}")

        mean_reward = np.mean(cumulative_rewards)

        eval_info = {'rew_mean': np.mean(mean_reward),
                     'rew_min': np.min(cumulative_rewards),
                     'rew_max': np.max(cumulative_rewards),
                     'rew_std': np.std(cumulative_rewards), }
        eval_text = '\n' + "=" * 4 + f'Evaluation_iter{self.last_iter.val}' + "=" * 4 +'\n'
        headers = ['Name','Value']
        table_data = [['num', n_episodes * env_num ]]
        for key, val in eval_info.items():
            table_data.append([key, f'{val:.3f}'])
        table_text = tabulate(table_data, headers=headers, tablefmt='grid',
                                          stralign='left', numalign='left')

        eval_text += table_text

        return mean_reward, eval_info, eval_text

    @torch.no_grad()
    def collect_data(self, env_manager, obs_row):
        self.model.eval()

        start_time = time.time()
        cumulative_rewards = []
        collect_episode = 0

        env_num = len(env_manager)
        obs_list = []
        action_list = []
        reward_list = []
        done_list = []
        value_list = []
        logp_list = []
        next_obs_list = [None for _ in range(env_num)]
        curr_obs_row = obs_row
        for i in range(self.rollout_nstep):
            agent_outputs = self.model.compute_logp_action(curr_obs_row, )
            actions = agent_outputs['action'].cpu().numpy()
            next_obs, env_rewards, env_dones, env_infos = env_manager.step(actions)
            obs_list.append(curr_obs_row)
            logp_list.append(agent_outputs['action_logp'])
            action_list.append(agent_outputs['action'])
            value_list.append(agent_outputs['value'])
            reward_list.append(default_collate_with_dim(env_rewards, device=self.device))
            done_list.append(default_collate_with_dim(env_dones, device=self.device))

            for env_id in range(env_num):
                # if done, put result to result queue
                if env_dones[env_id]:
                    env_info = env_infos[env_id]
                    next_obs_list[env_id] = env_manager.reset(env_id)[env_id]
                    cumulative_reward = env_info['cumulative_rewards']
                    cumulative_rewards.append(cumulative_reward)
                    collect_episode += 1
                    if self.print_collect_result:
                        print(f"Collect Env{env_id} finish its episode, with cumulative_rewards are {cumulative_reward}")
                else:
                    next_obs_list[env_id] = next_obs[env_id]
            curr_obs_row = self.preprocess_obs(next_obs_list, )

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
        self.total_collect_timesteps += collect_timestep
        self.total_collect_episode += collect_episode
        self._current_progress_remaining = max(1 - self.total_collect_timesteps/self.total_timesteps,0)
        return train_data, curr_obs_row, collect_info

    def setup_optimizer(self) -> None:
        """
        Overview:
            Setup Learner's optimizer, lr_scheduler and grad_clip
        """
        self.optimizer = build_optimizer(cfg=self.cfg.optimizer, params=self.model.parameters())
        self.grad_clip = build_grad_clip(self.cfg.grad_clip)
        if self.use_value_feature:
            self.critic_optimizer = build_optimizer(cfg=self.cfg.optimizer, params=self.model.value_head.parameters())
            self.critic_grad_clip = build_grad_clip(self.cfg.grad_clip)

    def setup_lr_scheduler(self):
        lr_scheduler_cfg = self.cfg.get('lr_scheduler',{})
        self.lr_scheduler_type = lr_scheduler_cfg.get('type', None)
        self.lr_scheduler = build_lr_scheduler(cfg=lr_scheduler_cfg,optimizer=self.optimizer)

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


def flatten_data(data):
    if isinstance(data, dict):
        return {k: flatten_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return torch.flatten(data, start_dim=0, end_dim=1)


def get_data_from_indices(data, indices):
    if isinstance(data, dict):
        return {k: get_data_from_indices(v, indices) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data[indices]