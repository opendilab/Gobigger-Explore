import os
import random
import shutil
import time

import numpy as np
import torch

from bigrl.core.utils.config_helper import save_config
from bigrl.core.utils.log_helper import TextLogger, pretty_print, LogVariableRecord
from torch.utils.tensorboard import SummaryWriter
from bigrl.core.env.env_manager import SyncVectorEnv
from bigrl.core.utils.config_helper import deep_merge_dicts
from bigrl.serial.import_helper import import_env_module
from functools import partial
from easydict import EasyDict
from bigrl.serial.import_helper import import_pipeline_agent


class Learner:
    def __init__(self, cfg):
        self.whole_cfg = cfg
        self.exp_dir = os.path.join(os.getcwd(), 'experiments', self.whole_cfg.common.experiment_name, )
        self.seed = self.whole_cfg.common.get('seed', None)
        self.env_name = self.whole_cfg.env.name
        self.pipeline = self.whole_cfg.agent.pipeline
        self.setup_logger()

        # evaluation related
        self.eval_freq = self.whole_cfg.evaluate.eval_freq
        self.eval_episodes_num = self.whole_cfg.evaluate.eval_episodes_num
        self.stop_value = self.whole_cfg.evaluate.stop_value
        self.total_timesteps = float(self.cfg.n_timesteps)

        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoint')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_checkpoint_freq = self.cfg.save_checkpoint_freq

        # setup env_manager, random_seed, agent
        self.setup_env_manager()
        self.set_random_seed()
        self.setup_agent()

    def set_random_seed(self):
        if self.seed is None:
            return
        set_random_seed(self.seed)
        self.collect_env_manager.seed(self.seed)
        self.eval_env_manager.seed(self.seed)

    def setup_env_manager(self):
        env_fn = import_env_module(self.env_name, 'make_env')
        collect_cfg = EasyDict(self.whole_cfg.get('collect',{}))
        self.collect_env_manager = self._setup_env_manager(env_fn, collect_cfg)
        evaluate_cfg = EasyDict(self.whole_cfg.get('evaluate',{}))
        self.eval_env_manager = self._setup_env_manager(env_fn, evaluate_cfg)

    def _setup_env_manager(self, env_fn, cfg):
        env_cfg = deep_merge_dicts(self.whole_cfg.env, cfg.get('env', {}))
        env_num = cfg.env_num
        env_manager = SyncVectorEnv(
            [partial(env_fn, env_cfg) for _ in range(env_num)])
        return env_manager

    def setup_agent(self):
        Agent = import_pipeline_agent(self.env_name, self.pipeline, 'Agent')
        self.agent = Agent(self.whole_cfg,)

    def setup_logger(self):
        self.logger = TextLogger(self.exp_dir, name='logger')
        self.logger.info(pretty_print({"config": self.whole_cfg, }, direct_print=False))
        self.tb_logger = SummaryWriter(os.path.join(self.exp_dir, 'tb_log'))

        self.config_dir = os.path.join(self.exp_dir, 'config')
        self.setup_dir(self.config_dir, remove=False)
        self.save_config(init_backup=True)

        # log show related
        self.log_show_freq = self.cfg.log_show_freq
        self.log_record = LogVariableRecord()
        self.reset_stats()

    def reset_stats(self):
        # timestep
        self.last_eval_timestep = 0
        self.last_collect_log_show_timestep = 0
        self.last_loss_show_timestep = 0
        self.last_save_ckpt_timestep = 0
        
    def run(self):
        """
        Overview:
            Serial pipeline entry.
        Arguments:
            - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
                ``str`` type means config file path. \
                ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        Returns:
            - policy (:obj:`Policy`): Converged policy.
        """
        converge_flag = False
        while self.agent.total_collect_timesteps < self.total_timesteps:
            
            if self.agent.total_collect_timesteps==0 or (self.agent.total_collect_timesteps - self.last_save_ckpt_timestep >= self.save_checkpoint_freq):
                self.last_save_ckpt_timestep = self.agent.total_collect_timesteps
                checkpoint_path = self.agent.save_checkpoint(self.checkpoint_dir)
                self.logger.info(f'save checkpoint {self.agent.last_iter.val}_{self.agent.total_collect_timesteps} to {checkpoint_path}!')

            # Evaluate policy performance
            if self.agent.total_collect_timesteps==0 or (self.agent.total_collect_timesteps - self.last_eval_timestep >= self.eval_freq):
                should_stop_selfplay = self.eval_agent()
                should_stop_vsbot = self.eval_vsbot()
                self.last_eval_timestep = self.agent.total_collect_timesteps
                if should_stop_selfplay and should_stop_vsbot:
                    converge_flag = True
                    break

            # Collect data by default config n_sample/n_episode
            train_data, collect_info = self.agent.collect_data(self.collect_env_manager, )
            self.log_record.update_var(collect_info)

            # Learn policy from collected data
            log_record_list = self.agent.train(train_data)
            for log_record in log_record_list:
                self.log_record.update_var(log_record)

            if self.agent.total_collect_timesteps - self.last_loss_show_timestep >= self.log_show_freq:
                self.last_loss_show_timestep = self.agent.total_collect_timesteps
                self.log_show()

        if not converge_flag:
            should_stop_selfplay = self.eval_agent()
            should_stop_vsbot = self.eval_vsbot()
            self.last_eval_timestep = self.agent.total_collect_timesteps
        return None



    def eval_agent(self):
        self.eval_env_manager.seed(self.seed)
        mean_reward, eval_info, eval_text  = self.agent.collect_episodes(self.eval_env_manager, self.eval_episodes_num)
        self.logger.info(eval_text,)
        for k, val in eval_info.items():
            self.tb_logger.add_scalar(tag='eval/' + k, scalar_value=val, global_step=self.agent.total_collect_timesteps)
        self.tb_logger.add_scalar(tag='eval/iter', scalar_value=self.agent.last_iter.val, global_step=self.agent.total_collect_timesteps)
        if mean_reward  >= self.stop_value:
            self.logger.info(f'agent has achieved reward {mean_reward}, converged!')
            checkpoint_path = self.agent.save_checkpoint(self.checkpoint_dir)
            self.logger.info(f'save convergent checkpoint {self.agent.last_iter.val} to {checkpoint_path}!')
            return True
        return False
    
    def eval_vsbot(self):
        self.eval_env_manager.seed(self.seed)
        mean_reward, eval_info, eval_text  = self.agent.collect_episodes_vsbot(self.eval_env_manager, self.eval_episodes_num)
        self.logger.info(eval_text,)
        for k, val in eval_info.items():
            self.tb_logger.add_scalar(tag='eval_vsbot/' + k, scalar_value=val, global_step=self.agent.total_collect_timesteps)
        self.tb_logger.add_scalar(tag='eval_vsbot/iter', scalar_value=self.agent.last_iter.val, global_step=self.agent.total_collect_timesteps)
        if mean_reward  >= self.stop_value:
            self.logger.info(f'agent has achieved reward {mean_reward}, converged!')
            checkpoint_path = self.agent.save_checkpoint(self.checkpoint_dir)
            self.logger.info(f'save convergent checkpoint {self.agent.last_iter.val} to {checkpoint_path}!')
            return True
        return False

    def log_show(self):
        show_text = '\n'+'=' * 12 + f'Iteration_{self.agent.last_iter.val}_EnvStep{self.agent.total_collect_timesteps}' + '=' * 12+'\n'
        show_text += self.log_record.get_vars_text()
        self.logger.info(show_text )
        for k, val in self.log_record.var_dict.items():
            self.tb_logger.add_scalar(k, val.val, global_step=self.agent.total_collect_timesteps)
        self.tb_logger.add_scalar(tag='train/iter',
                                  scalar_value=self.agent.last_iter.val,
                                  global_step=self.agent.total_collect_timesteps)
        self.log_record.reset()

    def save_config(self, init_backup=False):
        if init_backup:
            save_config(self.whole_cfg, os.path.join(self.exp_dir, f'user_config.yaml'))
        # save cfg
        time_label = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        config_path = os.path.join(self.config_dir, f'user_config_{time_label}.yaml')
        save_config(self.whole_cfg, config_path)

    def setup_dir(self, dir_path, remove=True):
        if remove:
            shutil.rmtree(dir_path, ignore_errors=True)
        os.makedirs(dir_path, exist_ok=True)

    @property
    def cfg(self):
        return self.whole_cfg.learner

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # Deterministic operations for CuDNN, it may impact performances
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
