"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. base class for model learning
"""
import os
import shutil
import sys
import time
from abc import ABC, abstractmethod

import torch
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter

from bigrl.core.torch_utils.checkpoint_helper import CountVar, CheckpointHelper
from bigrl.core.torch_utils.grad_clip import build_grad_clip
from bigrl.core.torch_utils.optimizer import build_optimizer
from bigrl.core.utils.log_helper import LossVariableRecord
from bigrl.core.torch_utils.lr_scheduler import build_lr_scheduler
from bigrl.core.utils.config_helper import save_config
from bigrl.single.import_helper import import_pipeline_module
from bigrl.core.utils.log_helper import pretty_print, TextLogger, ShareVariableRecord
from bigrl.core.utils.time_helper import EasyTimer


class BaseLearner(ABC):
    r"""
    Overview:
        base class for model learning(SL/RL), which is able to multi-GPU learning
    Interface:
        __init__, register_stats, run, close, call_hook, info, save_checkpoint, launch
    Property:
        last_iter, optimizer, lr_scheduler, computation_graph, agent, log_buffer, record,
        load_path, save_path, checkpoint_manager, name, tb_logger
    """

    _name = "BaseLearner"  # override this variable for sub-class learner
    use_value_network = True

    def __init__(self, cfg: EasyDict) -> None:
        """
        Overview:
            initialization method, load config setting and call ``_init`` for actual initialization,
            set the communication mode to `single_machine` or `flask_fs`.
        Arguments:
            - cfg (:obj:`EasyDict`): learner config, you can view `cfg <../../../configuration/index.html>`_ for ref.
        Notes:
            if you want to debug in sync CUDA mode, please use the following line code in the beginning of ``__init__``.

            .. code:: python

                os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # for debug async CUDA
        """
        self.whole_cfg = cfg
        self.env_name = self.whole_cfg.env.name
        self.player_id = self.whole_cfg.learner.get('player_id', 'MP0')
        self.pipeline = self.whole_cfg.learner.get('pipeline', 'default')
        self.dir_path = os.path.join(os.getcwd(), 'experiments', cfg.common.experiment_name, self.player_id)

        self.load_path = self.whole_cfg.learner.load_path
        self.experiment_name = self.whole_cfg.common.experiment_name
        self.use_cuda = self.whole_cfg.learner.use_cuda and torch.cuda.is_available()

        self.rank, self.world_size = 0, 1
        self.device = torch.cuda.current_device() if self.use_cuda else 'cpu'
        self.ip = '127.0.0.1'

        # checkpoint helper
        self.checkpoint_manager = CheckpointHelper()
        self.save_checkpoint_freq = self.cfg.get('save_checkpoint_freq', 100)
        self.setup_logger()


    def launch(self):
        self.setup_model()
        self.setup_loss()
        self.setup_dataloader()
        self.setup_optimizer()
        self.setup_lr_scheduler()
        self.remain_ignore_step = self.cfg.get('remain_ignore_step', -1)
        self.default_ignore_step = self.cfg.get('default_ignore_step', -1)

    def run(self, ) -> None:
        """
        Overview:
            Run the learner.
            For each iteration, learner will get training data and train.
            Learner will call hooks at four fixed positions(before_run, before_iter, after_iter, after_run).
        """
        # before train
        self.load_checkpoint()
        while True:
            start = time.time()
            data = self.dataloader.get_data()
            data_time = time.time()-start
            self.variable_record.update_var({'data_time': data_time})
            self.loss_record.update_var({'data_time': data_time})
            total_loss = self.model_forward(data)
            self.model_backward(total_loss)
            self.after_iter()
            self.last_iter.add(1)

    def model_forward(self, data):
        with self.timer:
            with torch.enable_grad():
                total_loss, loss_info_dict = self.model.train(data)
        forward_time = self.timer.value
        self.variable_record.update_var({'forward': forward_time})
        self.loss_record.update_var({'forward': forward_time})
        self.loss_record.update_var(loss_info_dict)
        return total_loss

    def model_backward(self, total_loss, ):
        if self.remain_ignore_step > 0:
            self.remain_ignore_step -= 1
        else:
            with self.timer:
                self.optimizer.zero_grad()
                total_loss.backward()
                gradient = self.grad_clip.apply(self.model.parameters())
                self.optimizer.step()
                self.lr_scheduler.step()
            self.variable_record.update_var({'backward': self.timer.value})
            self.loss_record.update_var({'gradient': gradient,
                                         'lr': self.lr_scheduler.get_last_lr()[0],
                                         'backward': self.timer.value,
                                         })

    def setup_model(self):
        self.ModelClass = import_pipeline_module(self.env_name, self.pipeline, 'Model', )
        self.model = self.ModelClass(self.whole_cfg, use_value_network=self.use_value_network)
        if self.use_cuda:
            self.model = self.model.to(device=self.device)

    @abstractmethod
    def setup_dataloader(self) -> None:
        raise not NotImplementedError

    @abstractmethod
    def setup_loss(self) -> None:
        raise not NotImplementedError

    def setup_optimizer(self) -> None:
        """
        Overview:
            Setup learner's optimizer and lr_scheduler
        """
        self.optimizer = build_optimizer(cfg=self.cfg.optimizer, params=self.model.parameters())
        self.grad_clip = build_grad_clip(self.whole_cfg.learner.grad_clip)

    def setup_lr_scheduler(self):
        self.lr_scheduler = build_lr_scheduler(cfg=self.cfg.get('lr_scheduler',{}),optimizer=self.optimizer)

    def after_iter(self):
        if self.last_iter.val % self.log_show_freq == 0:
            self.log_show()
        if self.last_iter.val % self.save_checkpoint_freq == 0:
            self.save_checkpoint()

    def setup_logger(self):
        self.logger = TextLogger(self.dir_path, name='logger')
        self.timer = EasyTimer(self.use_cuda)
        self.last_iter = CountVar(init_val=0)
        self.log_show_freq = self.cfg.get('log_show_freq', 10)
        self.config_dir = os.path.join(self.dir_path, 'config')
        self.setup_dir(self.config_dir,remove=False)
        self.save_config(init_backup=True)

        self.tb_logger = SummaryWriter(os.path.join(self.dir_path, 'tb_log'))
        self.logger.info(pretty_print({"config": self.whole_cfg, }, direct_print=False))

        self.setup_variable_record()
        self.setup_loss_record()

    def setup_variable_record(self):
        self.variable_record = ShareVariableRecord()
        for k in ['get_data', 'loads', 'collate_fn', 'to_device',
                  'data_time', 'forward', 'backward', ]:
            self.variable_record.register_var(k)

    def setup_loss_record(self):
        self.loss_record = LossVariableRecord(self.log_show_freq,)

    def log_show(self):
        loss_record_info, loss_record_text = self.loss_record.get_vars_info_text()
        self.logger.info('=' * 30 + f'{self.player_id}-{self.last_iter.val}' + '=' * 30)
        self.logger.info(self.variable_record.get_vars_text())
        self.logger.info(loss_record_text)
        for k, val in self.variable_record.var_dict.items():
            self.tb_logger.add_scalar(k, val.val, global_step=self.last_iter.val)
        for k, val in loss_record_info.items():
            self.tb_logger.add_scalar(k, val, global_step=self.last_iter.val)

    def load_checkpoint(self, ):
        if self.load_path == '' or not os.path.exists(self.load_path):
            print(f"Can't load from path:{self.load_path}")
            return False
        self.last_checkpoint_path = self.load_path
        self.checkpoint_manager.load(load_path=self.load_path, model=self.model, optimizer=self.optimizer,
                                     last_iter=self.last_iter, )
        self.lr_scheduler.last_epoch = self.last_iter.val
        self.logger.info(f'{self.player_id} load checkpoint from {self.load_path}')

    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.dir_path, 'checkpoint')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir,
                                       '{}_{}_iteration_{}.pth.tar'.format(self.whole_cfg.common.experiment_name,
                                                                           self.player_id,
                                                                           self.last_iter.val))
        self.checkpoint_manager.save(checkpoint_path,
                                     model=self.model,
                                     optimizer=self.optimizer,
                                     last_iter=self.last_iter,
                                     )
        self.last_checkpoint_path = checkpoint_path

    def setup_dir(self, dir_path,remove=True):
        if remove:
            shutil.rmtree(dir_path, ignore_errors=True)
        os.makedirs(dir_path, exist_ok=True)

    def save_config(self, init_backup=False):
        if init_backup:
            save_config(self.whole_cfg, os.path.join(self.dir_path, f'user_config.yaml'))
        # save cfg
        time_label = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        config_path = os.path.join(self.config_dir, f'user_config_{time_label}.yaml')
        save_config(self.whole_cfg, config_path)

    def close(self) -> None:
        if hasattr(self, 'dataloader') and hasattr(self.dataloader, 'close'):
            self.dataloader.close()
        if hasattr(self, 'comm'):
            self.comm.close()
        sys.exit()

    @property
    def name(self) -> str:
        return self._name + str(id(self))

    @property
    def cfg(self):
        return self.whole_cfg.learner
