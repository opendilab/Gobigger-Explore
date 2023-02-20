import os
import time

import torch

from bigrl.core.utils.config_helper import read_config, deep_merge_dicts
from bigrl.single.worker.learner.base_learner import BaseLearner
from bigrl.single.worker.learner.learner_comm import LearnerComm
from bigrl.single.worker.learner.rl_dataloader import RLDataLoader
from bigrl.single.import_helper import import_pipeline_agent,import_pipeline_module
class BaseRLLearner(BaseLearner):
    def __init__(self, cfg):
        super(BaseRLLearner, self).__init__(cfg)
        self.job_type = self.cfg.get('job_type', 'train')
        # setup dir

        self.reset_value_flag = False
        self.update_config_flag = False

        self.default_value_pretrain_iters = self.whole_cfg.learner.get('default_value_pretrain_iters', -1)
        self.remain_value_pretrain_iters = self.whole_cfg.learner.get('remain_value_pretrain_iters', -1)

    def launch(self,model_storage):
        super(BaseRLLearner, self).launch()
        self.setup_comm()
        self.model_storage = model_storage
        self.comm.launch(model_storage=self.model_storage)

    def setup_loss(self):
        self.LossClass = import_pipeline_agent(self.env_name, self.pipeline, 'ReinforcementLoss')
        self.loss = self.LossClass(self.whole_cfg)

    def setup_comm(self):
        self.comm = LearnerComm(self)

    def setup_dataloader(self):
        DataLoaderClass = import_pipeline_module(self.env_name, self.pipeline, 'RLDataLoader')
        self.DataLoaderClass = DataLoaderClass if DataLoaderClass else RLDataLoader
        self.dataloader = self.DataLoaderClass(self)

    def run(self):
        # before train
        self.load_checkpoint()
        self.comm.send_model(self, ignore_freq=True)
        while True:
            start = time.time()
            data = self.dataloader.get_data()
            data_time = time.time()-start
            self.variable_record.update_var({'data_time': data_time})
            self.loss_record.update_var({'data_time': data_time})

            staleness_info = self.get_staleness_info(data)
            self.variable_record.update_var(staleness_info)

            self.step_value_pretrain()
            total_loss =self.model_forward(data)
            self.model_backward(total_loss)

            self.after_iter()
            self.last_iter.add(1)

    def model_forward(self,data):
        with self.timer:
            with torch.enable_grad():
                data = self.model.rl_train(data)
            total_loss, loss_info_dict = self.loss.compute_loss(data)
        forward_time = self.timer.value
        self.variable_record.update_var({'forward': forward_time})
        self.loss_record.update_var({'forward': forward_time})
        self.loss_record.update_var(loss_info_dict)
        return total_loss

    def after_iter(self):
        super(BaseRLLearner, self).after_iter()
        if self.update_config_flag:
            self.update_config()
            self.update_config_flag = False
        if self.reset_value_flag:
            self.reset_value()
            self.reset_value_flag = False
        self.comm.send_model(self)
        self.comm.send_train_info(self)


    def step_value_pretrain(self):
        if self.remain_value_pretrain_iters > 0:
            self.loss.only_update_value = True
            self.remain_value_pretrain_iters -= 1
            self.model.only_update_value = True
            self.logger.info(f'only update baseline: {self.model.only_update_value}')

        elif self.remain_value_pretrain_iters == 0:
            self.loss.only_update_value = False
            self.remain_value_pretrain_iters -= 1
            self.logger.info('value pretrain iter is 0')
            self.model.only_update_value = False
            self.logger.info(f'only update baseline: {self.model.only_update_value}')

    def get_staleness_info(self, data):
        if self.remain_value_pretrain_iters > 0:
            staleness_mean = 0
            staleness_std = 0
            staleness_max = 0
        else:
            model_last_iter = data.get('model_last_iter')
            model_curr_iter = self.last_iter.val
            iter_diff = model_curr_iter - model_last_iter
            staleness_std, staleness, = torch.std_mean(iter_diff,unbiased=False)
            staleness_std = staleness_std.item()
            staleness_mean = staleness.item()
            staleness_max = torch.max(iter_diff).item()
        return {'staleness_std': staleness_std,
                'staleness_mean': staleness_mean,
                'staleness_max': staleness_max,
                }

    def setup_variable_record(self):
        super(BaseRLLearner, self).setup_variable_record()
        for k in ['send_model','redis_model','staleness_std','staleness_mean','staleness_max' ]:
            self.variable_record.register_var(k)

    def update_config(self):
        load_config_path = os.path.join(self.dir_path, f'user_config.yaml')
        load_config = read_config(load_config_path)
        player_id = self.whole_cfg.learner.player_id
        self.whole_cfg = deep_merge_dicts(self.whole_cfg, load_config)
        self.whole_cfg.learner.player_id = player_id
        self.setup_loss()
        self.default_value_pretrain_iters = self.whole_cfg.learner.get('default_value_pretrain_iters', -1)
        self.remain_value_pretrain_iters = self.whole_cfg.learner.get('remain_value_pretrain_iters', -1)
        for g in self.optimizer.param_groups:
            g['lr'] = self.whole_cfg.learner.optimizer.learning_rate
        self.logger.info(f'update config from config_path:{load_config_path}')
        self.save_config()

    def reset_rank_zero_value(self):
        ref_model = self.ModelClass(self.whole_cfg, use_value_network=True)
        value_state_dict = {k: val for k, val in ref_model.state_dict().items() if 'value' in k or 'auxiliary' in k}
        self.model.load_state_dict(value_state_dict, strict=False)

    def reset_value(self):
        self.reset_rank_zero_value()
        self.setup_optimizer()
        self.logger.info(f'successfully reset_value')
        print(f'rank {self.rank} successfully reset_value')
