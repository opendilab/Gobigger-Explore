import os
import threading
import time
import uuid

import torch
import torch.multiprocessing as tm
import traceback
from abc import abstractmethod

from bigrl.core.utils.log_helper import TextLogger, ShareVariableRecord
from bigrl.core.utils.time_helper import EasyTimer
from bigrl.single.worker.actor.actor_comm import ActorComm


class BaseActor:
    def __init__(self, cfg):
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.actor
        self.env_name = self.whole_cfg.env.name
        self.job_type = cfg.actor.get('job_type', 'train')
        self.actor_uid = str(uuid.uuid1())
        self.setup_logger()
        self.launch_flag = False

    def launch(self, model_storage,):
        self.launch_flag = True
        self.model_storage = model_storage
        self.error_count = torch.zeros(size=()).share_memory_()
        if self.cfg.get('debug_mode', False):
            p = threading.Thread(target=self.inference_loop,
                                 args=(0, self.model_storage, self.variable_record, self.error_count),
                                 daemon=True)
            p.start()
        else:
            self.worker_processes = []
            for env_id in range(self.cfg.env_num):
                p = tm.Process(target=self.inference_loop,
                               args=(env_id, self.model_storage, self.variable_record, self.error_count),
                               daemon=True)
                p.start()
                self.worker_processes.append(p)

    # def setup_comm(self):
    #     pass

    @abstractmethod
    def _inference_loop(self, env_id, comm, variable_record,):
        raise NotImplementedError

    def run(self):
        try:
            while True:
                if self.error_count >= self.whole_cfg.actor.get('max_error_count', 20):
                    self.close()
                else:
                    time.sleep(1)
        except Exception as e:
            self.logger.error(f'[MAIN LOOP ERROR]:{e}')
            self.logger.error(''.join(traceback.format_tb(e.__traceback__)))
            self.close()

    def inference_loop(self, env_id, model_storage, variable_record,error_count):
        while True:
            try:
                comm = ActorComm(self.whole_cfg)
                comm.launch(model_storage)
                self._inference_loop(env_id, comm, variable_record)
            except Exception as e:
                self.logger.info(f'[Inference Loop Error]{e}',)
                self.logger.info(''.join(traceback.format_tb(e.__traceback__)),)
                error_count.add_(1)
        return True

    def setup_logger(self):
        self.logger = TextLogger(
            path=os.path.join(os.getcwd(), 'experiments', self.whole_cfg.common.experiment_name, 'actor_log'),
            name=self.actor_uid)
        self.timer = EasyTimer(cuda=False)
        register_keys = ['ask_for_job', 'preprocess', 'agent_step',
                         'env_step', 'eval_postprocess', 'collect_data',
                         'push_data',
                         'dumps_data',
                         'send_data', 'update_model', ]
        self.variable_record = ShareVariableRecord(self.cfg.get('record_decay', 0.99),
                                                   self.cfg.get('record_warm_up_size', 100),
                                                   ignore_counts=0)
        self.send_data_counter = torch.tensor([0], dtype=torch.long).share_memory_()
        for k in register_keys:
            self.variable_record.register_var(k)
        self.last_log_show_time = time.time()
        self.log_show_freq = self.cfg.get('log_show_freq', 10)

    # def show_log(self):
    #     info_text = f"\n{'=' * 5}Actor-{self.actor_uid}{'=' * 5}\n{self.variable_record.get_vars_text()}\n"
    #     if self.job_type == 'train':
    #         info_text += f"send_data_count:{self.send_data_counter.item()}"
    #     self.logger.info(info_text)
    #     self.send_data_counter.copy_(torch.tensor([0]))
    #     self.last_log_show_time = time.time()
    def close(self):
        self.logger.info('actor close')
        if hasattr(self, 'worker_processes'):
            for p in self.worker_processes:
                p.terminate()
            for p in self.worker_processes:
                p.join()
