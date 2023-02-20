import threading
import time
import traceback
from copy import deepcopy
from functools import partial

import torch
import torch.multiprocessing as tm

from bigrl.core.torch_utils.data_helper import to_device, to_share, to_contiguous, to_pin_memory
from bigrl.core.utils import EasyTimer
from bigrl.single.import_helper import import_pipeline_module
from bigrl.single.buffer import get_buffer_class


class RLDataLoader(object):
    def __init__(self, learner, ) -> None:
        torch.set_num_threads(1)
        self.whole_cfg = learner.whole_cfg
        self.env_name = learner.env_name
        self.pipeline = learner.pipeline
        self.dir_path = learner.dir_path
        self.use_cuda = learner.use_cuda
        self.device = learner.device
        self.rank = learner.rank
        self.world_size = learner.world_size
        self.ip = learner.ip
        self.player_id = learner.player_id
        self.logger = learner.logger
        self.variable_record = learner.variable_record
        self.timer = EasyTimer(self.use_cuda)
        self.fake_dataloader = self.whole_cfg.learner.data.get('fake_dataloader', False)
        self.debug_mode = self.whole_cfg.learner.get('debug_mode', False)

        if self.fake_dataloader:
            print(f'use fake dataloader {self.fake_dataloader}')

        self.max_use = self.whole_cfg.learner.data.get('max_use',1)
        self.remaining_use = 0
        self.cache_data = None
        if self.debug_mode:
            self.batch_size = 1
            self.worker_num = 1
        else:
            self.batch_size = self.whole_cfg.learner.data.get('batch_size', 1)
            self.worker_num = self.whole_cfg.learner.data.get('worker_num', 1)
        self.unroll_len = self.whole_cfg.learner.data.get('unroll_len', 1)
        Features = import_pipeline_module(self.env_name, self.pipeline, 'Features')
        features = Features(self.whole_cfg)
        get_rl_batch_data = features.get_rl_batch_data
        self.use_pin_memory = self.whole_cfg.learner.data.get('pin_memory', False) and self.use_cuda

        self.shared_data = to_share(
            to_contiguous(get_rl_batch_data(unroll_len=self.unroll_len, batch_size=self.batch_size)))
        if self.use_pin_memory:
            self.shared_data = to_pin_memory(self.shared_data)

        if not self.fake_dataloader:
            self.start_worker_process()

    def start_worker_process(self):
        self.worker_processes = []
        self.signal_queue = tm.Queue(maxsize=self.batch_size)
        self.done_flags = torch.tensor([False for _ in range(self.batch_size)]).share_memory_()
        worker_loop = partial(_worker_loop, signal_queue=self.signal_queue, done_flags=self.done_flags,
                              shared_data=self.shared_data, cfg=self.whole_cfg,
                              variable_record=self.variable_record)

        for worker_idx in range(self.worker_num):
            if not self.debug_mode:
                worker_process = tm.Process(target=worker_loop,
                                            args=(),
                                            daemon=True)
            else:
                worker_process = threading.Thread(target=worker_loop,
                                                  args=(),
                                                  daemon=True)
            worker_process.start()
            self.worker_processes.append(worker_process)

        for idx in range(self.batch_size):
            self.signal_queue.put(idx)

    def get_data(self):
        if not self.fake_dataloader:
            if self.remaining_use > 0:
                self.remaining_use -= 1
                return self.cache_data
            self.remaining_use = self.max_use
            while True:
                if (self.done_flags == True).all():
                    break
                else:
                    time.sleep(0.001)
        if self.use_cuda:
            with self.timer:
                batch_data = to_device(self.shared_data, self.device)
            self.variable_record.update_var({'to_device': self.timer.value})
        else:
            with self.timer:
                batch_data = deepcopy(self.shared_data)
            self.variable_record.update_var({'to_device': self.timer.value})
        self.cache_data = batch_data
        if not self.fake_dataloader:
            self.done_flags.copy_(torch.zeros_like(self.done_flags))
            for batch_idx in range(self.batch_size):
                self.signal_queue.put(batch_idx)
        return batch_data

    def close(self):
        if not self.fake_dataloader:
            self.replay_buffer.close()
            print('has already close all subprocess in RLdataloader')
        return True


def _worker_loop(signal_queue, done_flags, shared_data, cfg, variable_record):
    torch.set_num_threads(1)
    timer = EasyTimer(cuda=False)
    replay_buffer_type = cfg.learner.data.get('type', 'max_use')
    replay_buffer_class = get_buffer_class(replay_buffer_type)
    replay_buffer = replay_buffer_class(cfg, type='receiver')
    replay_buffer.launch()
    while True:
        if signal_queue.qsize() > 0:
            batch_idx = signal_queue.get()
            with timer:
                traj_data = replay_buffer.get_data()
            variable_record.update_var({'get_data': timer.value})

            with timer:
                copy_data(batch_idx, traj_data=traj_data, shared_data=shared_data)
            variable_record.update_var({'collate_fn': timer.value})

            done_flags[batch_idx] = True

        else:
            time.sleep(0.001)


def _copy_data(dest_tensor, src_tensor, key=''):
    if dest_tensor.shape == src_tensor.shape:
        if dest_tensor.dtype != src_tensor.dtype:
            print(f'{key} dtype not same, dest: {dest_tensor.dtype}, src: {src_tensor.dtype}', flush=True)
        dest_tensor.copy_(src_tensor)
        return True
    else:
        print(key, dest_tensor.shape, src_tensor.shape)
        print(key, dest_tensor.dtype, src_tensor.dtype)
        raise NotImplementedError
        return False


def copy_data(batch_idx, traj_data, shared_data):
    for k, v in traj_data.items():
        if isinstance(v, torch.Tensor):
            try:
                _copy_data(shared_data[k][:, batch_idx], traj_data[k], key=k)
            except Exception as e:
                print(k, e, flush=True)
                print(''.join(traceback.format_tb(e.__traceback__)), flush=True)
        elif isinstance(v, dict):
            copy_data(batch_idx, v, shared_data[k])
        else:
            print(k, type(v))
            raise NotImplementedError
