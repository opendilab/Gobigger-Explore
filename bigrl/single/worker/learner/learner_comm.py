import os
import sys
import time
import traceback

import requests
import torch


class LearnerComm:
    def __init__(self, learner, ):
        self.whole_cfg = learner.whole_cfg
        self.dir_path = learner.dir_path
        self.player_id = learner.player_id
        self.logger = learner.logger
        self.batch_size = self.whole_cfg.learner.data.batch_size
        self.unroll_len = self.whole_cfg.learner.data.unroll_len
        self.ip = learner.ip
        self.variable_record = learner.variable_record
        self.learner = learner
        # communication related
        self.league_url_prefix = 'http://{}:{}/'.format(
            self.whole_cfg.communication.league_ip, self.whole_cfg.communication.league_port)
        self.register_learner(learner)

        # send train_info setting
        self.send_train_info_freq = self.whole_cfg.communication.learner_send_train_info_freq
        self.send_train_info_count = 0

        # send model setting
        self.send_model_freq = self.whole_cfg.communication.learner_send_model_freq
        self.send_model_count = 0

    def launch(self, model_storage):
        self.model_storage = model_storage
        self.model_storage.push(player_id=self.player_id, model_class=self.learner.ModelClass,
                                cfg=self.learner.whole_cfg, last_iter=self.learner.last_iter.val)
        self.model_storage.update(player_id=self.player_id, model=self.learner.model,
                                  last_iter=self.learner.last_iter.val)

    def register_learner(self, learner) -> None:
        request_info = {'player_id': learner.player_id,
                        'ip': learner.ip,
                        'port': '1234',
                        'rank': learner.rank,
                        'world_size': learner.world_size, }
        while True:
            result = self.flask_send(request_info, 'league/register_learner')
            if result is not None and result['code'] == 0:
                if not (learner.load_path and os.path.exists(learner.load_path)):
                    learner.load_path = result['info']['checkpoint_path']
                # print(f'learner load path is :{learner.load_path}')
                return
            else:
                time.sleep(1)

    def send_model(self, learner, ignore_freq=False) -> None:
        torch.set_num_threads(1)
        if ignore_freq or (
                self.send_model_count % self.send_model_freq == 0 and learner.remain_value_pretrain_iters < 0):
            start = time.time()

            self.model_storage.update(player_id=self.player_id, model=learner.model,
                                      last_iter=self.learner.last_iter.val)
            self.variable_record.update_var({'send_model': time.time() - start})

        if not ignore_freq:
            self.send_model_count += 1

    def send_train_info(self, learner):
        torch.set_num_threads(1)
        self.send_train_info_count += 1
        reset_checkpoint_path = 'none'
        if learner.rank == 0 and self.send_train_info_count % self.send_train_info_freq == 0:
            frames = int(
                self.send_train_info_freq * learner.world_size * self.batch_size * self.unroll_len)
            request_info = {'player_id': self.player_id, 'train_steps': frames,
                            'checkpoint_path': os.path.abspath(learner.last_checkpoint_path)}
            for try_times in range(10):
                result = self.flask_send(request_info, 'league/learner_send_train_info')
                if result is not None and result['code'] == 0:
                    reset_checkpoint_path = result['info']['reset_checkpoint_path']
                    break
                else:
                    time.sleep(1)
            if reset_checkpoint_path != 'none':
                learner.checkpoint_manager.load(
                    reset_checkpoint_path,
                    model=learner.model,
                    logger_prefix='({})'.format(learner.name),
                    strict=False,
                    info_print=learner.rank == 0,
                )
                learner.reset_rank_zero_value()
                learner.remain_value_pretrain_iters = learner.default_value_pretrain_iters
                self.logger.info(
                    '{} reset checkpoint in {}!!!!!!!!!!!!!!!!!'.format(learner.comm.player_id, reset_checkpoint_path))
                learner.comm.send_model(learner, ignore_freq=True)

    def flask_send(self, data: dict, api: str, ) -> dict:
        torch.set_num_threads(1)
        response = None
        t = time.time()
        try:
            response = requests.post(self.league_url_prefix + api, json=data).json()
            if response['code'] == 0:
                pass
                # self.logger.info(
                #     "{} succeed sending result: {}, cost time: {:.4f}".format(api, self.player_id, time.time() - t))
            else:
                self.logger.error(
                    "{} failed to send result: {}, cost time: {:.4f}".format(api, self.player_id, time.time() - t))
        except Exception as e:
            self.logger.error(''.join(traceback.format_tb(e.__traceback__)))
            self.logger.error("api({}): {}".format(api, sys.exc_info()))
        return response

    def close(self):
        for p in self.send_model_processes:
            p.terminate()
        time.sleep(1)
        print('close subprocess in learner_comm')
