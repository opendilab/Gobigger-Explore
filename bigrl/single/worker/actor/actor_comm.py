import sys
import time

import requests
import torch
from bigrl.single.buffer import get_buffer_class


class ActorComm:
    def __init__(self, cfg):
        self.whole_cfg = cfg
        self.env_name = self.whole_cfg.env.name
        self.league_url_prefix = 'http://{}:{}/'.format(
            self.whole_cfg.communication.league_ip, self.whole_cfg.communication.league_port)
        self.job_type = self.whole_cfg.actor.job_type
        self.use_replay_buffer = self.job_type not in {'ladder'}
        if self.use_replay_buffer:
            replay_buffer_type = self.whole_cfg.learner.data.get('type', 'max_use')
            replay_buffer_class = get_buffer_class(replay_buffer_type)
            self.replay_buffer = replay_buffer_class(cfg=self.whole_cfg, type='sender')

    def launch(self, model_storage, ):
        self.model_storage = model_storage
        if self.use_replay_buffer:
            self.replay_buffer.launch()

    def ask_for_job(self, ):
        torch.set_num_threads(1)
        request_info = {'job_type': self.job_type, }
        while True:
            result = self.flask_send(request_info, 'league/actor_ask_for_job', )
            if result is not None and result['code'] == 0:
                job = result['info']
                break
            else:
                time.sleep(3)
        return job

    def send_data(self, traj_data,player_id='MP0'):
        self.replay_buffer.push(traj_data)

    ## send result
    def send_result(self, result_info):
        torch.set_num_threads(1)
        self.flask_send(result_info, 'league/actor_send_result', )

    def flask_send(self, data, api, ):
        torch.set_num_threads(1)
        response = None
        t = time.time()
        try:
            response = requests.post(self.league_url_prefix + api, json=data).json()
            if response['code'] == 0:
                pass
                # self.logger.info("{} succeed sending result: {}, cost_time: {}".format(api, name, time.time() - t))
            else:
               print("failed to send result: {}, cost_time: {}".format(api, time.time() - t))
        except Exception as e:
            # self.logger.error(''.join(traceback.format_tb(e.__traceback__)))
            print(f"[error] url{self.league_url_prefix} api({api}): {sys.exc_info()}")
        return response
