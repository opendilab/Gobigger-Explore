import argparse
import time
from copy import deepcopy

import portpicker
import torch
import torch.multiprocessing as tm

from bigrl.core.utils import read_config
from bigrl.single.import_helper import import_pipeline_agent, import_env_module
from bigrl.single.buffer import get_buffer_class

from bigrl.single.storage.model_storage import ModelStorage


def get_args():
    parser = argparse.ArgumentParser(description="rl_train")
    parser.add_argument("--config", "-c", type=str, default='user_config.yaml', help='config_path')
    return parser.parse_args()


class Trainer:
    def __init__(self, cfg):
        cfg.communication.league_ip = 'localhost'
        league_port = cfg.communication.get('league_port', None)
        if league_port is None or league_port in {'none', 'None', ''} or not portpicker.is_port_free(league_port):
            new_league_port = portpicker.pick_unused_port()
            cfg.communication.league_port = new_league_port
            print(f'league port:{league_port} is not accessible, use new port:{new_league_port}!')
        else:
            print(f'league port is {league_port}!')
        cfg.communication.redis_ip = 'localhost'
        cfg.communication.redis_port = portpicker.pick_unused_port()
        self.whole_cfg = cfg
        self.workers = []
        self.launch()

    def setup_league(self):
        self.league_process = tm.Process(target=self._start_league_loop, args=(self.whole_cfg,), daemon=True)
        self.league_process.start()

    @staticmethod
    def _start_league_loop(whole_cfg):
        torch.set_num_threads(1)
        League = import_env_module(whole_cfg.env.name, 'League')
        league = League(whole_cfg)
        league_app = league.create_league_app(league)
        league_app.run(host=whole_cfg.communication.league_ip, port=whole_cfg.communication.league_port, )

    def setup_replay_buffer(self):
        replay_buffer_type = self.whole_cfg.learner.data.get('type', 'max_use')
        replay_buffer_class = get_buffer_class(replay_buffer_type)

        self.replay_buffer = replay_buffer_class(self.whole_cfg, type='server')
        self.replay_buffer.launch()

    def setup_model_storage(self):
        self.model_storage = ModelStorage()

    def setup_learner(self):
        RLLearner = import_pipeline_agent(self.whole_cfg.env.name,
                                           self.whole_cfg.learner.get('pipeline', 'default'),
                                           'RLLearner')
        self.learner = RLLearner(deepcopy(self.whole_cfg))
        self.learner.launch(self.model_storage)

    def setup_actor(self):
        Actor = import_env_module(self.whole_cfg.env.name, 'Actor')
        self.actor = Actor(deepcopy(self.whole_cfg))
        self.actor.launch(model_storage=self.model_storage)

    def launch(self):
        self.setup_league()
        self.setup_replay_buffer()
        self.setup_model_storage()
        self.setup_learner()
        self.setup_actor()

    def run(self):
        try:
            self.learner.run()
        except Exception as e:
            import traceback
            print(e, flush=True)
            print(''.join(traceback.format_tb(e.__traceback__)), flush=True)
            self.close()

    def close(self):
        self.replay_buffer.close()

        time.sleep(1)


if __name__ == '__main__':
    args = get_args()
    cfg = read_config(args.config)
    trainer = Trainer(cfg)
    trainer.run()
