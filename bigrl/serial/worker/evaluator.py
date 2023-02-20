import argparse
from random import random
from copy import deepcopy

import torch
import torch.multiprocessing as tm

from bigrl.core.utils import read_config
from gobigger.envs import GoBiggerEnv
from bigrl.serial.import_helper import import_pipeline_agent
import numpy as np
import os
from collections import defaultdict
from functools import partial
from easydict import EasyDict

def get_args():
    parser = argparse.ArgumentParser(description="rl_eval")
    parser.add_argument("--config", "-c", type=str, default='user_config.yaml', help='config_path')
    return parser.parse_args()

class BaseEvaluator:

    def __init__(self, cfg):
        self.whole_cfg = cfg
        self.pipeline = self.whole_cfg.agent.pipeline
        self.player_num = self.whole_cfg.env.player_num_per_team
        self.team_num = self.whole_cfg.env.team_num

        self.save_dir = self.whole_cfg.eval.save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.checkpoint_path = self.whole_cfg.eval.checkpoint_path
        self.repeat_time = self.whole_cfg.eval.repeat_time
        self.save_replay = self.whole_cfg.eval.save_replay

    def run(self):
        torch.set_num_threads(1)
        team_scores = []
        bot_team_scores = []
        player_state_dict = defaultdict(partial(defaultdict,list))
        for i in range(self.repeat_time):
            # save replay
            if self.save_replay:
                self.whole_cfg = self.get_save_replay_cfg(self.whole_cfg, self.save_dir)
            leaderboard, mean_players_stat = self.launch_game()
            print(leaderboard)
            team_scores.append(leaderboard[0])
            bot_team_scores.append(max(list(leaderboard.values())[1:])) # get max team score
            for p_, info in mean_players_stat.items():
                for k_, v_ in info.items():
                    player_state_dict[p_][k_].append(v_)
        result = {
            'team_score': np.mean(team_scores),
            'bot_score': np.mean(bot_team_scores),
            'player_state': player_state_dict,
        }
        torch.save(result, os.path.join(self.save_dir, "result.pth"))
        return None

    def launch_game(self):
        env = GoBiggerEnv(self.whole_cfg.env, step_mul=self.whole_cfg.env.step_mul)
        obs = env.reset()
        self.init_agents(self.whole_cfg)
        players_stat = defaultdict(partial(defaultdict,list))
        for i in range(self.whole_cfg.env.frame_limit // self.whole_cfg.env.step_mul):
            global_state, player_states = obs
            actions = {}
            for player_name, agent in self.agents.items():
                actions.update(agent.step(obs))
            obs, reward, done, info = env.step(actions=actions)
            if done:
                break
        # record info, only last step
        for idx, agent in self.agents.items():
            players_stat[agent.game_player_id]['score'].append(obs[1][agent.game_player_id]['score'])
            players_stat[agent.game_player_id]['team_score'].append(obs[0]['leaderboard'][agent.game_team_id])
        if hasattr(agent, 'stat'):
            for k, v in agent.stat.items():
                players_stat[agent.game_player_id][k].append(v)

        for k, v in info['eats'].items():
            game_player_id = self.agents[k].game_player_id
            for _k, _v in v.items():
                players_stat[game_player_id][_k].append(_v)

        mean_players_stat = defaultdict(partial(defaultdict,list))
        for p in players_stat:
            for _k in players_stat[p]:
                mean_players_stat[p][_k] = np.mean(players_stat[p][_k])

        return global_state['leaderboard'], mean_players_stat

    def init_agents(self, cfg):
        self.agents = {}
        self.agents_name = []
        for i in range(self.player_num):
            from bigrl.serial.import_helper import import_pipeline_agent
            Agent = import_pipeline_agent(self.whole_cfg.env.name, self.pipeline, 'Agent')
            cfg_cp = deepcopy(cfg)
            cfg_cp.agent.pipeline = self.pipeline
            cfg_cp.agent.player_id = i
            cfg_cp.agent.game_player_id = i
            cfg_cp.agent.game_team_id = i // self.player_num
            cfg_cp.agent.send_data = False
            agent = Agent(cfg_cp)
            agent.reset()
            agent.setup_model()
            agent.model.load_state_dict(torch.load(self.checkpoint_path, map_location='cpu')['model'], strict=False)
            self.agents[i] = agent
            self.agents_name.append('agent')

        for i in range(self.player_num, self.player_num * self.team_num):
            Agent = import_pipeline_agent(self.whole_cfg.env.name, 'bot', 'Agent')
            cfg_cp = deepcopy(cfg)
            cfg_cp.agent.game_player_id = i
            cfg_cp.agent.game_team_id = i // self.player_num
            cfg_cp.agent.send_data = False
            cfg_cp.agent.player_id = i
            agent = Agent(cfg_cp)
            agent.reset()
            self.agents[i] = agent
            self.agents_name.append('bot')
    
    def get_save_replay_cfg(self, cfg, save_dir):
        from datetime import datetime
        save_name_prefix = datetime.now().strftime("%Y-%m-%d-%H-%M")
        if 'playback_settings' not in cfg.env:
            cfg.env['playback_settings '] = EasyDict({})

        default_save_dir = os.path.join(save_dir, 'replays')
        playback_type = cfg.env.playback_settings.get('playback_type', 'by_frame')
        if playback_type == 'by_frame':
            cfg.env.playback_settings.by_frame.save_frame = True
            replay_dir = cfg.env.playback_settings.by_frame.get('save_dir', default_save_dir)
            cfg.env.playback_settings.by_frame.save_dir = replay_dir
            cfg.env.playback_settings.by_frame.save_name_prefix = save_name_prefix

        elif playback_type == 'by_video':
            cfg.env.playback_settings.by_video.save_video = True
            replay_dir = cfg.env.playback_settings.by_video.get('save_dir', default_save_dir)
            cfg.env.playback_settings.by_video.save_dir = replay_dir
            cfg.env.playback_settings.by_video.save_name_prefix = save_name_prefix
        return cfg
            

if __name__ == '__main__':
    args = get_args()
    cfg = read_config(args.config)
    evaluaor = BaseEvaluator(cfg)
    evaluaor.run()