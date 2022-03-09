import os
import sys
import importlib
import logging
import time
import argparse
from uuid import uuid1
import pickle
import multiprocessing
import random

from gobigger.server import Server
from gobigger.render import RealtimeRender, RealtimePartialRender, EnvRender
from gobigger.agents import BotAgent

logging.basicConfig(level=logging.INFO)

"""
GoBigger 离线数据集 For SL
保存内容：
1. replay 文件，包含随机种子和每一帧的动作。依靠这个文件可以复现这局游戏
2. 更详细的 obs 和 action

每场对局都会保存replay文件（以.replay结尾）和 obs & action（以.data结尾）
.replay文件结构：是一个字典，包含以下字段
    seed: 对局随机数种子
    actions: 对局中每个动作帧所执行的动作
    agent_name: 参与对局的agent名称
    leaderboard: 本次对局最终排名和分数
.data文件结构：是一个字典，包含以下字段
    observation: 对局中每个动作帧获取到的obs，是最原始的obs
    actions: 对局中每个动作帧所执行的动作

使用方式：
python -u generate_data.py
"""

AVAILABLE_AGENTS = ['bot', 'bot', 'bot', 'bot']


class BotSubmission:

    def __init__(self, team_name, player_names):
        self.team_name = team_name
        self.player_names = player_names
        self.agents = {}
        for player_name in self.player_names:
            self.agents[player_name] = BotAgent(name=player_name)

    def get_actions(self, obs):
        global_state, player_states = obs
        actions = {}
        for player_name, agent in self.agents.items():
            action = agent.step(player_states[player_name])
            actions[player_name] = action
        return actions


class DataUtil:

    def __init__(self, agent_names, save_path_prefix):
        self.agent_names = agent_names
        self.save_path_prefix = save_path_prefix
        if not os.path.isdir(self.save_path_prefix):
            os.mkdir(self.save_path_prefix)
        if self.agent_names == '':
            self.agent_names = random.sample(AVAILABLE_AGENTS, 4)

    def launch_a_game(self, seed=None):
        data_simple = {'seed': None, 'actions': [], 'agent_names': self.agent_names}
        data_hard = {
            'observations': [], 
            'actions': []
        }
        if seed is None:
            t = str(time.time()).strip().split('.')
            seed = int(t[0]+t[1])
        data_simple['seed'] = seed
        server = Server(dict(
            team_num=4, # 队伍数量
            player_num_per_team=3, # 每个队伍的玩家数量
            match_time=60*10, # 每场比赛的持续时间
            obs_settings=dict(
                with_spatial=False,
            )
        ), seed)
        render = EnvRender(server.map_width, server.map_height)
        server.set_render(render)
        server.reset()
        team_player_names = server.get_team_names()
        team_names = list(team_player_names.keys())
        agents, teams_agents_dict = self.init_agents(team_names, team_player_names)
        for i in range(1000000):
            obs = server.obs()
            global_state, player_states = obs
            actions = {}
            for agent in agents:
                agent_obs = [global_state, {
                    player_name: player_states[player_name] for player_name in agent.player_names
                }]
                try:
                    actions.update(agent.get_actions(agent_obs))
                except:
                    fake_action = {
                        player_name: [None, None, -1] for player_name in agent.player_names
                    }
                    actions.update(fake_action)
            finish_flag = server.step(actions=actions)
            data_simple['actions'].append(actions)
            data_hard['observations'].append(obs)
            data_hard['actions'].append(actions)
            logging.debug('{} lastime={:.3f}, leaderboard={}'.format(i, server.last_time, global_state['leaderboard']))
            if finish_flag:
                data_simple['leaderboard'] = global_state['leaderboard']
                logging.debug('Game Over')
                break
        file_name = str(uuid1()) + "-" + str(seed)
        replay_path = os.path.join(self.save_path_prefix, file_name+'.replay')
        with open(replay_path, "wb") as f:
            pickle.dump(data_simple, f)
        data_path = os.path.join(self.save_path_prefix, file_name+'.data')
        with open(data_path, "wb") as f:
            pickle.dump(data_hard, f)
        logging.info('save as: {} {}'.format(replay_path, data_path))

    def init_agents(self, team_names, team_player_names):
        agents = []
        teams_agents_dict = {}
        for index, agent_name in enumerate(self.agent_names):
            agents.append(BotSubmission(team_name=team_names[index], 
                                        player_names=team_player_names[team_names[index]]))
            teams_agents_dict[team_names[index]] = agent_name
        return agents, teams_agents_dict


def generate_data(agent_names, save_path_prefix):
    data_util = DataUtil(agent_names, save_path_prefix)
    while True:
        data_util.launch_a_game()


def generate_data_multi(num_worker, agent_names, save_path_prefix):
    all_p = []
    for i in range(0, num_worker):
        try:
            p = multiprocessing.Process(target=generate_data, args=(agent_names, save_path_prefix,))
            p.start()
            all_p.append(p)
            time.sleep(1)
            time.sleep(random.random())
        except Exception as e:
            print('!!!!!!!!!!!!!!!! {} failed, because {} !!!!!!!!!!!!!!!!!'.format(i, str(e)))
            continue
    for p in all_p:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agent-names', type=str, default='')
    parser.add_argument('-s', '--save-path-prefix', type=str, default='replays')
    parser.add_argument('-n', '--num-worker', type=int, default=1)
    args = parser.parse_args()

    if args.agent_names != '':
        args.agent_names = args.agent_names.strip().split(',')
    generate_data_multi(args.num_worker, args.agent_names, args.save_path_prefix)
