from cmath import pi
from typing import Any, List, Union, Optional, Tuple
import time
import copy
import math
from collections import OrderedDict
import cv2
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.torch_utils import to_tensor, to_ndarray, to_list
from ding.utils import ENV_REGISTRY
from gobigger.server import Server
from gobigger.render import EnvRender
from .gobigger_env import GoBiggerEnv
import os
import shutil

@ENV_REGISTRY.register('gobigger_explore',force_overwrite=True)
class GoBiggerExploreEnv(GoBiggerEnv):
    '''
    feature:
        - old unit id setting, self team's team id is always 0, other team's team ids are rearranged to 1, 2, ... , team_num - 1, self player's id is always 0.
        - old reward setting, which is defined as clip(new team size - old team size, -1, 1)
    '''
    config = dict(
        # env setting
        player_num_per_team=3,
        team_num=4,
        match_time=10*60,
        map_height=1000,
        map_width=1000,
        spatial=False,
        speed = False,
        all_vision = False,
        # whther train
        train=True,
        # obs setting
        reorder_team=True,
        reorder_player=True,
        # save setting
        frame_resume=False,
        frame_path='./frame',
        frame_period=10,
        frame_cfg={'type':'dist'},
    )
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._player_num_per_team = cfg.player_num_per_team
        self._team_num = cfg.team_num
        self._player_num = self._player_num_per_team * self._team_num
        self._match_time = cfg.match_time
        self._map_height = cfg.map_height
        self._map_width = cfg.map_width
        self._spatial = cfg.spatial
        self._train = cfg.train
        self._last_team_size = None
        self._init_flag = False
        self._speed = cfg.speed
        self._all_vision = cfg.all_vision
        self._cfg['obs_settings'] = dict(
                with_spatial=self._spatial,
                with_speed=self._speed,
                with_all_vision=self._all_vision)
        self._train = cfg.train

        self._observation_space = None
        self._action_space = None
        self.reward_space = None

        self._reorder_team = cfg.reorder_team
        self._reorder_player = cfg.reorder_player

        self._frame_resume = cfg.get('frame_resume', False) if self._train else False
        if self._frame_resume:
            self._env_id = cfg.get('env_id', 0)
            self._frame_path = cfg.get('frame_path', './frame')
            self._frame_cfg = cfg.get('frame_cfg', {'type':'dist'})
            self._frame_period = cfg.get('frame_period', 50)
            self._frame_id = 0
            if os.path.exists(self._frame_path):
                try:
                    shutil.rmtree(self._frame_path)
                except:
                    pass
            try:
                os.mkdir(self._frame_path)
            except:
                pass
            self.init_frame()
    
    def init_frame(self):
        if self._frame_cfg['type'] == 'dist':
            self._frame_buffer = None
            self._frame_freq = {'0_0_0_0':0}
            self._frame_num = {'0_0_0_0':1}

    def load_frame(self):
        if self._frame_cfg['type'] == 'dist':
            frames = {'0_0_0_0':['']}
            min_key = '0_0_0_0'
            min_freq = self._frame_freq[min_key]
            for f in os.listdir(self._frame_path):
                if '.pkl' in f:
                    k_1, k_2, k_3, k_4 = f.split('.')[0].split('_')[1:5]
                    key = '{}_{}_{}_{}'.format(k_1, k_2, k_3, k_4)
                    self._frame_freq[key] = self._frame_freq.get(key, 0)
                    frames[key] = frames.get(key, [])
                    frames[key].append(os.path.join(self._frame_path, f))
            for k in frames.keys():
                freq = self._frame_freq[k]
                if freq < min_freq:
                    min_freq = freq
                    min_key = k
            self._frame_freq[min_key] += 1
            self._cfg['jump_to_frame_file'] = np.random.choice(frames[min_key])

    def save_frame(self):
        frame = ''
        if self._frame_buffer is not None:
            frame = os.path.join(self._frame_path, '{}_{}.pkl'.format(self._env_id, self._frame_buffer))
            self._frame_buffer = None
        return frame

    def pre_save_frame(self, obs):
        self._frame_id = (self._frame_id + 1) % self._frame_period
        if self._frame_id % self._frame_period == 0:
            if self._frame_cfg['type'] == 'dist':
                ally_dist = 4
                enemey_dist = 4
                eaten = 3
                dirt = 3
                # parse state
                global_state, player_state = obs
                clone = player_state['0']['overlap']['clone']
                # load ego, ally and enemy clones
                ego_clone = []
                ally_clone = []
                enemey_clone = []
                for c in clone:
                    if int(c[-2]) == 0:
                        ego_clone.append(c[:3])
                    elif int(c[-1]) == 0:
                        ally_clone.append(c[:3])
                    else:
                        enemey_clone.append(c[:3])
                ego_clone = np.array(ego_clone)
                ally_clone = np.array(ally_clone)
                enemey_clone = np.array(enemey_clone)

                # get weight
                if ego_clone.shape[0] > 0:
                    if enemey_clone.shape[0] > 0:
                        delta_size = set_weight(ego_clone, ally_clone, z=enemey_clone)
                        if delta_size >0:
                            eaten = 1
                        else:
                            eaten = 2

                # get angle
                if ego_clone.shape[0]>0 and ally_clone.shape[0]>0 and enemey_clone.shape[0]>0:
                    ego_x = sorted(ego_clone, key=lambda s: s[2], reverse=True)
                    ally_y = sorted(ally_clone, key=lambda s: s[2], reverse=True)
                    enemey_z = sorted(enemey_clone, key=lambda s: s[2], reverse=True)
                    ego_x = ego_x[0][:2]
                    ally_y = ally_y[0][:2]
                    enemey_z = enemey_z[0][:2]
                    angle = set_angle(ego_x,ally_y,enemey_z)
                    # if angle >= 0.5: # 0~60
                    #     dirt = 1
                    # elif angle<0.5 and angle>=-0.5: # 60~120
                    #     dirt = 2
                    # else: # 120~180
                    #     dirt = 3
                    if angle >= 0:
                        dirt = 1
                    else:
                        dirt = 2

                # get distance
                if ego_clone.shape[0] > 0:
                    if ally_clone.shape[0] > 0:
                        dist = set_dist(ego_clone, ally_clone)
                        if dist <= 25:
                            ally_dist = 1
                        elif dist <= 50:
                            ally_dist = 2
                        else:
                            ally_dist = 3
                    if enemey_clone.shape[0] > 0:
                        dist = set_dist(ego_clone, enemey_clone)
                        if dist <= 25:
                            enemey_dist = 1
                        elif dist <= 50:
                            enemey_dist = 2
                        else:
                            enemey_dist = 3
                # get frame buffer
                key = '{}_{}_{}_{}'.format(ally_dist, enemey_dist, dirt, eaten)
                self._frame_num[key] = self._frame_num.get(key, 0)
                self._frame_buffer = '{}_{}'.format(key, self._frame_num[key])
                self._frame_num[key] = (self._frame_num[key] + 1) % 50

    def reset(self) -> np.ndarray:
        if self._frame_resume:
            self.load_frame()
        if not self._init_flag or self._frame_resume:
            self._env = self._launch_game()
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            # self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            pass
            # self._env.seed(self._seed)
        self._env.reset()
        self._final_eval_reward = [0. for _ in range(self._team_num)]
        raw_obs = self._env.obs()
        obs = self._obs_transform(raw_obs)
        rew = self._get_reward(raw_obs)
        if 'jump_to_frame_file' in self._cfg.keys() and self._cfg['jump_to_frame_file'] != '':
            print('reset: leaderboard={}'.format(self._last_team_size))
        return obs

    def step(self, action: list) -> BaseEnvTimestep:
        action = self._act_transform(action)
        if self._frame_resume:
            done = self._env.step(actions=action, save_frame_full_path=self.save_frame())
        else:
            done = self._env.step(actions=action)
        raw_obs = self._env.obs()
        if self._frame_resume:
            self.pre_save_frame(raw_obs)
        obs = self._obs_transform(raw_obs)
        rew = self._get_reward(raw_obs)
        info = [{} for _ in range(self._team_num)]

        for i, team_reward in enumerate(rew):
            self._final_eval_reward[i] += team_reward
        if done:
            for i in range(self._team_num):
                info[i]['final_eval_reward'] = self._final_eval_reward[i]
                info[i]['leaderboard'] = self._last_team_size
            leaderboard = self._last_team_size
            leaderboard_sorted = sorted(leaderboard.items(), key=lambda x: x[1], reverse=True)
            win_rate = self.win_rate(leaderboard_sorted)
            print('win_rate:{:.3f}, leaderboard_sorted:{}'.format(win_rate, leaderboard_sorted))
        return BaseEnvTimestep(obs, rew, done, info)

    def _unit_id(self, unit_player, unit_team, ego_player, ego_team, team_size):
        return unit_id(unit_player, unit_team, ego_player, ego_team, team_size)

    def _obs_transform(self, obs: tuple) -> list:
        global_state, player_state = obs
        player_state = OrderedDict(player_state)
        # global
        # 'border': [map_width, map_height] fixed map size
        total_time = global_state['total_time']
        last_time = global_state['last_time']
        rest_time = total_time - last_time

        # player
        obs = []
        for n, value in player_state.items():
            # scalar feat
            # get margin
            left_top_x, left_top_y, right_bottom_x, right_bottom_y = value['rectangle']
            center_x, center_y = (left_top_x + right_bottom_x) / 2, (left_top_y + right_bottom_y) / 2
            left_margin, right_margin = left_top_x, self._map_width - right_bottom_x
            top_margin, bottom_margin = left_top_y, self._map_height - right_bottom_y
            # get scalar feat
            scalar_obs = np.array([rest_time / 1000, left_margin / 1000, right_margin / 1000, top_margin / 1000, bottom_margin / 1000])  # dim = 5

            # unit feat
            overlap = value['overlap']
            team_id, player_id = self._unit_id(n, value['team_name'], n, value['team_name'], self._player_num_per_team) #always [0,0]
            # load units
            fake_thorn = np.array([[center_x, center_y, 0]]) if not self._speed else np.array([[center_x, center_y, 0, 0, 0]])
            fake_clone = np.array([[center_x, center_y, 0, team_id, player_id]]) if not self._speed else np.array([[center_x, center_y, 0, 0, 0, team_id, player_id]])
            food = overlap['food'] + overlap['spore'] # dim is not certain
            thorn = np.array(overlap['thorns']) if len(overlap['thorns']) > 0 else fake_thorn
            clone = np.array([[*x[:-2], *self._unit_id(x[-2], x[-1], n, value['team_name'], self._player_num_per_team)] for x in overlap['clone']]) if len(overlap['clone']) > 0 else fake_clone

            overlap['spore'] = [x[:3] for x in overlap['spore']]
            overlap['thorns'] = [x[:3] for x in overlap['thorns']]
            overlap['clone'] = [[*x[:3], int(x[-2]), int(x[-1])] for x in overlap['clone']]
            # encode units
            food, food_relation = food_encode(clone, food, left_top_x, left_top_y, right_bottom_x, right_bottom_y, team_id, player_id)
            
            cl_ego = np.where((clone[:,-1]==team_id) & (clone[:,-2]==player_id))
            cl_ego = clone[cl_ego]

            cl_other = np.where((clone[:,-1]!=team_id) | (clone[:,-2]!=player_id))
            cl_other = clone[cl_other]
            if cl_other.size == 0:
                cl_other = np.array([[center_x, center_y, 0, team_id+1, player_id]]) if not self._speed else np.array([[center_x, center_y, 0, 0, 0, team_id+1, player_id]])

            thorn_relation = relation_encode(clone, thorn)
            clone_relation = relation_encode(cl_ego, cl_other)
            clone = clone_encode(clone, speed=self._speed)

            player_obs = {
                'scalar': scalar_obs.astype(np.float32),
                'food': food.astype(np.float32),
                'food_relation': food_relation.astype(np.float32),
                'thorn_relation': thorn_relation.astype(np.float32),
                'clone': clone.astype(np.float32),
                'clone_relation': clone_relation.astype(np.float32),
                'collate_ignore_raw_obs': {'overlap': overlap},
            }
            obs.append(player_obs)

        team_obs = []
        for i in range(self._team_num):
            team_obs.append(team_obs_stack(obs[i * self._player_num_per_team: (i + 1) * self._player_num_per_team]))
        return team_obs
    
    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=self._player_num,
            obs_space=T(
                {
                    'scalar': (5, ),
                    'food': (2, ),
                },
                {
                    'min': 0,
                    'max': 1,
                    'dtype': np.float32,
                },
            ),
            # [min, max)
            act_space=T(
                (1, ),
                {
                    'min': 0,
                    'max': 16,
                    'dtype': int,
                },
            ),
            rew_space=T(
                (1, ),
                {
                    'min': -1000.0,
                    'max': 1000.0,
                    'dtype': np.float32,
                },
            ),
            use_wrappers=None,
        )



def unit_id(unit_player, unit_team, ego_player, ego_team, team_size):
    unit_player, unit_team, ego_player, ego_team = int(unit_player) % team_size, int(unit_team), int(ego_player) % team_size, int(ego_team)
    # The ego team's id is always 0, enemey teams' ids are 1,2,...,team_num-1
    # The ego player's id is always 0, allies' ids are 1,2,...,player_num_per_team-1
    if unit_team != ego_team:
        player_id = unit_player
        team_id = unit_team if unit_team > ego_team else unit_team + 1
    else:
        if unit_player != ego_player:
            player_id = unit_player if unit_player > ego_player else unit_player + 1
        else:
            player_id = 0
        team_id = 0

    return [team_id, player_id]

def food_encode(clone, food, left_top_x, left_top_y, right_bottom_x, right_bottom_y, team_id, player_id):
    w = (right_bottom_x - left_top_x) // 16 + 1
    h = (right_bottom_y - left_top_y) // 16 + 1
    food_map = np.zeros((2, h, w))

    w_ = (right_bottom_x - left_top_x) // 8 + 1
    h_ = (right_bottom_y - left_top_y) // 8 + 1
    food_grid = [ [ [] for j in range(w_) ] for i in range(h_) ]
    food_relation = np.zeros((len(clone), 7 * 7 + 1, 3))

    # food_map[0,:,:] represent food density map
    # food_map[1,:,:] represent cur clone ball density map
    # food_frid[:] represent food information(x,y,r)
    # food_relation represent food and cloen in 7*7 grid (offset_x, offset_y, r) 

    for p in food:
        x = min(max(p[0], left_top_x), right_bottom_x) - left_top_x
        y = min(max(p[1], left_top_y), right_bottom_y) - left_top_y
        radius = p[2]
        # encode food density map
        i, j = int(y // 16), int(x // 16)
        food_map[0, i, j] += radius * radius
        # encode food fine grid
        i, j = int(y // 8), int(x // 8)
        food_grid[i][j].append([(x - 8 * j) / 8, (y - 8 * i) / 8, radius])

    for c_id, p in enumerate(clone):
        x = min(max(p[0], left_top_x), right_bottom_x) - left_top_x
        y = min(max(p[1], left_top_y), right_bottom_y) - left_top_y
        radius = p[2]
        # encode food density map
        i, j = int(y // 16), int(x // 16)
        if int(p[3]) == team_id and int(p[4]) == player_id:
            food_map[1, i, j] += radius * radius
        # encode food fine grid
        i, j = int(y // 8), int(x // 8)
        t, b, l, r = max(i - 3, 0), min(i + 4, h_), max(j - 3, 0), min(j + 4, w_)
        for ii in range(t, b):
            for jj in range(l, r):
                for f in food_grid[ii][jj]:
                    food_relation[c_id][(ii - t) * 7 + jj - l][0] = f[0]
                    food_relation[c_id][(ii - t) * 7 + jj - l][1] = f[1]
                    food_relation[c_id][(ii - t) * 7 + jj - l][2] += f[2] * f[2]

        food_relation[c_id][-1][0] = (x - j * 8) / 8
        food_relation[c_id][-1][1] = (y - i * 8) / 8
        food_relation[c_id][-1][2] = radius / 10

    food_map[0, :, :] = np.sqrt(food_map[0, :, :]) / 2    #food
    food_map[1, :, :] = np.sqrt(food_map[1, :, :]) / 10   #cur clone
    food_relation[:, :-1, 2] = np.sqrt(food_relation[:, :-1, 2]) / 2
    food_relation = food_relation.reshape(len(clone), -1)
    return food_map, food_relation

def clone_encode(clone, speed=False):
    pos = clone[:, :2] / 100
    rds = clone[:, 2:3] / 10
    ids = np.zeros((len(clone), 12))
    ids[np.arange(len(clone)), (clone[:, -2] * 3 + clone[:, -1]).astype(np.int64)] = 1.0
    split = (clone[:, 2:3] - 10) / 10

    eject = (clone[:, 2:3] - 10) / 10
    if not speed:
        clone = np.concatenate([pos, rds, ids, split, eject], axis=1)  # dim = 17
    else:
        spd = clone[:, 3:5] / 60
        clone = np.concatenate([pos, rds, ids, split, eject, spd], axis=1)
    return clone

def relation_encode(point_1, point_2):
    pos_rlt_1 = point_2[None,:,:2] - point_1[:,None,:2] # relative position
    pos_rlt_2 = np.linalg.norm(pos_rlt_1, ord=2, axis=2, keepdims=True) # distance
    pos_rlt_3 = point_1[:,None,2:3] - pos_rlt_2 # whether source collides with target
    pos_rlt_4 = point_2[None,:,2:3] - pos_rlt_2 # whether target collides with source
    pos_rlt_5 = (2 + np.sqrt(0.5)) * point_1[:,None,2:3] - pos_rlt_2 # whether source's split collides with target
    pos_rlt_6 = (2 + np.sqrt(0.5)) * point_2[None,:,2:3] - pos_rlt_2 # whether target's split collides with source
    rds_rlt_1 = point_1[:,None,2:3] - point_2[None,:,2:3] # whether source can eat target
    rds_rlt_2 = np.sqrt(0.5) * point_1[:,None,2:3] - point_2[None,:,2:3] # whether source's split can eat target
    rds_rlt_3 = np.sqrt(0.5) * point_2[None,:,2:3] - point_1[:,None,2:3] # whether target's split can eat source
    rds_rlt_4 = point_1[:,None,2:3].repeat(len(point_2), axis=1) # source radius
    rds_rlt_5 = point_2[None,:,2:3].repeat(len(point_1), axis=0) # target radius
    relation = np.concatenate([pos_rlt_1 / 100, pos_rlt_2 / 100, pos_rlt_3 / 100, pos_rlt_4 / 100, pos_rlt_5 / 100, pos_rlt_6 / 100, rds_rlt_1 / 10, rds_rlt_2 / 10, rds_rlt_3 / 10, rds_rlt_4 / 10, rds_rlt_5 / 10], axis=2)
    return relation

def team_obs_stack(team_obs):
    result = {}
    for k in team_obs[0].keys():
        result[k] = [o[k] for o in team_obs]
    return result

def set_dist(x, y):
    # x:[m,>=2] y:[n,>=2]
    dst = np.linalg.norm(x[:,None,:2] - y[None,:,:2], ord=2, axis=2, keepdims=False) # [m,n]
    return np.minimum(dst - x[:,None,2], dst - y[None,:,2]).min()

def set_weight(*args, z):
    # r = x[], size = x*x
    size_sum = 0 
    for clone in args:
        if clone.shape[0] == 0: continue
        clone_size = np.sum(np.power(clone[:,2],2),axis=0)
        size_sum += clone_size
    size_z = np.sum(np.power(z[:,2],2),axis=0)
    delta_size = size_sum - size_z
    return delta_size

def set_angle(x, y, z):
    # x is ego, y is ally, z is enemy
    dst_x_y = np.linalg.norm(x-y, ord=None, axis=None, keepdims=False)
    dst_y_z = np.linalg.norm(y-z, ord=None, axis=None, keepdims=False)
    dst_x_z = np.linalg.norm(x-z, ord=None, axis=None, keepdims=False)

    angle_x = (dst_x_y * dst_x_y + dst_x_z * dst_x_z - dst_y_z * dst_y_z) / (2 * dst_x_y * dst_x_z)
    angle_y = (dst_x_y * dst_x_y + dst_y_z * dst_y_z - dst_x_z * dst_x_z) / (2 * dst_x_y * dst_y_z)
    angle_z = (dst_y_z * dst_y_z + dst_x_z * dst_x_z - dst_x_y * dst_x_y) / (2 * dst_y_z * dst_x_z)
    #assert np.arccos(angle_x) + np.arccos(angle_y) + np.arccos(angle_z) == np.pi
    return angle_z