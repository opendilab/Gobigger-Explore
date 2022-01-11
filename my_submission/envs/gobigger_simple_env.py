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


@ENV_REGISTRY.register('gobigger_simple',force_overwrite=True)
class GoBiggerSimpleEnv(GoBiggerEnv):
    '''
    feature:
        - old unit id setting, self team's team id is always 0, other team's team ids are rearranged to 1, 2, ... , team_num - 1, self player's id is always 0.
        - old reward setting, which is defined as clip(new team size - old team size, -1, 1)
    '''
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

    def _unit_id(self, unit_player, unit_team, ego_player, ego_team, team_size):
        return unit_id(unit_player, unit_team, ego_player, ego_team, team_size)
    

    def _obs_transform_eval(self, obs: tuple) -> list:
        player_bot_obs = copy.deepcopy(obs)
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

            thorn_relation = relation_encode(cl_ego, thorn)
            clone_relation = relation_encode(cl_ego, cl_other)
            clone = clone_encode(cl_ego, speed=self._speed)

            player_obs = {
                'scalar': scalar_obs.astype(np.float32),
                'food': food.astype(np.float32),
                'food_relation': food_relation.astype(np.float32),
                'thorn_relation': thorn_relation.astype(np.float32),
                'clone': clone.astype(np.float32),
                'clone_relation': clone_relation.astype(np.float32),
                #'collate_ignore_raw_obs': {'overlap': overlap,'player_bot_obs':player_bot_obs},
                'collate_ignore_raw_obs': {'overlap': overlap},
            }
            obs.append(player_obs)

        team_obs = []
        team_obs.append(team_obs_stack(obs[:self._player_num_per_team]))
        return team_obs 

    def _obs_transform(self, obs: tuple) -> list:
        player_bot_obs = copy.deepcopy(obs)
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

            thorn_relation = relation_encode(cl_ego, thorn)
            clone_relation = relation_encode(cl_ego, cl_other)
            clone = clone_encode(cl_ego, speed=self._speed)

            player_obs = {
                'scalar': scalar_obs.astype(np.float32),
                'food': food.astype(np.float32),
                'food_relation': food_relation.astype(np.float32),
                'thorn_relation': thorn_relation.astype(np.float32),
                'clone': clone.astype(np.float32),
                'clone_relation': clone_relation.astype(np.float32),
                #'collate_ignore_raw_obs': {'overlap': overlap,'player_bot_obs':player_bot_obs},
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



