import os
import sys
import random
import torch
torch.set_num_threads(1)
import torch.nn.functional as F
import pickle
import time
import math
from easydict import EasyDict
from collections import OrderedDict
import numpy as np
import copy
from pygame.math import Vector2
import torch.multiprocessing as mp
sys.path.append('..')

from ding.envs import BaseEnv


def unit_id(unit_player, unit_team, ego_player, ego_team, team_size):
    unit_player, unit_team, ego_player, ego_team = int(unit_player) % team_size, int(unit_team), \
                                                   int(ego_player) % team_size, int(ego_team)
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


def _obs_transform(obs):
    map_width = 1000
    map_height = 1000
    team_num = 4
    player_num_per_team = 3
    with_speed = False
    
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
        left_margin, right_margin = left_top_x, map_width - right_bottom_x
        top_margin, bottom_margin = left_top_y, map_height - right_bottom_y
        # get scalar feat
        scalar_obs = np.array([rest_time / 1000, left_margin / 1000, right_margin / 1000, top_margin / 1000, bottom_margin / 1000])  # dim = 5

        # unit feat
        overlap = value['overlap']
        team_id, player_id = unit_id(n, value['team_name'], n, value['team_name'], player_num_per_team) #always [0,0]
        # load units
        fake_thorn = np.array([[center_x, center_y, 0]]) if not with_speed else np.array([[center_x, center_y, 0, 0, 0]])
        fake_clone = np.array([[center_x, center_y, 0, team_id, player_id]]) if not with_speed else np.array([[center_x, center_y, 0, 0, 0, team_id, player_id]])
        food = overlap['food'] + overlap['spore'] # dim is not certain
        thorn = np.array(overlap['thorns']) if len(overlap['thorns']) > 0 else fake_thorn
        clone = np.array([[*x[:-2], *unit_id(x[-2], x[-1], n, value['team_name'], player_num_per_team)] for x in overlap['clone']]) if len(overlap['clone']) > 0 else fake_clone

        overlap['spore'] = [x[:3] for x in overlap['spore']]
        overlap['thorns'] = [x[:3] for x in overlap['thorns']]
        overlap['clone'] = [[*x[:3], int(x[-2]), int(x[-1])] for x in overlap['clone']]
        # encode units
        food, food_relation = food_encode(clone, food, left_top_x, left_top_y, right_bottom_x, right_bottom_y, team_id, player_id)

        cl_ego = []
        for cl in clone:
            if (int(cl[-1])==team_id and int(cl[-2])==player_id):
                cl_ego += [cl]
        cl_ego = np.array(cl_ego)

        thorn_relation = relation_encode(clone, thorn) 
        clone_relation = relation_encode(cl_ego, clone)
        clone = clone_encode(clone, speed=with_speed) 

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
    # for i in range(team_num):
        # team_obs.append(team_obs_stack(obs[i * player_num_per_team: (i + 1) * player_num_per_team]))
    team_obs.append(team_obs_stack(obs[:]))
    return team_obs


def gobigger_collate(data):
    torch.set_num_threads(1)
    B, player_num_per_team = len(data), len(data[0]['scalar'])
    data = {k: sum([d[k] for d in data], []) for k in data[0].keys() if not k.startswith('collate_ignore')}
    clone_num = max([x.shape[0] for x in data['clone']])
    thorn_num = max([x.shape[1] for x in data['thorn_relation']])
    food_h = max([x.shape[1] for x in data['food']])
    food_w = max([x.shape[2] for x in data['food']])
    data['scalar'] = torch.stack([torch.as_tensor(x) for x in data['scalar']]).float() # [B*player_num_per_team,5]
    data['food'] = torch.stack([F.pad(torch.as_tensor(x), (0, food_w - x.shape[2], 0, food_h - x.shape[1])) for x in data['food']]).float() 
    data['food_relation'] = torch.stack([F.pad(torch.as_tensor(x), (0, 0, 0, clone_num - x.shape[0])) for x in data['food_relation']]).float()
    data['thorn_mask'] = torch.stack([torch.arange(thorn_num) < x.shape[1] for x in data['thorn_relation']]).float()
    data['thorn_relation'] = torch.stack([F.pad(torch.as_tensor(x), (0, 0, 0, thorn_num - x.shape[1], 0, clone_num - x.shape[0])) for x in data['thorn_relation']]).float()
    data['clone_mask'] = torch.stack([torch.arange(clone_num) < x.shape[0] for x in data['clone']]).float()
    data['clone'] = torch.stack([F.pad(torch.as_tensor(x), (0, 0, 0, clone_num - x.shape[0])) for x in data['clone']]).float()
    data['clone_relation'] = torch.stack([F.pad(torch.as_tensor(x), (0, 0, 0, clone_num - x.shape[1], 0, clone_num - x.shape[0])) for x in data['clone_relation']]).float()
    # data['batch'] = B
    # data['player_num_per_team'] = player_num_per_team
    return data


def cal_angle(x, y):
    dx1 = x
    dy1 = y
    dx2 = 1
    dy2 = 0
    angle1 = math.atan2(y, x)
    angle1 = int(angle1 * 180/math.pi)
    angle2 = math.atan2(0, 1)
    angle2 = int(angle2 * 180/math.pi)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
    if y < 0:
        included_angle = 360 - included_angle
    if included_angle == 360:
        included_angle = 0
    return included_angle


def _action_transform(actions, angle_split_num=4, action_type_num=4):
    ret = []
    split_angle = 360 / angle_split_num
    for action in actions:
        x, y, action_type = action
        if x is None:
            x = 0.0
        if y is None:
            y = 0.0
        d = Vector2(x, y)
        if d.length() != 0:
            d = d.normalize()
        x, y = d.x, d.y
        action_type = (action_type + 1) * angle_split_num
        direction = -1

        angle = cal_angle(x, y)
        direction = int(angle / split_angle)
        ret.append(direction + action_type)
        # if (direction + action_type) >= (angle_split_num * action_type_num):
        #     print(action); sys.stdout.flush()
    return ret


def _action_transform_back(actions, angle_split_num=4, action_type_num=4):
    ret = []
    split_angle = 360 / angle_split_num
    for action in actions:
        action_type = action // angle_split_num - 1
        direction = action % angle_split_num
        angle = direction * split_angle
        x = math.cos(angle / 180 * math.pi)
        y = math.sin(angle / 180 * math.pi)
        ret.append([x, y, action_type])
    return ret


def format_shape(obs):
    full_shape = SLDataLoader.default_obs_shape()
    for i in range(len(obs)):
        for key, value in obs[i].items():
            if key not in ['scalar', 'collate_ignore_raw_obs']:
                for j in range(len(value)):
                    s = obs[i][key][j].shape
                    padding = tuple()
                    for ss1, ss2 in zip(s, full_shape[key]):
                        padding += ((0, ss2 - ss1),)
                    obs[i][key][j] = np.pad(obs[i][key][j], padding)
                    # print(s, full_shape[key], obs[i][key][j].shape)
    return obs


def build_batch(data, player_num_per_team):
    ret = {k: torch.cat([d[k] for d in data], 0) for k in data[0].keys() if k not in ['batch', 'player_num_per_team']}
    ret['batch'] = len(data)
    ret['player_num_per_team'] = player_num_per_team
    return ret


def build_single_step_data(obs, label=None, angle_split_num=4, action_type_num=4):
    obs = _obs_transform(obs)
    obs = format_shape(obs)
    obs = gobigger_collate(obs)
    if label is not None:
        label = _action_transform(label, angle_split_num=angle_split_num,
                                  action_type_num=action_type_num)
    return obs, label


def assign_data(data, cuda=False, share_memory=False, device=None):
    assert cuda or share_memory
    if isinstance(data, torch.Tensor):
        if cuda:
            data = data.to(device)
        if share_memory:
            data.share_memory_()
    elif isinstance(data, dict):
        if cuda:
            data = {k: v.to(device) for k, v in data.items()}
        if share_memory:
            data = {k: v.share_memory_() for k, v in data.items()}
    return data


def get_fake_obs_data(team_num=4, player_num_per_team=3):
    full_shape = SLShareDataLoader.default_obs_shape()
    fake_obs_data = {}
    for k, v in full_shape.items():
        fake_obs_data[k] = torch.zeros([player_num_per_team] + v, dtype=torch.float)
    return fake_obs_data


def get_fake_label_data(player_num_per_team=3):
    fake_label_data = torch.zeros([player_num_per_team], dtype=torch.int)
    return fake_label_data


class SLDataLoader:

    @staticmethod
    def default_config():
        cfg = dict(
            team_num=4,
            player_num_per_team=3,
            batch_size=2,
            cache_size=4,
            train_data_prefix='PATH/replays',
            train_data_file='PATH/replays.txt.train',
            worker_num=2,
            angle_split_num=4,
            action_type_num=4,
            specific_agent_name='',
            specific_agent_rank=0,
            # epoches=1,
        )
        return EasyDict(cfg)

    @staticmethod
    def default_obs_shape():
        d = {
            'scalar': [5], # √ time + top + bottom + left + right
            'food': [2, 63, 63], # √ 2是固定的，63 = 1000/16 + 1
            'food_relation': [16*12, 150],  # √ 16*12=192=len(clone)，150固定的
            'thorn_relation': [16*12, 20, 12], # √ 16*12=192=len(clone)，thorns_max_num=20, dim=12
            'clone': [16*12, 17], # √ 16*12=192=len(clone)，dim=17
            'clone_relation': [16, 16*12, 12], # √ 16=max_clone_num，dim=17, 16*12=192=len(clone), dim=12
        }
        return EasyDict(d)

    def __init__(self, cfg):
        torch.set_num_threads(2)
        random.seed(233)
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.cache_size = self.cfg.cache_size
        self.train_data_prefix = self.cfg.train_data_prefix
        self.data_paths = []
        self.indexes = []
        data_index = 0
        with open(self.cfg.train_data_file, 'r') as f:
            for line in f.readlines():
                data_path = line.strip().split()[0]
                if data_path.endswith('data'):
                    self.data_paths.append(data_path)
                    self.indexes.append(data_index)
                    data_index += 1
        self.indexes = self.indexes * self.cfg.get('epoches', 100)
        random.shuffle(self.indexes)
        self.queue = mp.Queue()
        self.start = 0
        self.data_lock = mp.Lock()

        for i in range(self.cfg.worker_num):
            p = mp.Process(target=self.loop, daemon=True)
            p.start()
        # self.loop()

    def loop(self):
        while True:
            self.data_lock.acquire()
            data_path = self.data_paths[self.indexes[self.start]]
            self.start += 1
            self.data_lock.release()
            step_indexes = []

            with open(os.path.join(self.train_data_prefix, data_path.split('.')[0]+'.replay'), 'rb') as f:
                meta = pickle.load(f)
            agent_names = meta['agent_names']
            if str(self.cfg.specific_agent_name):
                if not str(self.cfg.specific_agent_name) in agent_names:
                    continue
                else:
                    agent_index = agent_names.index(str(self.cfg.specific_agent_name))
                    for i in range(3000):
                        step_indexes.append([i, agent_index])
            else:
                for i in range(3000):
                    for j in range(4):
                        step_indexes.append([i, j])

            random.shuffle(step_indexes)

            with open(os.path.join(self.train_data_prefix, data_path), 'rb') as f:
                data = pickle.load(f)
            observations = data['observations']
            actions = data['actions']

            for (content_index, player_index) in step_indexes:
                obs = [observations[content_index][0], {}]
                for i in range(self.cfg.player_num_per_team):
                    obs[1][str(player_index*self.cfg.player_num_per_team+i)] = \
                        observations[content_index][1][str(player_index*self.cfg.player_num_per_team+i)]
                label = []
                for i in range(self.cfg.player_num_per_team):
                    label.append(actions[content_index][str(player_index*self.cfg.player_num_per_team+i)])

                obs = _obs_transform(obs)
                obs = format_shape(obs)
                obs = gobigger_collate(obs)
                label = _action_transform(label, angle_split_num=self.cfg.angle_split_num,
                                          action_type_num=self.cfg.action_type_num)
                # obs, label = build_single_step_data(obs, label, self.cfg.angle_split_num, self.cfg.action_type_num)

                while True:
                    if self.queue.qsize() < self.cache_size:
                        self.queue.put([obs, label, data_path, content_index, player_index])
                        break
                    else:
                        time.sleep(0.01)

    def __iter__(self):
        return self

    def __len__(self):
        return 3000 * 4 * len(self.data_paths)

    def __next__(self):
        # t1 = time.time()
        while True:
            if self.queue.qsize() > self.batch_size:
                # t2 = time.time()
                batch_data = []
                labels = []
                data_paths = []
                content_indexes = []
                player_indexes = []
                for _ in range(self.batch_size):
                    obs, label, data_path, content_index, player_index = self.queue.get()
                    batch_data.append(obs)
                    labels.append(label)
                    data_paths.append(data_path)
                    content_indexes.append(content_index)
                    player_indexes.append(player_index)
                # t3 = time.time()
                batch_data = build_batch(batch_data, self.cfg.player_num_per_team)
                # t4 = time.time()
                # print('total: {}, {} / {}'.format(t4-t1, t3-t2, t4-t3))
                return batch_data, labels, data_paths, content_indexes, player_indexes
            else:
                time.sleep(1)


class SLShareDataLoader:

    @staticmethod
    def default_config():
        cfg = dict(
            team_num=4,
            player_num_per_team=3,
            batch_size=40,
            cache_size=120,
            train_data_prefix='PATH/replays',
            train_data_file='PATH/replays.txt.train',
            worker_num=1,
            angle_split_num=4,
            action_type_num=4,
            specific_agent_name='',
            specific_agent_rank=0,
            epoches=1,
        )
        return EasyDict(cfg)

    @staticmethod
    def default_obs_shape():
        d = {
            'scalar': [5], # √ time + top + bottom + left + right
            'food': [2, 63, 63], # √ 2是固定的，63 = 1000/16 + 1
            'food_relation': [16*12, 150],  # √ 16*12=192=len(clone)，150固定的
            'thorn_relation': [16*12, 20, 12], # √ 16*12=192=len(clone)，thorns_max_num=20, dim=12
            'clone': [16*12, 17], # √ 16*12=192=len(clone)，dim=17
            'clone_relation': [16*12, 16*12, 12], # √ 16=max_clone_num，dim=17, 16*12=192=len(clone), dim=12
            'thorn_mask': [20],
            'clone_mask': [192],
        }
        return EasyDict(d)

    def __init__(self, cfg):
        torch.set_num_threads(2)
        random.seed(233)
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.cache_size = self.cfg.cache_size
        self.train_data_prefix = self.cfg.train_data_prefix
        self.data_paths = []
        self.indexes = []
        data_index = 0
        with open(self.cfg.train_data_file, 'r') as f:
            for line in f.readlines():
                data_path = line.strip().split()[0]
                if data_path.endswith('data'):
                    self.data_paths.append(data_path)
                    self.indexes.append(data_index)
                    data_index += 1
        self.indexes = self.indexes * self.cfg.get('epoches', 100)
        random.shuffle(self.indexes)
        self.idle_queue = mp.Manager().Queue()
        self.ready_queue = mp.Manager().Queue()
        self.cache_obs = [assign_data(get_fake_obs_data(self.cfg.team_num, self.cfg.player_num_per_team), share_memory=True) \
                          for i in range(self.cfg.cache_size)]
        self.cache_label = [assign_data(get_fake_label_data(self.cfg.player_num_per_team), share_memory=True) \
                            for i in range(self.cfg.cache_size)]
        self.last_idle_indexes = []
        self.start = mp.Manager().Value('i', 0)
        self.record = mp.Manager().dict()

        for i in range(self.cfg.cache_size):
            self.idle_queue.put(i)

        for i in range(self.cfg.worker_num):
            p = mp.Process(target=self.loop, daemon=True)
            p.start()

        # print('start loop')
        # self.loop()

    def loop(self):
        while True:
            # print('[loop] get {}'.format(self.start)); sys.stdout.flush()
            data_path_index = self.start.value
            data_path = self.data_paths[self.indexes[data_path_index]]
            self.start.value += 1
            step_indexes = []

            with open(os.path.join(self.train_data_prefix, data_path.split('.')[0]+'.replay'), 'rb') as f:
                meta = pickle.load(f)
            agent_names = meta['agent_names']
            leaderboard = meta['leaderboard']
            leaderboard = sorted(leaderboard.items(), key=lambda x: x[1], reverse=True)
            leaderboard = [int(x) for x, y in leaderboard]

            if self.cfg.specific_agent_name is not None and self.cfg.specific_agent_name != '' and self.cfg.specific_agent_rank != -1:
                if not str(self.cfg.specific_agent_name) in agent_names:
                    continue
                if self.cfg.specific_agent_rank < 0 or self.cfg.specific_agent_rank >= self.cfg.team_num:
                    continue
                agent_index = agent_names.index(str(self.cfg.specific_agent_name))
                if leaderboard.index(agent_index) == self.cfg.specific_agent_rank:
                    for i in range(3000):
                        step_indexes.append([i, agent_index])
                else:
                    continue
            elif self.cfg.specific_agent_name is not None and self.cfg.specific_agent_name != '':
                if not str(self.cfg.specific_agent_name) in agent_names:
                    continue
                else:
                    agent_index = agent_names.index(str(self.cfg.specific_agent_name))
                    for i in range(3000):
                        step_indexes.append([i, agent_index])
            elif self.cfg.specific_agent_rank != -1:
                agent_index = leaderboard[self.cfg.specific_agent_rank]
                for i in range(3000):
                    step_indexes.append([i, agent_index])
            else:
                for i in range(3000):
                    for j in range(4):
                        step_indexes.append([i, j])

            random.shuffle(step_indexes)

            with open(os.path.join(self.train_data_prefix, data_path), 'rb') as f:
                data = pickle.load(f)
            observations = data['observations']
            actions = data['actions']

            for idx, (content_index, player_index) in enumerate(step_indexes):
                obs = [observations[content_index][0], {}]
                for i in range(self.cfg.player_num_per_team):
                    obs[1][str(player_index*self.cfg.player_num_per_team+i)] = \
                        observations[content_index][1][str(player_index*self.cfg.player_num_per_team+i)]
                label = []
                for i in range(self.cfg.player_num_per_team):
                    label.append(actions[content_index][str(player_index*self.cfg.player_num_per_team+i)])

                obs = _obs_transform(obs)
                obs = format_shape(obs)
                obs = gobigger_collate(obs)
                label = _action_transform(label, angle_split_num=self.cfg.angle_split_num,
                                          action_type_num=self.cfg.action_type_num)
                # obs, label = build_single_step_data(obs, label, self.cfg.angle_split_num, self.cfg.action_type_num)

                idle_index = self.idle_queue.get()
                for k, v in obs.items():
                    self.cache_obs[idle_index][k].copy_(v)
                self.cache_label[idle_index].copy_(torch.tensor(label))
                self.ready_queue.put(idle_index)
                self.record[idle_index] = data_path_index

    def __iter__(self):
        return self

    def __len__(self):
        return 3000 * 4 * len(self.data_paths)

    def __next__(self):
        while True:
            if len(self.last_idle_indexes) > 0:
                for idle_index in self.last_idle_indexes:
                    self.idle_queue.put(idle_index)
                self.last_idle_indexes = []
            if self.ready_queue.qsize() > self.batch_size:
                # t2 = time.time()
                batch_data = []
                labels = []
                for _ in range(self.batch_size):
                    ready_index = self.ready_queue.get()
                    batch_data.append(self.cache_obs[ready_index])
                    labels.append(self.cache_label[ready_index])
                    self.last_idle_indexes.append(ready_index)
                # t3 = time.time()
                batch_data = build_batch(batch_data, self.cfg.player_num_per_team)
                labels = torch.stack(labels).long()
                # t4 = time.time()
                # print('total: {}, {} / {}'.format(t4-t2, t3-t2, t4-t3))
                # print('{}'.format(self.record.values()))
                # print('{}'.format(self.last_idle_indexes))
                return batch_data, labels
            else:
                time.sleep(0.01)


if __name__ == '__main__':
    # loader = SLDataLoader(SLDataLoader.default_config())
    loader = SLShareDataLoader(SLShareDataLoader.default_config())
    while True:
        data = next(loader)
        import pdb; pdb.set_trace()