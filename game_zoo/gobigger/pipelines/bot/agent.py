import copy
import math
import queue
import random

from pygame.math import Vector2


class Agent:
    HAS_MODEL = False

    def __init__(self, cfg, ):
        self.whole_cfg = cfg
        self.player_num = self.whole_cfg.env.player_num_per_team
        self.team_num = self.whole_cfg.env.team_num
        self.game_player_id = self.whole_cfg.agent.game_player_id  # start from 0
        self.game_team_id = self.game_player_id // self.player_num # start from 0
        self.send_data = self.whole_cfg.agent.send_data
        self.player_id = self.whole_cfg.agent.player_id
        self.actions_queue = queue.Queue()

    def reset(self):
        pass

    def eval_postprocess(self, *args, **kwargs):
        return None

    def collect_data(self, *args, **kwargs):
        return None
    
    def step(self, obs):
        obs = obs[1][self.game_player_id]
        if self.actions_queue.qsize() > 0:
            return {self.game_player_id: self.actions_queue.get()}
        overlap = obs['overlap']
        overlap = self.preprocess(overlap)
        food_balls = overlap['food']
        thorns_balls = overlap['thorns']
        spore_balls = overlap['spore']
        clone_balls = overlap['clone']

        my_clone_balls, others_clone_balls = self.process_clone_balls(clone_balls)

        if len(my_clone_balls) >= 9 and my_clone_balls[4]['radius'] > 4:
            self.actions_queue.put([0, 0, 0])
            self.actions_queue.put([0, 0, 0])
            self.actions_queue.put([0, 0, 0])
            self.actions_queue.put([0, 0, 0])
            self.actions_queue.put([0, 0, 0])
            self.actions_queue.put([0, 0, 0])
            self.actions_queue.put([0, 0, 0])
            self.actions_queue.put([None, None, 1])
            self.actions_queue.put([None, None, 1])
            self.actions_queue.put([None, None, 1])
            self.actions_queue.put([None, None, 1])
            self.actions_queue.put([None, None, 1])
            self.actions_queue.put([None, None, 1])
            self.actions_queue.put([None, None, 1])
            self.actions_queue.put([None, None, 1])
            action_ret = self.actions_queue.get()
            return {self.game_player_id: action_ret}

        if len(others_clone_balls) > 0 and self.can_eat(others_clone_balls[0]['radius'], my_clone_balls[0]['radius']):
            direction = (my_clone_balls[0]['position'] - others_clone_balls[0]['position'])
            action_type = 0
        else:
            min_distance, min_thorns_ball = self.process_thorns_balls(thorns_balls, my_clone_balls[0])
            if min_thorns_ball is not None:
                direction = (min_thorns_ball['position'] - my_clone_balls[0]['position'])
            else:
                min_distance, min_food_ball = self.process_food_balls(food_balls, my_clone_balls[0])
                if min_food_ball is not None:
                    direction = (min_food_ball['position'] - my_clone_balls[0]['position'])
                else:
                    direction = (Vector2(0, 0) - my_clone_balls[0]['position'])
            action_random = random.random()
            if action_random < 0.02:
                action_type = 1
            if action_random < 0.04 and action_random > 0.02:
                action_type = 2
            else:
                action_type = 0
        if direction.length()>0:
            direction = direction.normalize()
        else:
            direction = Vector2(1, 1).normalize()
        direction = self.add_noise_to_direction(direction).normalize()
        self.actions_queue.put([direction.x, direction.y, action_type])
        action_ret = self.actions_queue.get()
        return {self.game_player_id: action_ret}

    def process_clone_balls(self, clone_balls):
        my_clone_balls = []
        others_clone_balls = []
        for clone_ball in clone_balls:
            if clone_ball['player'] == self.game_player_id:
                my_clone_balls.append(copy.deepcopy(clone_ball))
        my_clone_balls.sort(key=lambda a: a['radius'], reverse=True)

        for clone_ball in clone_balls:
            if clone_ball['player'] != self.game_player_id:
                others_clone_balls.append(copy.deepcopy(clone_ball))
        others_clone_balls.sort(key=lambda a: a['radius'], reverse=True)
        return my_clone_balls, others_clone_balls

    def process_thorns_balls(self, thorns_balls, my_max_clone_ball):
        min_distance = 10000
        min_thorns_ball = None
        for thorns_ball in thorns_balls:
            if self.can_eat(my_max_clone_ball['radius'], thorns_ball['radius']):
                distance = (thorns_ball['position'] - my_max_clone_ball['position']).length()
                if distance < min_distance:
                    min_distance = distance
                    min_thorns_ball = copy.deepcopy(thorns_ball)
        return min_distance, min_thorns_ball

    def process_food_balls(self, food_balls, my_max_clone_ball):
        min_distance = 10000
        min_food_ball = None
        for food_ball in food_balls:
            distance = (food_ball['position'] - my_max_clone_ball['position']).length()
            if distance < min_distance:
                min_distance = distance
                min_food_ball = copy.deepcopy(food_ball)
        return min_distance, min_food_ball

    def preprocess(self, overlap):
        new_overlap = {}
        for k, v in overlap.items():
            if k =='clone':
                new_overlap[k] = []
                for index, vv in enumerate(v):
                    tmp={}
                    tmp['position'] = Vector2(vv[0],vv[1])
                    tmp['radius'] = vv[2]
                    tmp['player'] = int(vv[-2])
                    tmp['team'] = int(vv[-1])
                    new_overlap[k].append(tmp)
            else:
                new_overlap[k] = []
                for index, vv in enumerate(v):
                    tmp={}
                    tmp['position'] = Vector2(vv[0],vv[1])
                    tmp['radius'] = vv[2]
                    new_overlap[k].append(tmp)
        return new_overlap

    def preprocess_tuple2vector(self, overlap):
        new_overlap = {}
        for k, v in overlap.items():
            new_overlap[k] = []
            for index, vv in enumerate(v):
                new_overlap[k].append(vv)
                new_overlap[k][index]['position'] = Vector2(*vv['position'])
        return new_overlap
    
    def add_noise_to_direction(self, direction, noise_ratio=0.1):
        direction = direction + Vector2(((random.random() * 2 - 1)*noise_ratio)*direction.x, 
                                        ((random.random() * 2 - 1)*noise_ratio)*direction.y)
        return direction

    def radius_to_score(self, radius):
        return (math.pow(radius,2) - 0.15) / 0.042 * 100
    
    def can_eat(self, radius1, radius2):
        return self.radius_to_score(radius1) > 1.3 * self.radius_to_score(radius2)