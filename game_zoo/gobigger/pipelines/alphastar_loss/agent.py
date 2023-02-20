import math
from collections import deque

import torch

from bigrl.core.torch_utils.collate_fn import default_collate_with_dim
from .features import Features
from .model.model import Model

class Agent:
    HAS_MODEL = True
    ModelClass = Model

    def __init__(self, cfg,):
        self.whole_cfg = cfg
        self.model_last_iter = torch.zeros(size=())
        self.player_num = self.whole_cfg.env.player_num_per_team
        self.team_num = self.whole_cfg.env.team_num
        self.game_player_id = self.whole_cfg.agent.game_player_id  # start from 0
        self.game_team_id = self.game_player_id // self.player_num # start from 0
        self.send_data = self.whole_cfg.agent.send_data
        self.eval_padding = self.whole_cfg.agent.get('eval_padding',False) or self.send_data
        self.player_id = self.whole_cfg.agent.player_id
        self.features = Features(self.whole_cfg)
        self.reward_div_value = self.whole_cfg.agent.get('reward_div_value', 0.01)
        self.reward_type = self.whole_cfg.agent.get('reward_type', 'log_reward')
        self.rank_reward_interval = self.whole_cfg.agent.get('rank_reward_interval', 100)
        self.score_bin_ratio = self.whole_cfg.agent.get('score_bin_ratio', 1.3)
        self.spirit = self.whole_cfg.agent.get('spirit', 1)
        self.spore_reward_div_value = self.whole_cfg.agent.get('spore_reward_div_value', 10)
        self.mate_spore_reward_div_value = self.whole_cfg.agent.get('mate_spore_reward_div_value',10)
        self.clip_spore_reward = self.whole_cfg.agent.get('clip_spore_reward', False)
        self.clip_clone_reward = self.whole_cfg.agent.get('clip_clone_reward', False)
        self.clip_opponent_reward = self.whole_cfg.agent.get('clip_opponent_reward', False)

        self.clone_reward_div_value = self.whole_cfg.agent.get('clone_reward_div_value', 1)
        self.opponent_reward_div_value = self.whole_cfg.agent.get('opponent_reward_div_value', 1)
        self.dist_reward_div_value = self.whole_cfg.agent.get('dist_reward_div_value', 10)
        self.dist_avg_size_div_norm = self.whole_cfg.agent.get('dist_avg_size_div_norm', 8)

        self.start_spirit_step = self.whole_cfg.agent.get('start_spirit_step', 1000)
        self.end_spirit_step = self.whole_cfg.agent.get('end_spirit_step', 3000)
        self.use_action_mask = self.whole_cfg.agent.get('use_action_mask', False)

    def setup_model(self):
        self.model = self.ModelClass(self.whole_cfg)

    def reset(self):
        self.last_player_score = None
        self.last_leaderboard = {team_idx: 1000 * self.player_num for team_idx in range(self.team_num)}
        self.last_player_spore = 0
        self.last_team_spore = 0
        self.last_mate_spore = 0
        self.last_player_clone = 0
        self.last_team_clone = 0
        self.last_player_opponent = 0
        self.last_team_opponent = 0
        self.last_max_dist = 0
        self.last_min_dist = 0
        if self.send_data:
            self.data_buffer = deque(maxlen=self.whole_cfg.learner.data.unroll_len)
            self.push_count = 0
        self.game_step = 0
        self.stat = {'act_split': 0., 'act_spore': 0., 'act_stop': 0., 'act_split_cnt': 0.,
                     'act_spore_cnt': 0., 'act_split_ratio': 0., 'act_spore_ratio': 0., }
        if self.reward_type == 'score_bin':
            self.last_score_bin = 1000
            self.last_team_score_bin = 1000 * self.player_num
        self.last_action_type = self.features.direction_num * 2

    def preprocess(self, obs):
        self.last_player_score = obs[1][self.game_player_id]['score']
        if self.use_action_mask:
            can_eject = obs[1][self.game_player_id]['can_eject']
            can_split = obs[1][self.game_player_id]['can_split']
            action_mask = self.features.generate_action_mask(can_eject=can_eject,can_split=can_split)
        else:
            action_mask = self.features.generate_action_mask(can_eject=True,can_split=True)
        obs = self.features.transform_obs(obs, game_player_id=self.game_player_id,
                                          last_action_type=self.last_action_type,padding=self.eval_padding)
        obs = default_collate_with_dim([obs])

        obs['action_mask'] = action_mask.unsqueeze(0)
        return obs

    def step(self, obs):
        self.raw_obs = obs
        obs = self.preprocess(obs)
        self.model_input = obs
        if self.send_data:
            with torch.no_grad():
                self.model_output = self.model(self.model_input)
        else:
            with torch.no_grad():
                self.model_output = self.model.compute_action(self.model_input)
        actions = self.postprocess(self.model_output['action'].detach().numpy())
        self.game_step += 1
        return actions

    def postprocess(self, model_actions):
        actions = {}
        actions[self.game_player_id] = self.features.transform_action(model_actions[0])
        self.last_action_type = model_actions[0].item()
        self.update_stat(model_actions[0])
        return actions

    def get_spirit(self):
        if self.game_step <= self.start_spirit_step:
            return 0
        elif self.game_step <= self.end_spirit_step:
            spirit = (self.game_step - self.start_spirit_step) / (self.end_spirit_step - self.start_spirit_step)
            return spirit
        else:
            return 1

    def eval_postprocess(self, *args, **kwargs):
        return None

    def get_teammate_dist(self, observations):
        own_left_top_x, own_left_top_y, own_right_bottom_x, own_right_bottom_y = observations[1][self.game_player_id]['rectangle']
        own_view_x  = (own_left_top_x + own_right_bottom_x)/2
        own_view_y =  (own_left_top_y + own_right_bottom_y)/2
        own_width = own_right_bottom_x - own_left_top_x
        own_height = own_right_bottom_y - own_left_top_y

        min_dist = 200 * math.sqrt(2)
        max_dist = 0


        for player_id in observations[1].keys():
            team_id =  player_id // self.player_num
            player_obs = observations[1][player_id]
            if player_id != self.game_player_id and self.game_team_id == team_id:
                left_top_x, left_top_y, right_bottom_x, right_bottom_y = player_obs['rectangle']
                view_x = (left_top_x + right_bottom_x) / 2
                view_y = (left_top_y + right_bottom_y) / 2
                width = right_bottom_x - left_top_x
                height = right_bottom_y - left_top_y
                if self.dist_avg_size_div_norm > 0:
                    avg_size = (own_width + own_height + width + height) / self.dist_avg_size_div_norm
                    dist = max(0, math.sqrt((view_x - own_view_x) ** 2 + (view_y - own_view_y) ** 2) - avg_size)
                else:
                    dist = math.sqrt((view_x - own_view_x) ** 2 + (view_y - own_view_y) ** 2)
                max_dist = max(max_dist, dist)
                min_dist = min(min_dist, dist)
        return max_dist, min_dist

    def collect_data(self, next_obs, reward, done, info):
        leader_board = next_obs[0]['leaderboard']

        max_dist, min_dist = self.get_teammate_dist(next_obs)
        max_dist_rew = - (max_dist - self.last_max_dist)/self.dist_reward_div_value
        min_dist_rew = - (min_dist - self.last_min_dist)/self.dist_reward_div_value
        self.last_max_dist = max_dist
        self.last_min_dist = min_dist
        if self.spirit >= 0:
            spirit = self.spirit
        else:
            spirit = self.get_spirit()

        if self.reward_type == 'score':
            player_reward = next_obs[1][self.game_player_id]['score'] - self.last_player_score
            team_name = next_obs[1][self.game_player_id]['team_name']
            team_rewards_list = reward
            team_reward = team_rewards_list[team_name - 1]
            rew = (1 - spirit) * player_reward + spirit * team_reward / self.player_num
            rew /= self.reward_div_value
        elif self.reward_type == 'rank':
            if next_obs[0]['last_time'] % self.rank_reward_interval == 0:
                leader_board = list(next_obs[0]['leaderboard'].values())
                sorted_leader_board = sorted(leader_board)
                rank = sorted_leader_board.index(leader_board[self.game_team_id ])
                rew = (rank - ((self.team_num - 1) / 2)) / (self.team_num - 1) * 2
            else:
                rew = 0.
        elif self.reward_type == 'score_bin':
            player_reward = team_reward = 0
            if next_obs[1][self.game_player_id]['score'] / self.last_score_bin >= self.score_bin_ratio:
                bins = next_obs[1][self.game_player_id]['score'] / self.last_score_bin // self.score_bin_ratio
                player_reward = bins
                self.last_score_bin *= (self.score_bin_ratio ** bins)
            team_name = next_obs[1][self.game_player_id]['team_name']
            team_score = sum([next_obs[1][i]['score'] for i in next_obs[1] if next_obs[1][i]['team_name'] == team_name])
            if team_score / self.last_team_score_bin >= self.score_bin_ratio:
                bins = team_score / self.last_team_score_bin // self.score_bin_ratio
                team_reward = bins
                self.last_team_score_bin *= (self.score_bin_ratio ** bins)
            rew = (1 - spirit) * player_reward + spirit * team_reward / self.player_num
        elif self.reward_type == 'log_reward':
            team_rewards = {}
            for team_idx in range(self.team_num):
                team_rewards[team_idx] = math.log(leader_board[team_idx]) - math.log(
                    self.last_leaderboard[team_idx])
            player_reward = math.log(next_obs[1][self.game_player_id]['score']) - math.log(self.last_player_score)
            team_reward = team_rewards[self.game_team_id]
            rew = (1 - spirit) * player_reward + spirit * team_reward / self.player_num
            rew /= self.reward_div_value
        score_rew = rew

        team_spore = 0
        mate_spore = 0
        for player_id in range(self.game_team_id * self.player_num, (self.game_team_id+1) * self.player_num):
            team_spore += info['eats'][player_id]['spore']
            if player_id != self.game_player_id:
                mate_spore += info['eats'][player_id]['spore']
        player_spore = info['eats'][self.game_player_id]['spore']

        spore_rew = (player_spore - self.last_player_spore) / self.spore_reward_div_value
        team_spore_rew = ((team_spore - self.last_team_spore) / self.player_num) / self.spore_reward_div_value
        mate_spore_rew = ((mate_spore-self.last_mate_spore)/ (self.player_num + 1e-8)) / self.mate_spore_reward_div_value

        self.last_team_spore = team_spore
        self.last_mate_spore = mate_spore
        self.last_player_spore = player_spore
        if self.clip_spore_reward:
            spore_rew = min(spore_rew,1)
            team_spore_rew = min(team_spore_rew,1)
            mate_spore_rew = min(mate_spore_rew,1)


        team_clone = 0
        for player_id in range(self.game_team_id * self.player_num, (self.game_team_id+1) * self.player_num):
            team_clone += info['eats'][player_id]['clone_team']
        player_clone = info['eats'][player_id]['clone_team']

        clone_rew = (player_clone - self.last_player_clone) / self.clone_reward_div_value
        team_clone_rew = ((team_clone - self.last_team_clone) / self.player_num) / self.clone_reward_div_value

        self.last_team_clone = team_clone
        self.last_player_clone = player_clone

        if self.clip_clone_reward:
            clone_rew = min(clone_rew,1)
            team_clone_rew = min(team_clone_rew,1)


        team_opponent = 0
        for player_id in range(self.game_team_id * self.player_num, (self.game_team_id+1) * self.player_num):
            team_opponent += info['eats'][player_id]['clone_other']
        player_opponent = info['eats'][player_id]['clone_other']

        opponent_rew = (player_opponent - self.last_player_opponent) / self.opponent_reward_div_value
        team_opponent_rew = ((team_opponent - self.last_team_opponent) / self.player_num) / self.opponent_reward_div_value

        self.last_team_opponent = team_opponent
        self.last_player_opponent = player_opponent

        if self.clip_opponent_reward:
            opponent_rew = min(opponent_rew,1)
            team_opponent_rew = min(team_opponent_rew,1)

        self.last_leaderboard = leader_board
        step_data = {
            'obs': self.model_input,
            # 'hidden_state': self.hidden_state,
            'action': self.model_output['action'],
            'action_logp': self.model_output['action_logp'],
            'reward': {'score': torch.tensor([score_rew], dtype=torch.float),
                       'spore': torch.tensor([spore_rew], dtype=torch.float),
                       'mate_spore':torch.tensor([mate_spore_rew], dtype=torch.float),
                       'team_spore': torch.tensor([team_spore_rew], dtype=torch.float),
                       'clone': torch.tensor([clone_rew], dtype=torch.float),
                       'team_clone': torch.tensor([team_clone_rew], dtype=torch.float),
                       'opponent': torch.tensor([opponent_rew], dtype=torch.float),
                       'team_opponent': torch.tensor([team_opponent_rew], dtype=torch.float),
                       'max_dist':torch.tensor([max_dist_rew], dtype=torch.float),
                       'min_dist': torch.tensor([min_dist_rew], dtype=torch.float),
                       },
            'done': torch.tensor([done], dtype=torch.bool),
            'model_last_iter': torch.tensor([self.model_last_iter.item()], dtype=torch.float),
        }
        # push data
        self.data_buffer.append(step_data)
        self.push_count += 1
        # self.hidden_state = self.model_output["hidden_state"]

        if self.push_count == self.whole_cfg.learner.data.unroll_len or done:
            last_step_data = {
                'obs': self.preprocess(next_obs),
                # 'hidden_state': self.hidden_state,
            }
            list_data = list(self.data_buffer)
            list_data.append(last_step_data)
            self.push_count = 0
            return_data = default_collate_with_dim(list_data, cat=True)

            return_data = return_data
        else:
            return_data = None

        return return_data

    def update_stat(self, action=None):
        direction_num = self.whole_cfg.agent.features.direction_num
        if action == direction_num * 2:
            self.stat['act_stop'] += 1
        elif action == (direction_num * 2 + 1):
            self.stat['act_spore_cnt'] += 1
            if self.raw_obs[1][self.game_player_id]['can_eject']:
                self.stat['act_spore'] += 1
            self.stat['act_spore_ratio'] = self.stat['act_spore'] / (self.stat['act_spore_cnt'] + 1e-9)
        elif action == (direction_num * 2 + 2):
            self.stat['act_split_cnt'] += 1
            if self.raw_obs[1][self.game_player_id]['can_split']:
                self.stat['act_split'] += 1
            self.stat['act_split_ratio'] = self.stat['act_split'] / (self.stat['act_split_cnt'] + 1e-9)
