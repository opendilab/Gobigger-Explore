import math
import torch


class EnvManagerStatus:
    def __init__(self, cfg, env_num):
        self.whole_cfg = cfg
        self.env_num = env_num
        self.env_status = {
            env_id: EnvStatus(self.whole_cfg) for env_id in
            range(self.env_num)}

    def update(self, next_obs, reward, done, info, ):
        reward_list = []
        done_list = []
        for env_id in range(self.env_num):
            env_reward_list, env_done_list = self.env_status[env_id].update(next_obs[env_id], reward[env_id],
                                                                            done[env_id],
                                                                            info[env_id])
            reward_list.extend(env_reward_list)
            done_list.extend(env_done_list)
        return reward_list, done_list

    def reset(self, env_id=None):
        if env_id is None:
            env_ids = range(self.env_num)
        elif isinstance(env_id, list):
            env_ids = env_id
        elif isinstance(env_id, int):
            env_ids = [env_id]
        else:
            raise NotImplementedError

        for env_id in env_ids:
            self.env_status[env_id].reset()

    def __len__(self):
        return len(self.env_status)

    def __getitem__(self, idx):
        return self.env_status[idx]

class EnvStatus:
    def __init__(self, cfg):
        self.whole_cfg = cfg
        self.player_num = self.whole_cfg.env.player_num_per_team
        self.team_num = self.whole_cfg.env.team_num
        self.direction_num = self.whole_cfg.agent.features.get('direction_num', 12)
        self.reward_div_value = self.whole_cfg.agent.get('reward_div_value', 0.01)
        self.reward_type = self.whole_cfg.agent.get('reward_type', 'log_reward')
        self.spirit = self.whole_cfg.agent.get('spirit', 1)
        self.spore_reward_div_value = self.whole_cfg.agent.get('spore_reward_div_value', 10)
        self.mate_spore_reward_div_value = self.whole_cfg.agent.get('mate_spore_reward_div_value', 10)
        self.clip_spore_reward = self.whole_cfg.agent.get('clip_spore_reward', False)
        self.clip_clone_reward = self.whole_cfg.agent.get('clip_clone_reward', False)
        self.clip_opponent_reward = self.whole_cfg.agent.get('clip_opponent_reward', False)
        self.clone_reward_div_value = self.whole_cfg.agent.get('clone_reward_div_value', 1)
        self.opponent_reward_div_value = self.whole_cfg.agent.get('opponent_reward_div_value', 1)
        self.dist_reward_div_value = self.whole_cfg.agent.get('dist_reward_div_value', 10)
        self.dist_avg_size_div_norm = self.whole_cfg.agent.get('dist_avg_size_div_norm', 8)

        self.start_spirit_progress = self.whole_cfg.agent.get('start_spirit_progress', 0.2)
        self.end_spirit_progress = self.whole_cfg.agent.get('end_spirit_progress', 0.8)
        self.player_init_score = self.whole_cfg.agent.get('player_init_score',13000)
        self.reset()

    def update(self, next_obs, reward, done, info, ):
        done_list = []
        last_time = next_obs[0]['last_time']
        total_frame = next_obs[0]['total_frame']
        progress = last_time / total_frame
        spirit = self.get_spirit(progress)
        score_rewards_list = []
        for game_player_id in range(self.player_num * self.team_num):
            game_team_id = game_player_id // self.player_num
            player_score = next_obs[1][game_player_id]['score']
            team_score = next_obs[0]['leaderboard'][game_team_id]
            if self.reward_type == 'log_reward':
                player_reward = math.log(player_score) - math.log(self.last_player_scores[game_player_id])
                team_reward = math.log(team_score) - math.log(self.last_leaderboard[game_team_id])
                score_reward = (1 - spirit) * player_reward + spirit * team_reward / self.player_num
                score_reward = score_reward / self.reward_div_value
                score_rewards_list.append(score_reward)
            elif self.reward_type == 'score':
                player_reward = player_score - self.last_player_scores[game_player_id]
                team_reward = team_score - self.last_leaderboard[game_team_id]
                score_reward = (1 - spirit) * player_reward + spirit * team_reward / self.player_num
                score_reward = score_reward / self.reward_div_value
                score_rewards_list.append(score_reward)
            elif self.reward_type == 'sqrt_player':
                player_reward = player_score - self.last_player_scores[game_player_id]
                reward_sign =  (player_reward > 0) - (player_reward < 0) # np.sign
                score_rewards_list.append(reward_sign * math.sqrt(abs(player_reward)) / 2)
            elif self.reward_type == 'sqrt_team':
                team_reward = team_score - self.last_leaderboard[game_team_id]
                reward_sign =  (team_reward > 0) - (team_reward < 0) # np.sign
                score_rewards_list.append(reward_sign * math.sqrt(abs(team_reward)) / 2)
            else:
                raise NotImplementedError
            self.last_player_scores[game_player_id] = player_score
            done_list.append(done)
        self.last_leaderboard = next_obs[0]['leaderboard']
        return score_rewards_list, done_list

    def get_spirit(self, progress):
        if progress < self.start_spirit_progress:
            return 0
        elif progress <= self.end_spirit_progress:
            spirit = (progress - self.start_spirit_progress) / (self.end_spirit_progress - self.start_spirit_progress)
            return spirit
        else:
            return 1

    def reset(self):
        self.last_action_types = {player_id: self.direction_num * 2 for player_id in
                                  range(self.player_num * self.team_num)}
        self.last_leaderboard = {team_idx: self.player_init_score * self.player_num for team_idx in range(self.team_num)}
        self.last_player_scores = {player_id: self.player_init_score for player_id in range(self.player_num * self.team_num)}
        self.last_player_spores = {player_id: 0 for player_id in range(self.player_num * self.team_num)}
        self.last_player_clones = {player_id: 0 for player_id in range(self.player_num * self.team_num)}
        self.last_player_opponents = {player_id: 0 for player_id in range(self.player_num * self.team_num)}
        self.last_player_max_dists = {player_id: 0 for player_id in range(self.player_num * self.team_num)}
        self.last_player_min_dists = {player_id: 0 for player_id in range(self.player_num * self.team_num)}
