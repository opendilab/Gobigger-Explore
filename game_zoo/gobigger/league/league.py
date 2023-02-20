import os
import pickle
import random
import shutil
import threading
import time

import lz4.frame
from torch.utils.tensorboard import SummaryWriter

from bigrl.core.utils.config_helper import read_config
from bigrl.core.league.ladder.trueskill_utils import trueskill
from bigrl.core.league.base_league import BaseLeague
from .player import ActivePlayer, HistoricalPlayer, MainPlayer


class League(BaseLeague):
    ActivePlayerClass = ActivePlayer
    MainPlayerClass = MainPlayer
    HistPlayerClass = HistoricalPlayer

    # ************************** league init *********************************
    def setup_logger(self):
        super(League, self).setup_logger()
        self.use_player_tb_log = self.cfg.get('use_player_tb_log', False)
        if self.use_player_tb_log:
            self.player_tb_log_dir = os.path.join(os.getcwd(), self.exp_dir, 'league_player_tb_log', )
            os.makedirs(self.player_tb_log_dir, exist_ok=True)
            self.player_tb_logs = {}

        self.use_trueskill = self.cfg.get('use_trueskill', False) and trueskill is not None
        if self.use_trueskill:
            self.logger.info('Use trueskill!')
            self.trueskill_log_dir = os.path.join(self.exp_dir, 'trueskill_tb_log')
            os.makedirs(self.trueskill_log_dir, exist_ok=True)
            trueskill_cfg = self.cfg.get('trueskill', {})
            self.trueskill_save_freq = trueskill_cfg.get('save_freq', 100)
            self.trueskill_show_freq = trueskill_cfg.get('show_freq', 1000)
            from bigrl.single.worker.ladder.trueskill_utils import TrueSkillSystem
            self.trueskill_system = TrueSkillSystem(trueskill_cfg)
            self.trueskill_tb_logs = {}
        else:
            self.logger.info('Not use trueskill!')


    # ************************** league init *********************************
    def add_active_players(self, active_cfg):
        for player_id, pipeline, checkpoint_path, config_path, \
            one_phase_step, chosen_weight \
                in zip(active_cfg.player_id, active_cfg.pipeline, active_cfg.checkpoint_path,
                       active_cfg.get('config_path', [''] * len(active_cfg.player_id)),
                       active_cfg.one_phase_step, active_cfg.chosen_weight):
            self.add_active_player(player_id=player_id, pipeline=pipeline, checkpoint_path=checkpoint_path,
                                   config_path=config_path,
                                   one_phase_step=one_phase_step, chosen_weight=chosen_weight,
                                   )

    def add_hist_players(self, hist_cfg):
        for player_id, pipeline, checkpoint_path, config_path in zip(hist_cfg.player_id,
                                                                     hist_cfg.pipeline,
                                                                     hist_cfg.checkpoint_path,
                                                                     hist_cfg.get('config_path',
                                                                                  [''] * len(hist_cfg.player_id))):
            self.add_hist_player(player_id=player_id, pipeline=pipeline, checkpoint_path=checkpoint_path,
                                 config_path=config_path, )

    def add_active_player(self, player_id, pipeline, checkpoint_path, config_path='', one_phase_step='1e9',
                          chosen_weight=1, **kwargs):
        # Notice: when we load league resume, we will not use this function to init active_player
        # We will use player_id to determine activer player type
        # MP: mainplayer, ME: main exploiter, EP: BaseLeague exploiter, AE: Adaptive evolutionary exploiter
        player_checkpoint_path = os.path.join(self.model_dir, '{}_checkpoint.pth.tar'.format(player_id))
        if os.path.exists(checkpoint_path):
            self.copy_checkpoint(checkpoint_path, player_checkpoint_path, zero_last_iter=True)
        else:
            self.logger.info(f"{player_id} will use random init model")
            self.save_random_checkpoint(pipeline, player_checkpoint_path)

        player_config_path = os.path.join(self.exp_dir,'agent_config', f'{player_id}_agent_config.yaml')
        if config_path and os.path.exists(config_path):
            os.makedirs(os.path.join(self.exp_dir, 'agent_config'), exist_ok=True)
            shutil.copyfile(config_path, player_config_path)
            self.logger.info(f"copy agent_config of {player_id} from {config_path} to {player_config_path}")
        else:
            player_config_path = ''

        player_type = self.get_active_player_type(player_id)
        if player_type is None:
            return False

        player = player_type(
            player_id=player_id,
            pipeline=pipeline,
            checkpoint_path=player_checkpoint_path,
            config_path=player_config_path,
            chosen_weight=chosen_weight,
            one_phase_step=int(float(one_phase_step)),
            decay=self.stat_decay,
            warm_up_size=self.stat_warm_up_size,
            **kwargs,
        )
        with self.lock:
            self.active_players[player_id] = player
        self.logger.info(f'league add active_players:{player_id}\n')
        # Notice: we only initial snapshot for main player
        if isinstance(player, self.MainPlayerClass) and self.cfg.get('save_initial_snapshot', False):
            self.save_snapshot(player)
        return True

    def add_hist_player(self, player_id, pipeline, checkpoint_path, config_path='', parent_id='none',
                        copy_checkpoint=True, **kwargs):
        if 'bot' not in pipeline:
            if checkpoint_path == 'none' or not os.path.exists(checkpoint_path):
                print(f'cant find checkpoint path {checkpoint_path}', flush=True)
                return False
            if copy_checkpoint:
                player_checkpoint_path = os.path.join(self.model_dir,
                                                      player_id + '_' + os.path.basename(checkpoint_path))
                self.copy_checkpoint(checkpoint_path, player_checkpoint_path)
            else:
                player_checkpoint_path = checkpoint_path
        else:
            player_checkpoint_path = 'none'

        player_config_path = os.path.join(self.exp_dir,'agent_config', f'{player_id}_agent_config.yaml')
        if config_path and os.path.exists(config_path):
            os.makedirs(os.path.join(self.exp_dir, 'agent_config'), exist_ok=True)
            shutil.copyfile(config_path, player_config_path)
            self.logger.info(f"copy agent_config of {player_id} from {config_path} to {player_config_path}")
        else:
            player_config_path = ''

        player = self.HistPlayerClass(
            player_id=player_id,
            pipeline=pipeline,
            checkpoint_path=player_checkpoint_path,
            config_path=player_config_path,
            total_agent_step=0,
            decay=self.stat_decay,
            warm_up_size=self.stat_warm_up_size,
            parent_id=parent_id,
            **kwargs,
        )

        with self.lock:
            self.hist_players[player_id] = player
        self.logger.info(f'league add hist_players:{player_id}')
        return True

    # ************************** actor *********************************
    def deal_with_actor_ask_for_job(self, request_info: dict):
        job_type = request_info['job_type']
        job_player_id = request_info.get('job_player_id', None)
        if job_type == 'ladder':
            if job_player_id is not None and job_player_id in self.hist_players:
                player = self.hist_players[job_player_id]
            else:
                player = self.choose_hist_player()
            job_info = player.get_job(self.active_players,
                                      self.hist_players,
                                      branch_probs_dict=self.cfg.branch_probs,
                                      job_type='ladder',
                                      cfg=self.whole_cfg)
        else:
            if job_player_id is not None and job_player_id in self.active_players:
                player = self.active_players[job_player_id]
            else:
                player = self.choose_active_player()
            job_info = player.get_job(self.active_players, self.hist_players,
                                      branch_probs_dict=self.cfg.get('branch_probs', {}), job_type=job_type,
                                      cfg=self.whole_cfg)
        if self.cfg.get('show_job', False):
            print(job_info, flush=True)
        return job_info

    def get_opponent_stat(self, home_player_id, branch, result_info,):
        if '1v1' not in branch or len(result_info) != 2:
            return None, None
        player_game_ranks = {player_id: result_info[player_id]['rank'] for player_id in result_info}
        player_ranks = {key: rank for rank, key in
                        enumerate(sorted(player_game_ranks, key=player_game_ranks.get, ), 1)}
        player_winrates = {}
        for player_id, rank in player_ranks.items():
            if rank == 1:
                win_rate = 1
            elif rank == 1.5:
                win_rate = 0.5
            else:
                win_rate = 0
            player_winrates[player_id] = win_rate
        opponent_id = [p for p in result_info if p != home_player_id][0]
        opponent_stat = {'win_rate': player_winrates[home_player_id],
                         'score': result_info[home_player_id]['score'],
                         'max_score': result_info[home_player_id]['max_score'],
                         'max_team_score': result_info[home_player_id]['max_team_score'],
                         'rank': result_info[home_player_id]['rank'],
                         'op_score': result_info[opponent_id]['score'],
                         'op_max_score': result_info[opponent_id]['max_score'],
                         'op_max_team_score': result_info[opponent_id]['max_team_score'],
                         }
        return opponent_id,opponent_stat

    def update_result(self, result_info):
        branch = result_info.pop('branch')

        # update trueskill
        if self.use_trueskill and len(result_info) > 1:
            player_ids = [player_id for player_id in result_info]
            player_max_scores = {player_id: result_info[player_id]['max_team_score'] for player_id in player_ids}
            player_rank = {key: rank for rank, key in
                           enumerate(sorted(player_max_scores, key=player_max_scores.get, reverse=True), )}
            rank_list = [player_rank[player_id] for player_id in player_ids]
            self.trueskill_system.update(agent_names=player_ids, rank_list=rank_list)
            if self.trueskill_system.update_count % self.trueskill_save_freq:
                for player_id in self.trueskill_system.ratings.keys():
                    if player_id not in self.trueskill_tb_logs:
                        self.trueskill_tb_logs[player_id] = SummaryWriter(
                            os.path.join(self.trueskill_log_dir, player_id))
                    player_tb_log = self.trueskill_tb_logs[player_id]
                    stat_info = self.trueskill_system.get_agent_stat(player_id)
                    for k,v in stat_info.items():
                        player_tb_log.add_scalar(tag=k, scalar_value=v,
                                             global_step=self.trueskill_system.update_count)
            if self.trueskill_system.update_count % self.trueskill_show_freq:
                self.trueskill_system.get_text()

        for player_id, player_stat in result_info.items():
            if player_id not in self.all_players:
                continue
            else:
                player = self.all_players[player_id]
            if self.use_player_tb_log:
                if player_id not in self.player_tb_logs:
                    self.player_tb_logs[player_id] = SummaryWriter(
                        os.path.join(self.player_tb_log_dir, player_id))
                player_tb_log = self.player_tb_logs[player_id]
            with self.lock:
                player.total_game_count += 1
                player.player_stat.update(player_stat)
                opponent_id,opponent_stat = self.get_opponent_stat(player_id, branch, result_info)
                if opponent_id:
                    player.payoff.update(opponent_id=opponent_id, opponent_stat=opponent_stat)
                    opponent_stat_info = player.payoff.get_opponent_stat_info(opponent_id)
                    opponent_game_count = opponent_stat_info.pop('game_count')
                    if self.use_player_tb_log and player_id in self.active_players and opponent_game_count % self.save_log_freq == 0:
                        for k,val in opponent_stat_info.items():
                            player_tb_log.add_scalar(tag=f'{opponent_id}/{k}',
                                                     scalar_value=val,
                                                     global_step=opponent_game_count)

            if player.total_game_count % self.log_show_freq == 0:
                self.logger.info('=' * 30 + f'{player.player_id}' + '=' * 30)
                self.logger.info(player.player_stat.get_text())
                if hasattr(player,'payoff'):
                    self.logger.info(player.payoff.get_text())


            if player.total_game_count % self.save_log_freq == 0:
                player_stat_info = player.player_stat.stat_info_dict()
                if self.use_player_tb_log:
                    player_tb_log.add_scalar(tag='agent_step',
                                              scalar_value=player.total_agent_step,
                                              global_step=player.total_game_count)
                    for k, val in player_stat_info.items():
                        player_tb_log.add_scalar(tag=k, scalar_value=val,
                                                  global_step=player.total_game_count)
                else:
                    if isinstance(player, ActivePlayer):
                        self.tb_logger.add_scalar(tag=f'{player_id}/agent_step',
                                                  scalar_value=player.total_agent_step,
                                                  global_step=player.total_game_count)
                    for k, val in player_stat_info.items():
                        self.tb_logger.add_scalar(tag=f'{player_id}/{k}', scalar_value=val,
                                                  global_step=player.total_game_count)
        self.update_eval_result(result_info)

    def save_resume(self, ):
        resume_data = {}
        resume_data['active_players'] = self.active_players
        resume_data['hist_players'] = self.hist_players
        if self.use_trueskill:
            resume_data['trueskill'] = self.trueskill_system
        resume_label = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        resume_data_path = os.path.join(self.resume_dir, f'league.resume.' + resume_label)
        with open(resume_data_path, "wb") as f:
            compress_data = lz4.frame.compress(pickle.dumps(resume_data))
            f.write(compress_data)
        self.logger.info('[resume] save to {}'.format(resume_data_path))
        if not self.send_result_thread.is_alive():
            self.logger.info('send result thread is not alive, restart it')
            self.send_result_thread = threading.Thread(target=self.send_result_loop, daemon=True)
            self.send_result_thread.start()
        return resume_data_path

    def load_resume(self, resume_path: str):
        with open(resume_path, "rb") as f:
            resume_data = pickle.loads(lz4.frame.decompress(f.read()))
        self.active_players = resume_data['active_players']
        self.hist_players = resume_data['hist_players']
        if self.use_trueskill and 'trueskill' in resume_data:
            self.trueskill_system = resume_data['trueskill']
        if self.whole_cfg.league.get('copy_checkpoint', True):
            for player_id, player in self.all_players.items():
                old_checkpoint_path = player.checkpoint_path
                new_checkpoint_path = os.path.join(self.model_dir, os.path.basename(old_checkpoint_path))
                if old_checkpoint_path == 'none':
                    continue
                if not os.path.exists(old_checkpoint_path):
                    if os.path.exists(new_checkpoint_path):
                        player.checkpoint_path = new_checkpoint_path
                    else:
                        print(f"cant find checkpoint path:{old_checkpoint_path}")
                        raise FileNotFoundError
                else:
                    if os.path.exists(new_checkpoint_path):
                        player.checkpoint_path = new_checkpoint_path
                    elif old_checkpoint_path != new_checkpoint_path:
                        player.checkpoint_path = new_checkpoint_path
                        shutil.copyfile(old_checkpoint_path, new_checkpoint_path)
                        print(f"copy player_id model to path:{new_checkpoint_path}")
        self.logger.info('successfully load league, path: {}'.format(resume_path))

    def show_trueskill(self, ):
        self.logger.info(self.trueskill_system.get_text())
        return True

    def refresh_trueskill(self, ):
        self.logger.info(self.trueskill_system.get_text())
        trueskill_players_list = list(self.trueskill_system.ratings.keys())
        for p in trueskill_players_list:
            if p not in self.all_players:
                self.trueskill_system.ratings.pop(p, None)
                self.trueskill_system.game_counts.pop(p, None)
        self.logger.info(self.trueskill_system.get_text())
        return True
                            
    def update_eval_result(self, result_info):
        if 'bot' in result_info.keys():
            print(result_info)
            env_step = self.all_players['MP0'].total_agent_step//3
            for player_id, player_stat in result_info.items():
                if player_id not in self.all_players:
                    continue
                else:
                    player = self.all_players[player_id]
                if player.total_game_count % self.save_log_freq == 0:
                    player_stat_info = player.player_stat.stat_info_dict()
                    
                    if isinstance(player, ActivePlayer):
                        self.tb_logger.add_scalar(tag=f'eval_vsbot/env_step',
                                                    scalar_value=env_step,
                                                    global_step=player.total_game_count)
                    for k, val in player_stat_info.items():
                        self.tb_logger.add_scalar(tag=f'eval_vsbot/{player_id}-{k}', scalar_value=val,
                                                    global_step=env_step)

if __name__ == '__main__':
    import os.path as osp

    cfg = read_config(
        osp.join(
            osp.dirname(__file__),
            "league_default_config.yaml"))
    league = League(cfg)
