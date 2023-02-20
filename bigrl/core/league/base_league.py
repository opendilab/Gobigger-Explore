import os
import pickle
import pprint
import random
import shutil
import threading
import time
import traceback
from abc import ABC
from pprint import pformat
import lz4.frame
import torch
import torch.multiprocessing as tm
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter
from bigrl.core.utils import LockContextType, read_config, LockContext
from bigrl.core.utils.config_helper import save_config, deep_merge_dicts
from bigrl.core.utils.import_helper import import_pipeline_module
from bigrl.core.utils.log_helper import TextLogger
from .player import HistoricalPlayer, MainPlayer, ActivePlayer
from .league_api import create_league_app

class BaseLeague(ABC):
    MainPlayerClass = MainPlayer
    HistPlayerClass = HistoricalPlayer
    ActivePlayerClass = ActivePlayer

    def __init__(self, cfg: EasyDict) -> None:
        self.whole_cfg = cfg
        self.env_name = self.whole_cfg.env.name

        # dir related
        self.setup_dir()
        self.lock = LockContext(type_=LockContextType.THREAD_LOCK)

        # setup logger and save user_config
        self.setup_logger()
        self.save_config(print_config=True, init_backup=True)

        # setup players
        self.init_league()

        # thread to save resume
        self.start_save_resume()
        # thread for deal_with_actor_send_result
        self.setup_collect_result_service()


    def setup_dir(self):
        self.exp_dir = os.path.join('experiments', self.whole_cfg.common.experiment_name)
        self.config_dir = os.path.join(self.exp_dir, 'config')
        self.resume_dir = os.path.join(self.exp_dir, 'league_resume')
        self.model_dir = os.path.abspath(os.path.join(self.exp_dir, 'league_models'))
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.resume_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def start_save_resume(self):
        self.save_resume_freq = self.cfg.get('save_resume_freq', 3600)
        self.save_resume_thread = threading.Thread(target=self._save_resume_loop, daemon=True)
        self.save_resume_thread.start()

    def setup_collect_result_service(self):
        self.result_queue = tm.Queue()
        self.send_result_thread = threading.Thread(target=self._send_result_loop, daemon=True)
        self.send_result_thread.start()

    def setup_logger(self):
        # pay off related
        self.stat_decay = self.cfg.get('stat_decay', 0.99)
        self.stat_warm_up_size = self.cfg.get('stat_warm_up_size', 100)

        # logging related
        self.logger = TextLogger(path=os.path.join(self.exp_dir, 'league_log'),
                                 name='league')

        # tb_logger related
        # we have different tb_log for different player
        self.log_show_freq = self.cfg.get('log_show_freq', 1000)
        self.save_log_freq = self.cfg.get('save_log_freq', 100)
        self.tb_log_dir = os.path.join(os.getcwd(), self.exp_dir, 'league_tb_log', )
        self.tb_logger = SummaryWriter(self.tb_log_dir, )

    # ************************** league init *********************************
    def init_league(self) -> None:
        if self.cfg.resume_path and os.path.isfile(self.cfg.resume_path):
            self.logger.info('load league, path: {}'.format(self.cfg.resume_path))
            self.load_resume(self.cfg.resume_path)
        else:
            self.active_players = {}
            self.hist_players = {}
            active_cfg = self.cfg.active_players
            self.add_active_players(active_cfg)
            hist_cfg = self.cfg.hist_players
            self.add_hist_players(hist_cfg)
            self.logger.info('init league with active players:')
            self.logger.info(pformat(self.active_players.keys()))
            self.logger.info('init league with hist players:')
            self.logger.info(pformat(self.hist_players.keys()))

    def add_active_players(self, active_cfg):
        for player_id, pipeline, checkpoint_path, one_phase_step, chosen_weight \
                in zip(active_cfg.player_id, active_cfg.pipeline, active_cfg.checkpoint_path,
                       active_cfg.one_phase_step, active_cfg.chosen_weight):
            self.add_active_player(player_id, pipeline, checkpoint_path, one_phase_step, chosen_weight, )

    def add_active_player(self, player_id, pipeline, checkpoint_path, one_phase_step, chosen_weight, **kwargs):
        # Notice: when we load league resume, we will not use this function to init active_player
        # We will use player_id to determine activer player type
        # MP: mainplayer, ME: main exploiter, EP: BaseLeague exploiter, AE: Adaptive evolutionary exploiter
        player_checkpoint_path = os.path.join(self.model_dir, '{}_checkpoint.pth.tar'.format(player_id))
        if os.path.exists(checkpoint_path):
            self.copy_checkpoint(checkpoint_path, player_checkpoint_path, zero_last_iter=True)
        else:
            self.logger.info(f"{player_id} will use random init model")
            self.save_random_checkpoint(pipeline, player_checkpoint_path)
        player_type = self.get_active_player_type(player_id)
        if player_type is None:
            return False

        player = player_type(
            player_id=player_id,
            pipeline=pipeline,
            checkpoint_path=player_checkpoint_path,
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

    def add_hist_players(self, hist_cfg):
        for player_id, pipeline, checkpoint_path in zip(hist_cfg.player_id,
                                                        hist_cfg.pipeline, hist_cfg.checkpoint_path, ):
            self.add_hist_player(player_id, pipeline, checkpoint_path, )

    def add_hist_player(self, player_id, pipeline, checkpoint_path, parent_id='none',copy_checkpoint=True, **kwargs):
        if pipeline != 'bot':
            if checkpoint_path == 'none' or not os.path.exists(checkpoint_path):
                print(f'cant find checkpoint path {checkpoint_path}', flush=True)
                return False
            if copy_checkpoint:
                player_checkpoint_path = os.path.join(self.model_dir, player_id + '_' + os.path.basename(checkpoint_path))
                self.copy_checkpoint(checkpoint_path, player_checkpoint_path)
            else:
                player_checkpoint_path = checkpoint_path
        else:
            player_checkpoint_path = 'none'

        player = self.HistPlayerClass(
            player_id=player_id,
            pipeline=pipeline,
            checkpoint_path=player_checkpoint_path,
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

    def save_snapshot(self, player):
        hp = player.snapshot(self.model_dir)
        self.copy_checkpoint(player.checkpoint_path, hp.checkpoint_path)
        with self.lock:
            self.hist_players[hp.player_id] = hp
        self.logger.info(f'add history player:{hp.player_id}')
        return hp

    # ************************** learner *********************************
    def deal_with_register_learner(self, request_info):
        player_id = request_info['player_id']
        ip = request_info['ip']
        port = request_info['port']
        rank = request_info['rank']
        world_size = request_info['world_size']
        self.logger.info((ip, port, rank, world_size,))
        assert player_id in self.active_players.keys(), f'{player_id} not in active players{self.active_players.keys()}'
        self.logger.info('register learner: {}'.format(player_id))
        return {'checkpoint_path': self.active_players[player_id].checkpoint_path}

    def deal_with_learner_send_train_info(self, request_info: dict):
        """
        Overview:
            Update an active player's info
        Arguments:
            - player_info (:obj:`dict`): an info dict of the player which is to be updated
        """
        player_id = request_info['player_id']
        train_steps = request_info['train_steps']
        checkpoint_path = request_info['checkpoint_path']
        player = self.active_players[player_id]
        with self.lock:
            player.total_agent_step += train_steps
            player.checkpoint_path = checkpoint_path
        reset_flag = player.reset_flag
        if player.is_trained_enough(self.active_players, self.hist_players, ):
            self.save_snapshot(player)
            reset_flag |= player.is_reset()
        if reset_flag:
            player.reset_flag = False
            with self.lock:
                reset_player_id, reset_checkpoint_path = player.reset_checkpoint()
                if reset_player_id == 'none':
                    player_checkpoint_path = os.path.join(self.model_dir, '{}_checkpoint.pth.tar'.format(player_id))
                    self.save_random_checkpoint(player.pipeline, player_checkpoint_path)
                    player.checkpoint_path = reset_checkpoint_path = player_checkpoint_path


                src_player = self.active_players[reset_player_id] if reset_player_id in self.all_players else None
                player.reset_stats(src_player=src_player)
                self.logger.info(f'{player_id} has been reset to checkpoint path {reset_checkpoint_path}')
            return {'reset_checkpoint_path': reset_checkpoint_path}
        return {'reset_checkpoint_path': 'none'}

    # ************************** actor *********************************
    def deal_with_register_actor(self, request_info):
        # actor_id = request_info['actor_id']
        active_players = [player for player in self.active_players.values() if player.chosen_weight > 0]
        return {'player_id': [p.player_id for p in active_players],
                'pipeline': [p.pipeline for p in active_players],
                'checkpoint_path': [p.checkpoint_path for p in active_players],
                }

    def deal_with_actor_ask_for_job(self, request_info: dict):
        job_type = request_info['job_type']
        job_player_id = request_info.get('job_player_id', None)
        if job_type == 'ladder':
            if job_player_id is not None and job_player_id != 'none' and job_player_id in self.hist_players:
                player = self.hist_players[job_player_id]
            else:
                player = self.choose_hist_player()
            job_info = player.get_job(self.active_players,self.hist_players, branch_probs_dict=self.cfg.branch_probs, job_type='ladder')
        else:
            if job_player_id is not None and job_player_id != 'none' and job_player_id in self.active_players:
                player = self.active_players[job_player_id]
            else:
                player = self.choose_active_player()
            job_info = player.get_job(self.active_players, self.hist_players,
                                      branch_probs_dict=self.cfg.get('branch_probs',{}), job_type=job_type)
        if self.cfg.get('show_job', False):
            print(job_info, flush=True)
        return job_info

    def deal_with_actor_send_result(self, request_info: dict):
        """
        Overview:
            Finish current job. Update active players' ``launch_count`` to release job space,
            and shared payoff to record the game result.
        Arguments:
            - job_result (:obj:`dict`): a dict containing job result information
        """
        self.result_queue.put(request_info)
        return True

    def _send_result_loop(self):
        torch.set_num_threads(1)
        while True:
            if self.result_queue.empty():
                time.sleep(0.01)
            else:
                result_info = self.result_queue.get()
                try:
                    self.update_result(result_info)
                except Exception as e:
                    self.logger.error(f'[Update Result ERROR]{e}', )
                    self.logger.error(''.join(traceback.format_tb(e.__traceback__)))

    def update_result(self, result_info):
        player_id = result_info['player_id']
        player = self.all_players[player_id]
        player_stat = result_info['player_stat']
        with self.lock:
            player.total_game_count += 1
            player.player_stat.update(player_stat)
        if player.total_game_count % self.log_show_freq == 0:
            self.logger.info('=' * 30 + f'{player.player_id}' + '=' * 30)
            self.logger.info(player.player_stat.get_text())
        if player.total_game_count % self.save_log_freq == 0:
            if isinstance(player,self.ActivePlayerClass):
                self.tb_logger.add_scalar(tag=f'{player_id}/agent_step', scalar_value=player.total_agent_step, global_step=player.total_game_count)
            player_stat_info = player.player_stat.stat_info_dict()
            for k, val in player_stat_info.items():
                self.tb_logger.add_scalar(tag=f'{player_id}/{k}', scalar_value=val, global_step=player.total_game_count)

    def choose_active_player(self):
        active_player_ids = list(self.active_players.keys())
        active_player_weights = [self.active_players[player_id].chosen_weight for player_id in
                                 active_player_ids]
        chosen_player_id = random.choices(active_player_ids, weights=active_player_weights, k=1)[0]
        chosen_player = self.active_players[chosen_player_id]
        return chosen_player

    def choose_hist_player(self):
        hist_player_ids = [player_id for player_id,player in self.hist_players.items() if player.pipeline != 'bot']
        if len(hist_player_ids) == 0:
            hist_player_ids = [player_id for player_id,player in self.hist_players.items()]
        hist_player_game_counts = [self.hist_players[player_id].total_game_count for player_id in
                                   hist_player_ids]
        all_game_counts = sum(hist_player_game_counts)
        if all_game_counts == 0 or len(hist_player_ids) == 1:
            hist_player_weights = None
        else:
            hist_player_weights = [all_game_counts - game_count for game_count in hist_player_game_counts]
        chosen_player_id = random.choices(hist_player_ids, weights=hist_player_weights, k=1)[0]
        chosen_player = self.hist_players[chosen_player_id]
        return chosen_player

    # ************************** config *********************************
    def save_config(self, print_config=False, init_backup=False):
        time_label = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        config_path = os.path.join(self.config_dir, f'user_config_{time_label}.yaml')
        if init_backup:
            whole_user_config_path = os.path.join(os.getcwd(), 'user_config.yaml')
            if os.path.exists(whole_user_config_path):
                shutil.copyfile(whole_user_config_path, os.path.join(self.exp_dir, f'user_config.yaml'))
        save_config(self.whole_cfg, config_path)
        if print_config:
            self.logger.info(pprint.pformat(self.whole_cfg))
        self.logger.info(f'save config to config_path:{config_path}')
        return True

    def update_config(self):
        resume_path = self.save_resume()
        self.logger.info(f'save resume to resume_path:{resume_path}')
        load_config_path = os.path.join(self.exp_dir, 'user_config.yaml')
        load_config = read_config(load_config_path)
        self.whole_cfg = deep_merge_dicts(self.whole_cfg, load_config)
        self.logger.info(f'update config from config_path:{load_config_path}')
        self.save_config(print_config=False)
        return True


    # ************************** resume *********************************
    def _save_resume_loop(self):
        self.lasttime = int(time.time())
        self.logger.info("[UP] check resume thread start")
        while True:
            now_time = int(time.time())
            if now_time - self.lasttime >= self.save_resume_freq:
                self.save_resume()
                self.lasttime = now_time
            time.sleep(self.save_resume_freq)

    def save_resume(self, ):
        resume_data = {}
        resume_data['active_players'] = self.active_players
        resume_data['hist_players'] = self.hist_players
        resume_label = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        resume_data_path = os.path.join(self.resume_dir, f'league.resume.' + resume_label)
        with open(resume_data_path, "wb") as f:
            compress_data = lz4.frame.compress(pickle.dumps(resume_data))
            f.write(compress_data)
        self.logger.info('[resume] save to {}'.format(resume_data_path))
        if not self.send_result_thread.is_alive():
            self.logger.info('send result thread is not alive, restart it')
            self.send_result_thread = threading.Thread(target=self._send_result_loop, daemon=True)
            self.send_result_thread.start()
        return resume_data_path

    def load_resume(self, resume_path: str):
        with open(resume_path, "rb") as f:
            resume_data = pickle.loads(lz4.frame.decompress(f.read()))
        self.active_players = resume_data['active_players']
        self.hist_players = resume_data['hist_players']
        # sl_eval will not copy checkpoint_path
        copy_checkpoint_flag = self.whole_cfg.league.get('copy_checkpoint',True)
        if copy_checkpoint_flag:
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

    # *********
    # debug use
    # *********
    # ************************** resume *********************************
    def deal_with_load_resume(self, request_info):
        resume_path = request_info['path']
        if resume_path and os.path.exists(resume_path):
            self.load_resume(resume_path)
            return True
        else:
            return False

    # ************************** player *********************************
    def display_player(self, request_info):
        player_id = request_info['player_id']
        stat_types = request_info.get('stat_types', [])
        player_ids = self.get_correspondent_player_ids(player_id)
        for player_id in player_ids:
            self.logger.info('=' * 30 + player_id + '=' * 30)
            player = self.all_players[player_id]
            self.logger.info(player)
            if stat_types == 'all':
                display_stat_types = player.stat_keys
            elif isinstance(stat_types, str):
                display_stat_types = [stat_types]
            for stat_type in display_stat_types:
                if stat_type in player.stat_keys and hasattr(player, stat_type):
                    self.logger.info(getattr(player, stat_type).get_text())
        return True

    def deal_with_add_active_player(self, request_info):
        try:
            self.add_active_player(**request_info)
            return True
        except Exception as e:
            print(f'[Add Player Error]{e}', flush=True)
            print(''.join(traceback.format_tb(e.__traceback__)), flush=True)

    def deal_with_add_hist_player(self, request_info):
        try:
            self.add_hist_player(**request_info)
            return True
        except Exception as e:
            print(f'[Add Player Error]{e}', flush=True)
            print(''.join(traceback.format_tb(e.__traceback__)), flush=True)

    def deal_with_update_player(self, request_info):
        player_id = request_info.pop('player_id', None)
        if not player_id or player_id not in self.all_players:
            return False
        player = self.all_players[player_id]
        with self.lock:
            for attr_type, attr_value in request_info.items():
                if hasattr(player, attr_type):
                    if attr_type == 'one_phase_step' and isinstance(attr_value, str):
                        attr_value = int(float(attr_value))
                    setattr(player, attr_type, attr_value)
        self.logger.info(f'successfully update player{player_id}')
        self.logger.info(player)
        return True

    def deal_with_refresh_players(self, ):
        for player_id, old_player in self.active_players.items():
            new_player_info = {attr_type: getattr(old_player, attr_type, None) for attr_type in old_player.attr_keys + old_player.stat_keys}
            player_type = self.get_active_player_type(player_id)
            new_player = player_type(**new_player_info)
            with self.lock:
                self.active_players[player_id] = new_player
        for player_id, old_player in self.hist_players.items():
            new_player_info = {attr_type: getattr(old_player, attr_type, None) for attr_type in old_player.attr_keys+ old_player.stat_keys}
            new_player = self.HistPlayerClass(**new_player_info)
            with self.lock:
                self.hist_players[player_id] = new_player
        self.logger.info(f'successfully refresh all players')
        return True

    def deal_with_reset_player(self, request_info):
        player_id = request_info['player_id']
        stat_types = request_info.get('stat_types', [])
        player_ids = self.get_correspondent_player_ids(player_id)
        for player_id in player_ids:
            self.logger.info('=' * 30 + player_id + '=' * 30)
            player = self.all_players[player_id]
            for stat_type in stat_types:
                if hasattr(player, stat_type) and hasattr(getattr(player, stat_type),'reset'):
                    getattr(player, stat_type).reset()
                    print(f"success reset player {player_id} stat {stat_type}")
        return True

    def deal_with_remove_player(self, request_info):
        player_id = request_info['player_id']
        player_ids = self.get_correspondent_player_ids(player_id)
        for del_player_id in player_ids:
            with self.lock:
                if del_player_id in self.active_players:
                    self.active_players.pop(del_player_id, None)
                else:
                    self.hist_players.pop(del_player_id, None)
                self.logger.info(f'remove player{del_player_id}')
        return True

    # ************************** property *********************************
    def get_active_player_type(self, player_id):
        if 'MP' in player_id:
            return self.MainPlayerClass
        else:
            print(f"not support{player_id} for active players, must include one of ['MP']")
            return None

    @property
    def all_players(self):
        return {**self.active_players, **self.hist_players, }

    @property
    def cfg(self):
        with self.lock:
            return self.whole_cfg.league

    def copy_checkpoint(self, src, dest, keep_value=False, zero_last_iter=False):
        checkpoint = torch.load(src, map_location='cpu')
        # this part is actually for pretrained model
        checkpoint.pop('optimizer', None)
        if zero_last_iter and 'last_iter' in checkpoint.keys():
            checkpoint['last_iter'] = 0
        if not keep_value:
            checkpoint['model'] = {k: v for k, v in checkpoint['model'].items() if
                                   'value' not in k}
        torch.save(checkpoint, dest)
        print(f'copy checkpoint from {src} to {dest}', flush=True)

    def save_random_checkpoint(self, pipeline, dest):
        Model = import_pipeline_module(self.env_name, pipeline, 'Model')
        model = Model(cfg=self.whole_cfg,use_value_network=False)
        checkpoint = {'model': model.state_dict()}
        torch.save(checkpoint, dest)

    def get_correspondent_player_ids(self, player_id):
        if player_id == 'all':
            player_ids = self.all_players.keys()
        elif player_id == 'active':
            player_ids = self.active_players.keys()
        elif player_id == 'hist':
            player_ids = self.hist_players.keys()
        elif isinstance(player_id, list):
            player_ids = [p for p in player_id if p in self.all_players]
        elif player_id not in self.all_players:
            print(f'{player_id} not in league pool')
            player_ids = []
        else:
            player_ids = [player_id]
        return player_ids

    @staticmethod
    def create_league_app(league):
        return create_league_app(league)