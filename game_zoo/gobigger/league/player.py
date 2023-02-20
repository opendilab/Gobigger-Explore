import os
import pprint
import random
from abc import abstractmethod
from copy import deepcopy
from bigrl.core.league.algorithms import pfsp
from bigrl.core.utils.config_helper import read_config
import numpy as np
from .player_stat import PlayerStat
from .payoff import Payoff


class Player:
    """
    Overview:
        Base player class, player is the basic member of a league
    Interfaces:
        __init__
    Property:
        race, payoff, checkpoint_path, config_path, player_id, total_agent_step
    """
    name = "BasePlayer"  # override this variable for sub-class player
    attr_keys = ['player_id', 'pipeline', 'checkpoint_path', 'config_path', 'total_agent_step',
                 'decay', 'warm_up_size', 'total_game_count', 'min_win_rate_games']
    stat_keys = ['player_stat', 'payoff']

    def __init__(
            self,
            player_id: str,
            pipeline: str,
            checkpoint_path: str,
            config_path: str = '',
            total_agent_step: int = 0,
            decay: float = 0.99,
            warm_up_size: int = 100,
            total_game_count: int = 0,
            min_win_rate_games: int = 100,
            player_stat: PlayerStat = None,
            payoff: Payoff = None,
    ) -> None:
        """
        Overview:
            Initialize base player metadata
        Arguments:
            - cfg (:obj:`EasyDict`): player config dict
                e.g. StarCraft has 3 races ['terran', 'protoss', 'zerg']
            - checkpoint_path (:obj:`str`): one training phase step
            - player_id (:obj:`str`): player id
            - total_agent_step (:obj:`int`):  for active player, it should be 0, \
                for hist player, it should be parent player's ``_total_agent_step`` when ``snapshot``
        """
        self.player_id = player_id
        self.pipeline = pipeline
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.total_agent_step = total_agent_step
        self.warm_up_size = warm_up_size
        self.min_win_rate_games = min_win_rate_games
        self.decay = decay
        self.total_game_count = total_game_count
        self.player_stat = PlayerStat(self.decay, self.warm_up_size)
        if player_stat:
            self.player_stat.load_state_dict(player_stat.state_dict())
        self.payoff = Payoff(decay, warm_up_size, min_win_rate_games)
        if payoff:
            self.payoff.load_state_dict(payoff.state_dict())

    @abstractmethod
    def get_branch_players(self, active_players, hist_players, branch_probs_dict, job_type, cfg):
        raise NotImplementedError

    def get_branch(self, branch_probs_dict, ):
        branch_probs = branch_probs_dict[self.name]
        branches, branch_weights = list(branch_probs.keys()), branch_probs.values()
        branch = random.choices(branches, weights=branch_weights, k=1)[0]
        return branch

    def get_job(self, active_players, hist_players, branch_probs_dict, job_type, cfg):
        branch, players = self.get_branch_players(active_players, hist_players, branch_probs_dict, job_type, cfg)
        job_info = {
            'branch': branch,
            'player_id': [p.player_id for p in players],
            'pipeline': [p.pipeline for p in players],
            'checkpoint_path': [p.checkpoint_path for p in players],
            'config_path': [p.config_path for p in players],
            'send_data_players': [p.player_id for p in players if isinstance(p, ActivePlayer)]
        }
        if 'eval' in branch:
            job_info['send_data_players'] = []
        return job_info

    def reset_stats(self, src_player=None, stat_types=[]):
        if src_player is None:
            reset_stat_types = self.stat_keys if not stat_types else stat_types
            for k in reset_stat_types:
                stat = getattr(self, k)
                stat.reset()
        else:
            reset_stat_types = self.stat_keys if not stat_types else stat_types
            for k in reset_stat_types:
                setattr(self, k, deepcopy(getattr(src_player, k)))

    def __repr__(self):
        info = {attr_type: getattr(self, attr_type) for attr_type in self.attr_keys}
        return pprint.pformat(info)


class HistoricalPlayer(Player):
    """
    Overview:
        Historical player with fixed checkpoint, has a unique attribute ``parent_id``.
    Property:
        race, payoff, checkpoint_path, player_id, total_agent_step, parent_id
    """
    name = "HistoricalPlayer"
    attr_keys = Player.attr_keys + ['parent_id', ]

    def __init__(self,
                 player_id: str,
                 pipeline: str,
                 checkpoint_path: str,
                 config_path: str = '',
                 total_agent_step: int = 0,
                 decay: float = 0.99,
                 warm_up_size: int = 100,
                 min_win_rate_games: int = 100,
                 total_game_count: int = 0,
                 parent_id: str = 'none',
                 player_stat: PlayerStat = None,
                 payoff: Payoff = None,
                 ) -> None:
        """
        Overview:
            Initialize ``_parent_id`` additionally
        Arguments:
            - parent_id (:obj:`str`): id of hist player's parent, should be an active player
        """
        super(HistoricalPlayer, self).__init__(
            player_id=player_id,
            pipeline=pipeline,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            total_agent_step=total_agent_step,
            decay=decay,
            warm_up_size=warm_up_size,
            min_win_rate_games=min_win_rate_games,
            total_game_count=total_game_count,
            player_stat=player_stat,
            payoff=payoff,
        )
        self.parent_id = parent_id

    def get_branch_players(self, active_players, hist_players, branch_probs_dict, job_type, cfg):
        branch = self.get_branch(branch_probs_dict)
        team_num = cfg.env.team_num
        player_num = cfg.env.player_num_per_team
        players = [self] * player_num
        hist_players_list = list(hist_players.values())
        bot_players_list = [p for p in hist_players.values() if 'bot' in p.pipeline]
        if 'bot' in branch and len(bot_players_list) == 0:
            branch = 'ladder'

        if branch == 'ladder':
            opponent_players = random.choices(hist_players_list, k=team_num - 1)

        elif branch in {'ladder_1v1', }:
            opponent_player = random.choice(hist_players_list, )
            opponent_players = [opponent_player] * (team_num // 2)
            opponent_players += [self] * (team_num - 1 - team_num // 2)

        elif branch == 'bot':
            opponent_players = random.choices(bot_players_list, k=team_num - 1)
        elif branch == 'diff_all':
            hist_players_without_bot = [p for p in hist_players_list if 'bot' not in p.pipeline]
            player_list = random.sample(hist_players_without_bot, k=team_num)
            players = []
            for p in player_list:
                players += [p] * player_num
            return branch, players
        elif branch == 'specified':
            specified_player_id_list = cfg.league.get('specified', None)
            hist_players_without_bot = [p for p in hist_players_list if 'bot' not in p.pipeline]
            if specified_player_id_list is None or len(specified_player_id_list) != team_num:
                player_list = random.sample(hist_players_without_bot, k=team_num)
            else:
                player_list = [hist_players[p] for p in specified_player_id_list]
            players = []
            for p in player_list:
                players += [p] * player_num
            return branch, players
        elif branch == 'sp':
            players = []
            for _ in range(team_num):
                players += [self] * player_num
            return branch, players
        elif branch in {'bot_1v1', }:
            opponent_player = random.choice(bot_players_list, )
            opponent_players = [opponent_player] * (team_num // 2)
            opponent_players += [self] * (team_num - 1 - team_num // 2)
        elif branch == 'pure_bot':
            players = random.choices(bot_players_list, k=1) * player_num
            opponent_players = random.choices(bot_players_list, k=team_num - 1)
        for p in opponent_players:
            players += [p] * player_num
        return branch, players


class ActivePlayer(Player):
    """
    Overview:
        Active player class, active player can be updated
    Interface:
        __init__, is_trained_enough, snapshot, mutate, get_job
    Property:
        race, payoff, checkpoint_path, player_id, total_agent_step
    """
    name = "ActivePlayer"
    attr_keys = Player.attr_keys + ['one_phase_step', 'chosen_weight', 'last_enough_step', 'snapshot_times',
                                    'reset_flag', 'snapshot_flag', ]

    def __init__(self,
                 player_id: str,
                 pipeline: str,
                 checkpoint_path: str,
                 config_path: str = '',
                 total_agent_step: int = 0,
                 decay: float = 0.99,
                 warm_up_size: int = 100,
                 min_win_rate_games: int = 100,
                 total_game_count: int = 0,
                 chosen_weight: float = 1,
                 one_phase_step: int = 2e8,
                 last_enough_step: int = 0,
                 snapshot_times: int = 0,
                 reset_flag: bool = False,
                 snapshot_flag: bool = False,
                 payoff: Payoff = None,
                 player_stat: PlayerStat = None,
                 ) -> None:
        """
        Overview:
            Initialize player metadata, depending on the game
        Note:
            - one_phase_step (:obj:`int`): active player will be considered trained enough after one phase step
            - last_enough_step (:obj:`int`): player's last step number that satisfies ``_is_trained_enough``
            - exploration (:obj:`function`): exploration function, e.g. epsilon greedy with decay
            - snapshot flag and reset flag will change only when we use league update
        """
        super(ActivePlayer, self).__init__(
            player_id=player_id,
            pipeline=pipeline,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            total_agent_step=total_agent_step,
            decay=decay,
            warm_up_size=warm_up_size,
            min_win_rate_games=min_win_rate_games,
            total_game_count=total_game_count,
            player_stat=player_stat,
            payoff=payoff,
        )
        self.chosen_weight = chosen_weight
        self.one_phase_step = one_phase_step
        self.last_enough_step = last_enough_step
        self.snapshot_times = snapshot_times
        self.snapshot_flag = snapshot_flag
        self.reset_flag = reset_flag

    def get_branch(self, branch_probs_dict, job_type=None):
        if job_type == 'eval':
            branch = 'eval_1v1'
        else:
            branch = super(ActivePlayer, self).get_branch(branch_probs_dict)
        return branch

    def is_trained_enough(self, active_players, hist_players, *args, **kwargs) -> bool:
        """
        Overview:
            Judge whether this player is trained enough for further operation
        Returns:
            - flag (:obj:`bool`): whether this player is trained enough
        """
        if self.snapshot_flag:
            self.snapshot_flag = False
            self.last_enough_step = self.total_agent_step
            return True
        step_passed = self.total_agent_step - self.last_enough_step
        if step_passed >= self.one_phase_step:
            self.last_enough_step = self.total_agent_step
            return True
        return False

    def snapshot(self, model_dir) -> HistoricalPlayer:
        self.snapshot_times += 1
        h_player_id = self.player_id + f'H{self.snapshot_times}'
        h_checkpoint_path = os.path.join(model_dir, h_player_id + f'_{self.total_agent_step}.pth.tar')
        hp = HistoricalPlayer(player_id=h_player_id,
                              pipeline=self.pipeline,
                              checkpoint_path=h_checkpoint_path,
                              config_path=self.config_path,
                              total_agent_step=self.total_agent_step,
                              decay=self.decay,
                              warm_up_size=self.warm_up_size,
                              min_win_rate_games=self.min_win_rate_games,
                              total_game_count=self.total_game_count,
                              parent_id=self.player_id,
                              payoff=deepcopy(self.payoff),
                              player_stat=deepcopy(self.player_stat),
                              )
        return hp

    def is_reset(self):
        return False

    def reset_checkpoint(self):
        if os.path.exists(self.config_path):
            player_config = read_config(self.config_path)
            reset_checkpoint_path = player_config.agent.get('reset_checkpoint_path', None)
            if reset_checkpoint_path and os.path.exists(reset_checkpoint_path):
                reset_player_id = player_config.agent.get('reset_player_id', None)
                return reset_player_id, reset_checkpoint_path

            teacher_checkpoint_path = player_config.agent.get('teacher_checkpoint_path', None)
            if teacher_checkpoint_path and os.path.exists(teacher_checkpoint_path):
                return None, teacher_checkpoint_path
        return 'none', 'none'


class MainPlayer(ActivePlayer):
    name = "MainPlayer"

    def get_branch_players(self, active_players, hist_players, branch_probs_dict, job_type, cfg):
        branch = self.get_branch(branch_probs_dict, job_type)
        hist_players_list = list(hist_players.values())
        bot_players_list = [p for p in hist_players.values() if 'bot' in p.pipeline]
        test_players_list = [p for p in hist_players.values() if 'TEST' in p.pipeline]
        team_num = cfg.env.team_num
        player_num = cfg.env.player_num_per_team
        players = [self] * player_num

        if len(hist_players_list) == 0 \
                or ('bot' in branch and len(bot_players_list) == 0) \
                or ('test' in branch and len(test_players_list) == 0):
            branch = 'sp'

        if branch == 'sp':
            main_players = [p for p in active_players.values() if isinstance(p, MainPlayer)]
            opponent_players = random.choices(main_players, k=team_num - 1)
        elif branch == 'sp_1v1':
            main_players = [p for p in active_players.values() if isinstance(p, MainPlayer)]
            opponent_player = random.choice(main_players, )
            if opponent_player != self and self.payoff.get_pfsp_max_score_weight(opponent_player.player_id) < 0.3:
                hist_player_keys = [hist_player_id for hist_player_id, hist_player in hist_players.items() if
                                    hist_player.parent_id == opponent_player.player_id]
                if len(hist_player_keys):
                    hist_player_weights = [self.payoff.get_pfsp_max_score_weight(player_id) for player_id in
                                           hist_player_keys]
                    hist_player_pfsp_probs = pfsp(np.array(hist_player_weights), weighting='variance')
                    opponent_player_id = random.choices(hist_player_keys, weights=hist_player_pfsp_probs, k=1)[0]
                    opponent_player = hist_players[opponent_player_id]
                else:
                    opponent_player = self

            opponent_players = [opponent_player] * (team_num // 2)
            opponent_players += [self] * (team_num - 1 - team_num // 2)

        elif branch == 'bot':
            opponent_players = random.choices(bot_players_list, k=team_num - 1)

        elif branch in {'bot_1v1', 'eval_bot_1v1'}:
            opponent_player = random.choice(bot_players_list, )
            opponent_players = [opponent_player] * (team_num // 2)
            opponent_players += [self] * (team_num - 1 - team_num // 2)

        elif branch in {'fsp', 'eval'}:
            opponent_players = random.choices(hist_players_list, k=team_num - 1)

        elif branch in {'fsp_1v1', 'eval_1v1'}:
            opponent_player = random.choice(hist_players_list, )
            opponent_players = [opponent_player] * (team_num // 2)
            opponent_players += [self] * (team_num - 1 - team_num // 2)

        elif branch == 'pfsp':
            weights = []
            opponent_players_list = [p for p in hist_players_list if
                                     ('bot' not in p.pipeline) and ('TEST' not in p.pipeline)] + [self]
            for p in opponent_players_list:
                if 'rank' in p.player_stat.stat_info_record:
                    weights.append(team_num - p.player_stat.stat_info_record['rank'].val)
                else:
                    weights.append(team_num // 2)
            opponent_players = random.choices(opponent_players_list, weights=weights, k=team_num - 1)

        elif branch == 'pfsp_1v1':
            weights = []
            opponent_players_list = [p for p in hist_players_list if
                                     ('bot' not in p.pipeline) and ('TEST' not in p.pipeline)] + [self]
            for p in opponent_players_list:
                if 'rank' in p.player_stat.stat_info_record:
                    weights.append(team_num - p.player_stat.stat_info_record['rank'].val)
                else:
                    weights.append(team_num // 2)
            opponent_player = random.choices(opponent_players_list, weights=weights, k=1)[0]
            opponent_players = [opponent_player] * (team_num // 2)
            opponent_players += [self] * (team_num - 1 - team_num // 2)

        elif branch == 'pfsp_payoff_1v1':
            hist_player_keys = [p.player_id for p in hist_players_list if
                                ('bot' not in p.pipeline) and ('TEST' not in p.pipeline)]
            hist_player_weights = [self.payoff.get_pfsp_win_rate(player_id, ) for player_id in
                                   hist_player_keys]
            hist_player_pfsp_probs = pfsp(np.array(hist_player_weights), weighting='squared')
            opponent_player_id = random.choices(hist_player_keys, weights=hist_player_pfsp_probs, k=1)[0]
            opponent_player = hist_players[opponent_player_id]
            opponent_players = [opponent_player] * (team_num // 2)
            opponent_players += [self] * (team_num - 1 - team_num // 2)

        elif branch == 'pfsp_score_1v1':
            hist_player_keys = [p.player_id for p in hist_players_list if
                                ('bot' not in p.pipeline) and ('TEST' not in p.pipeline)]
            hist_player_weights = [self.payoff.get_pfsp_max_score_weight(player_id, ) for player_id in
                                   hist_player_keys]
            hist_player_pfsp_probs = pfsp(np.array(hist_player_weights), weighting='squared')
            opponent_player_id = random.choices(hist_player_keys, weights=hist_player_pfsp_probs, k=1)[0]
            opponent_player = hist_players[opponent_player_id]
            opponent_players = [opponent_player] * (team_num // 2)
            opponent_players += [self] * (team_num - 1 - team_num // 2)


        elif branch == 'test':
            opponent_players = random.choices(test_players_list, k=team_num - 1)

        elif branch in {'test_1v1', }:
            opponent_player = random.choice(test_players_list, )
            opponent_players = [opponent_player] * (team_num // 2)
            opponent_players += [self] * (team_num - 1 - team_num // 2)

        elif branch == 'mix':
            branch_probs = cfg.league.get('mix_branch_probs', {'sp': 0.5, 'fsp': 0., 'pfsp': 0.5, 'bot': 0.})
            opponent_players = []
            opponent_players_list = [p for p in hist_players_list if
                                     ('bot' not in p.pipeline) and ('TEST' not in p.pipeline)] + [self]
            weights = []
            for p in opponent_players_list:
                if 'rank' in p.player_stat.stat_info_record:
                    weights.append(team_num - p.player_stat.stat_info_record['rank'].val)
                else:
                    weights.append(team_num // 2)

            hist_player_keys = [p.player_id for p in hist_players_list if
                                ('bot' not in p.pipeline) and ('TEST' not in p.pipeline)]
            if len(hist_player_keys) != 0:
                hist_player_weights = [self.payoff.get_pfsp_win_rate(player_id, ) for player_id in
                                       hist_player_keys]
                hist_player_pfsp_probs = pfsp(np.array(hist_player_weights), weighting='squared')
                hist_player_score_weights = [self.payoff.get_pfsp_max_score_weight(player_id, ) for player_id in
                                             hist_player_keys]
                hist_player_pfsp_score_probs = pfsp(np.array(hist_player_score_weights), weighting='squared')

            for _ in range(team_num - 1):
                branches, branch_weights = list(branch_probs.keys()), branch_probs.values()
                mix_branch = random.choices(branches, weights=branch_weights, k=1)[0]
                if mix_branch == 'sp':
                    main_players = [p for p in active_players.values() if isinstance(p, MainPlayer)]
                    opponent_player = random.choices(main_players)[0]
                    if opponent_player != self and self.payoff.get_pfsp_max_score_weight(
                            opponent_player.player_id) < 0.3:
                        hist_opponent_keys = [hist_player_id for hist_player_id, hist_player in hist_players.items() if
                                              hist_player.parent_id == opponent_player.player_id]
                        if len(hist_opponent_keys):
                            opponent_hist_player_weights = [self.payoff.get_pfsp_max_score_weight(player_id) for
                                                            player_id in
                                                            hist_opponent_keys]
                            opponent_hist_player_pfsp_probs = pfsp(np.array(opponent_hist_player_weights),
                                                                   weighting='variance')
                            opponent_player_id = \
                            random.choices(hist_opponent_keys, weights=opponent_hist_player_pfsp_probs, k=1)[
                                0]
                            opponent_player = hist_players[opponent_player_id]
                        else:
                            opponent_player = self
                    opponent_players += [opponent_player]
                elif mix_branch == 'fsp':
                    opponent_players += random.choices(opponent_players_list, k=1)
                elif mix_branch == 'pfsp':
                    opponent_players += random.choices(opponent_players_list, weights=weights, k=1)
                elif mix_branch == 'pfsp_payoff' and len(hist_player_keys) != 0:
                    opponent_player_id = random.choices(hist_player_keys, weights=hist_player_pfsp_probs, k=1)[0]
                    opponent_player = hist_players[opponent_player_id]
                    opponent_players += [opponent_player]
                elif mix_branch == 'pfsp_score' and len(hist_player_keys) != 0:
                    opponent_player_id = random.choices(hist_player_keys, weights=hist_player_pfsp_score_probs, k=1)[0]
                    opponent_player = hist_players[opponent_player_id]
                    opponent_players += [opponent_player]
                elif mix_branch == 'bot' and len(bot_players_list) != 0:
                    opponent_players += random.choices(bot_players_list, k=1)
                else:
                    mix_branch = 'sp'
                    opponent_players += [self]
            branch = f'mix_{mix_branch}'
        for p in opponent_players:
            players += [p] * player_num
        return branch, players


if __name__ == '__main__':
    player = MainPlayer('MP0', 'default', 'none')
    print(player)
