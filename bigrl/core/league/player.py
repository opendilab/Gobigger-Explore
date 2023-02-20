import os
import pprint
from .player_stat import PlayerStat
from bigrl.core.utils.config_helper import read_config
from copy import deepcopy

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
    attr_keys = ['player_id', 'pipeline', 'checkpoint_path','config_path', 'total_agent_step',
                 'decay', 'warm_up_size', 'total_game_count']
    stat_keys = ['player_stat']

    def __init__(
            self,
            player_id: str,
            pipeline: str,
            checkpoint_path: str,
            config_path: str='',
            total_agent_step: int = 0,
            decay: float = 0.99,
            warm_up_size: int = 100,
            total_game_count: int = 0,
            player_stat: PlayerStat=None,
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
        self.decay = decay
        self.total_game_count = total_game_count
        self.player_stat = PlayerStat(self.decay, self.warm_up_size)
        if player_stat:
            self.player_stat.load_state_dict(player_stat.state_dict())

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

    def get_job(self,*args, **kwargs):
        job_info = {'player_id': [self.player_id], 'pipeline': [self.pipeline],'config_path': [self.config_path],
                    'checkpoint_path': [self.checkpoint_path], 'send_data_players': [self.player_id]}
        return job_info

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
                 total_game_count: int = 0,
                 parent_id: str = 'none',
                 player_stat: PlayerStat = None,
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
            total_game_count=total_game_count,
            player_stat=player_stat,
        )
        self.parent_id = parent_id

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
    attr_keys = Player.attr_keys + ['one_phase_step', 'chosen_weight', 'last_enough_step', 'snapshot_times','reset_flag','snapshot_flag', ]

    def __init__(self,
                 player_id: str,
                 pipeline: str,
                 checkpoint_path: str,
                 config_path: str='',
                 total_agent_step: int = 0,
                 decay: float = 0.99,
                 warm_up_size: int = 100,
                 total_game_count: int = 0,
                 chosen_weight: float = 1,
                 one_phase_step: int = 2e8,
                 last_enough_step: int = 0,
                 snapshot_times: int = 0,
                 reset_flag:bool=False,
                 snapshot_flag: bool=False,
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
            total_game_count=total_game_count,
            player_stat=player_stat,
        )
        self.chosen_weight = chosen_weight
        self.one_phase_step = one_phase_step
        self.last_enough_step = last_enough_step
        self.snapshot_times = snapshot_times
        self.snapshot_flag = snapshot_flag
        self.reset_flag = reset_flag


    def is_trained_enough(self,  active_players, hist_players, *args, **kwargs) -> bool:
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
        """
        Overview:
            Generate a snapshot hist player from the current player, called in league manager's ``_snapshot``.
        Returns:
            - snapshot_player (:obj:`HistoricalPlayer`): new instantiated hist player
        Note:
            This method only generates a hist player object without saving the checkpoint, which should be
            completed by the interaction between coordinator and learner.
        """
        self.snapshot_times += 1
        h_player_id = self.player_id + f'H{self.snapshot_times}'
        h_checkpoint_path = os.path.join(model_dir, h_player_id + f'_{self.total_agent_step}.pth.tar')
        hp = HistoricalPlayer(player_id=h_player_id,
                              pipeline=self.pipeline,
                              checkpoint_path=h_checkpoint_path,
                              config_path = self.config_path,
                              total_agent_step=self.total_agent_step,
                              decay=self.decay,
                              warm_up_size=self.warm_up_size,
                              total_game_count=self.total_game_count,
                              parent_id=self.player_id,
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
                reset_player_id =  player_config.agent.get('reset_player_id', None)
                return reset_player_id, reset_checkpoint_path
        return 'none', 'none'

class MainPlayer(ActivePlayer):
    name = "MainPlayer"


if __name__ =='__main__':
    player = MainPlayer('MP0','default','none')
    print(player)