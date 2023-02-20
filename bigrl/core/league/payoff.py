from collections import defaultdict
from functools import partial

from tabulate import tabulate
from copy import deepcopy
from bigrl.core.utils.log_helper import EmaMeter
from numbers import Number

class Payoff:
    """
    Overview:
        Payoff data structure to record historical match result, each player owns one specific payoff
    Interface:
        __init__, __getitem__, add_player, update
    Property:
        players
    """
    data_keys = ['win_rate',]
    def __init__(self, decay: float = 0.99, warm_up_size: int = 100, min_win_rate_games=100) -> None:
        """
        Overview: Initialize payoff
        Arguments:
            - home_id (:obj:`str`): home player id
            - decay (:obj:`float`): update step decay factor
            - min_win_rate_games (:obj:`int`): min games for win rate calculation
        """
        self.decay = decay
        self.warm_up_size = warm_up_size
        self.min_win_rate_games = min_win_rate_games
        self.reset()

    def win_rate_opponent(self, opponent_id, min_win_rate_games=True) -> float:
        """
        Overview:
            Get win rate against an opponent player
        Arguments:
            - player (:obj:`Player`): the opponent player to calculate win rate
        Returns:
            - win rate (:obj:`float`): float win rate value. \
                Only when total games is no less than ``self.min_win_rate_games``, \
                can the win rate be calculated according to [win, draw, loss, game], or return 0.5 by default.
        """
        # not enough game record case
        if (self.stat_info_record[opponent_id]['win_rate'].count < self.min_win_rate_games) and min_win_rate_games:
            return 0.5
        else:
            return self.stat_info_record[opponent_id]['win_rate'].val

    def update(self, opponent_id, opponent_stat, ) -> bool:
        """
        Overview:
            Update payoff with a match_info
        Arguments:
            - match_info (:obj:`dict`): a dict contains match information,
                owning at least 3 keys('home_id', 'away_id', 'result')
        Returns:
            - result (:obj:`bool`): whether update is successful
        """
        for k,val in opponent_stat.items():
            if k not in self.stat_info_record[opponent_id]:
                self.stat_info_record[opponent_id][k] = EmaMeter(self.decay, self.warm_up_size)
            self.stat_info_record[opponent_id][k].update(val)
        self.stat_info_record[opponent_id]['game_count'] += 1
        return True

    def get_pfsp_win_rate(self, opponent_id):
        if opponent_id in self.stat_info_record and\
                self.stat_info_record[opponent_id]['win_rate'].count >= self.min_win_rate_games:
            return self.stat_info_record[opponent_id]['win_rate'].val
        else:
            return 0.5

    def get_opponent_stat_info(self, opponent_id):
        opponent_stat_info = {}
        for k,val in self.stat_info_record[opponent_id].items():
            if isinstance(val,Number):
                opponent_stat_info[k] = val
            else:
                opponent_stat_info[k] = self.stat_info_record[opponent_id][k].val
        opponent_stat_info['game_count'] = self.stat_info_record[opponent_id]['game_count']
        return opponent_stat_info

    def get_text(self):
        table_data = []
        data_keys = set(self.data_keys)
        for opponent_id, opponent_stat in self.stat_info_record.items():
            data_keys.update(opponent_stat.keys())
        if 'game_count' in data_keys:
            data_keys.remove('game_count')
        data_keys = sorted(list(data_keys))
        headers = ['opponent_id'] + data_keys + ['game_count']
        for opponent_id, stat_info in sorted(self.stat_info_record.items()):
            value_info = []
            for k in data_keys:
                if k not in stat_info:
                    value_info.append(0.0)
                else:
                    value_info.append(f"{stat_info[k].val:.3f}")
            line_data = [opponent_id] + value_info + \
                    [stat_info['game_count']]
            table_data.append(line_data)
        if len(table_data) == 0:
            table_text = ''
        else:
            table_text = "\n" + tabulate(table_data, headers=headers, tablefmt='grid',
                                     stralign='left', numalign='left')
        return table_text

    @staticmethod
    def stat_template(decay, warm_up_size, data_keys):
        template = {item: EmaMeter(decay, warm_up_size, ) for item in data_keys}
        template['game_count'] = 0
        return template

    def reset(self):
        self.stat_info_record = defaultdict(
            partial(self.stat_template, self.decay, self.warm_up_size, self.data_keys))

    def load_state_dict(self, state_dict):
        self.stat_info_record = deepcopy(state_dict.pop('stat_info_record', {}))

    def state_dict(self,):
        return {'stat_info_record': self.stat_info_record}

if __name__ == '__main__':
    import random

    stat = Payoff(decay=0.99, warm_up_size=100, min_win_rate_games=1000)
    import pickle

    opponent_id_list = ['model1', 'model2']
    for i in range(1000):
        opponent_id = random.choice(opponent_id_list)
        win_rate = random.random()
        opponent_stat = {
            'win_rate': win_rate,
            'op_health': 1,
        }
        stat.update(opponent_id, opponent_stat)
    for i in range(1000):
        opponent_id = 'model3'
        win_rate = random.random()
        opponent_stat = {
            'win_rate': win_rate,
            'op_health': 1,
            'op_alive': random.random(),
        }
        stat.update(opponent_id, opponent_stat)
    x = pickle.dumps(stat)
    y = pickle.loads(x)
    print(y.get_text())
    for op in opponent_id_list:
        stat.get_opponent_stat_info(op)