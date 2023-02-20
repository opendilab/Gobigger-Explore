from tabulate import tabulate

from bigrl.core.utils.log_helper import EmaMeter
from copy import deepcopy

class PlayerStat:
    """
    Overview:
        Payoff data structure to record historical match result, each player owns one specific payoff
    Interface:
        __init__, __getitem__, add_player, update
    Property:
        players
    """

    def __init__(self, decay, warm_up_size, max_column=10) -> None:
        """
        Overview: Initialize payoff
        Arguments:
            - home_id (:obj:`str`): home player id
            - decay (:obj:`float`): update step decay factor
            - min_win_rate_games (:obj:`int`): min games for win rate calculation
        """
        self.decay = decay
        self.warm_up_size = warm_up_size
        self.reset()
        self.max_column = max_column

    def update(self, stat_info, ) -> bool:
        """
        Overview:
            Update payoff with a match_info
        Arguments:
            - match_info (:obj:`dict`): a dict contains match information,
                owning at least 3 keys('home_id', 'away_id', 'result')
        Returns:
            - result (:obj:`bool`): whether update is successful
        """
        self.game_count += 1
        for k in stat_info:
            if k not in self.stat_info_record:
                self.stat_info_record[k] = EmaMeter(self.decay, self.warm_up_size)
            self.stat_info_record[k].update(stat_info[k])
        return True

    def stat_info_dict(self):
        return {k: val.val for k, val in self.stat_info_record.items()}

    def get_text(self, ):
        info_text = ''
        table_text = '\n' + "=" * 12 + 'Stat' + "=" * 12
        headers, line_data = ['game_count'], [self.game_count]
        stat_keys_list = list(self.stat_info_record.keys())
        for idx, k in enumerate(stat_keys_list):

            line_data.append(f'{self.stat_info_record[k].val:.3f}')
            headers.append(k)
            if len(headers) > self.max_column or idx == len(stat_keys_list) - 1:
                table_data = []
                table_data.append(line_data)
                table_text += "\n" + tabulate(table_data, headers=headers, tablefmt='grid',
                                              stralign='left', numalign='left')
                headers, line_data = [], []
        info_text += table_text
        return info_text

    def reset(self):
        self.stat_info_record = {}
        self.game_count = 0

    def load_state_dict(self, state_dict):
        self.game_count = deepcopy(state_dict.pop('game_count', 0))
        self.stat_info_record = deepcopy(state_dict.pop('stat_info_record', {}))

    def state_dict(self,):
        return {'game_count':self.game_count,'stat_info_record': self.stat_info_record}

if __name__ == '__main__':
    import random

    stat = PlayerStat(decay=0.99, warm_up_size=100)
    import pickle

    for i in range(100):
        stat_info = {k: random.random() for k in range(100)}
        stat.update(stat_info, )
    x = pickle.dumps(stat)
    y = pickle.loads(x)
    print(y.get_text())
