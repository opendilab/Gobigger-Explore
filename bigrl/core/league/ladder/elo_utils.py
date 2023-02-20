import random
from collections import defaultdict
from functools import partial
from typing import List

from tabulate import tabulate


class ELORating:
    ELO_RESULT_WIN = 1
    ELO_RESULT_LOSS = -1
    ELO_RESULT_DRAW = 0

    def __init__(self, K=16):
        self.K = K

    def updateRating(self, ratingA, ratingB, result):
        scoreAwin = self.computeScore(ratingA, ratingB)

        if result == self.ELO_RESULT_WIN:
            score_adjust = 1
        elif result == self.ELO_RESULT_LOSS:
            score_adjust = 0
        else:
            score_adjust = 0.5

        new_ratingA = ratingA + self.K * (score_adjust - scoreAwin)
        new_ratingB = ratingB - self.K * (score_adjust - scoreAwin)
        return new_ratingA, new_ratingB

    def updateRatingDelta(self, ratingA, ratingB, result):
        scoreAwin = self.computeScore(ratingA, ratingB)

        if result == self.ELO_RESULT_WIN:
            score_adjust = 1
        elif result == self.ELO_RESULT_LOSS:
            score_adjust = 0
        else:
            score_adjust = 0.5

        delta_ratingA = self.K * (score_adjust - scoreAwin)
        delta_ratingB = - self.K * (score_adjust - scoreAwin)
        return delta_ratingA, delta_ratingB

    @classmethod
    def computeScore(cls, ratingA, ratingB):
        return 1 / (1 + pow(10, (ratingB - ratingA) / 400))


class EloSystem:
    def __init__(self, cfg={}):
        self.cfg = cfg
        self.K = self.cfg.get('K', 16)
        self.default_value = self.cfg.get('default_value', 1500)
        self.elo_rating = ELORating(K=self.K)
        self.ratings = defaultdict(partial(default_elo_rating, self.default_value))
        self.game_counts = defaultdict(int)
        self.update_count = 0

    def update(self, agent_names: List[str], reward_list: List[float], ):
        agent1, agent2 = agent_names
        reward1, reward2 = reward_list
        if reward1 > reward2:
            agent1_result = 1
        elif reward1 == reward2:
            agent1_result = 0
        else:
            agent1_result = -1

        delta_rating1, delta_rating2 = self.elo_rating.updateRatingDelta(self.ratings[agent1],
                                                                         self.ratings[agent2],
                                                                         result=agent1_result)
        self.ratings[agent1] += delta_rating1
        self.ratings[agent2] += delta_rating2
        for name in agent_names:
            self.game_counts[name] += 1
        self.update_count += 1

    def leaderboard(self):
        leaderboard_content = []
        rank = 1
        last_elo = -float('inf')
        anchors = list(self.ratings.items())
        anchors_sorted = sorted(anchors, key=lambda x: x[1], reverse=True)
        for i, (name, _) in enumerate(anchors_sorted):
            if self.ratings[name] != last_elo:
                rank = i + 1
            last_elo = self.ratings[name]
            leaderboard_content.append([
                name,
                self.ratings[name],
                self.game_counts[name],
                rank
            ])
        return leaderboard_content

    def get_text(self):
        headers = ['name', 'elo', 'count', 'rank']
        table_data = self.leaderboard()
        text = tabulate(table_data, headers=headers, tablefmt='grid',
                        stralign='left', numalign='left')
        text = "\n" + text
        return text

    def get_agent_stat(self, agent_name):
        stat = {'elo': self.ratings[agent_name],
                'count': self.game_counts[agent_name]}
        return stat

    def get_scores(self, agent_names=None):
        if agent_names == None:
            agent_names = self.ratings.keys()
        agent_scores = [[self.ratings[n], self.game_counts[n]] for n in
                        agent_names]
        return agent_scores

    def get_win_probability(self, agent1, agent2):
        win_probability = self.elo_rating.computeScore(self.ratings[agent1], self.ratings[agent2])
        return win_probability

    def delete_agent(self, name=None):
        if name in self.ratings:
            del self.ratings[name]
        if name in self.game_counts:
            del self.game_counts[name]

    def check_agent(self, name=None):
        if name in self.ratings:
            return True
        else:
            return False

    def get_agent_num(self):
        return len(self.ratings)


def default_elo_rating(num):
    return num


if __name__ == '__main__':
    elo_system = EloSystem()
    agent_names = [f'MP{idx}' for idx in range(2)]
    ranks = [100, 99]
    for _ in range(1200):
        random.shuffle(ranks)
        elo_system.update(agent_names=agent_names, reward_list=ranks)
    print(elo_system.leaderboard())
    print(elo_system.get_text())

    print(elo_system.get_win_probability(agent_names[0], agent_names[1]))
    import pickle

    x = pickle.dumps(elo_system)
    y = pickle.loads(x)
