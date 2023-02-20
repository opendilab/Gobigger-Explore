import itertools
import math
import random
from collections import defaultdict
from functools import partial
from typing import List, Union

try:
    import trueskill
except ImportError:
    trueskill = None
from tabulate import tabulate


class TrueSkillSystem:
    def __init__(self, cfg={},):
        self.cfg = cfg
        self.default_mu = self.cfg.get('default_mu', 25.000)
        self.default_sigma = self.cfg.get('default_sigma', 8.333)
        self.ratings = defaultdict(partial(trueskill.Rating, self.default_mu, self.default_mu))
        self.game_counts = defaultdict(int)
        self.update_count = 0

    def update(self, agent_names, rank_list, ):
        rating_groups = [[self.ratings[n]] for n in agent_names]
        result_tmp = trueskill.rate(rating_groups=rating_groups, ranks=rank_list)
        for i, n in enumerate(agent_names):
            self.ratings[n] = result_tmp[i][0]
        for name in agent_names:
            self.game_counts[name] += 1
        self.update_count += 1

    def leaderboard(self):
        leaderboard_content = []
        rank = 1
        last_skill = -1
        anchors = list(self.ratings.items())
        anchors_sorted = sorted(anchors, key=lambda x: x[1], reverse=True)
        for i, (name, _) in enumerate(anchors_sorted):
            if self.ratings[name].mu != last_skill:
                rank = i + 1
            last_skill = self.ratings[name].mu
            leaderboard_content.append([
                name,
                self.ratings[name].mu,
                self.ratings[name].sigma,
                self.game_counts[name],
                rank
            ])
        return leaderboard_content

    def get_text(self):
        headers = ['name', 'mu', 'sigma', 'count','rank']
        table_data = self.leaderboard()
        text = tabulate(table_data, headers=headers, tablefmt='grid',
                        stralign='left', numalign='left')
        text = "\n" + text
        return text

    def get_agent_stat(self, agent_name):
        stat = {'mu':self.ratings[agent_name].mu,
                'sigma':self.ratings[agent_name].sigma,
                'count': self.game_counts[agent_name]}
        return stat

    def get_scores(self, agent_names=None):
        if agent_names == None:
            agent_names = self.ratings.keys()
        agent_scores = [[self.ratings[n].mu, self.ratings[n].sigma, self.game_counts[n]] for n in
                        agent_names]
        return agent_scores

    def get_win_probability(self, team1: Union[List[str], str], team2: Union[List[str], str]):
        if isinstance(team1, str):
            team1 = [team1]
        if isinstance(team2, str):
            team2 = [team2]

        for n in itertools.chain(team1, team2):
            if n not in self.ratings:
                print(f'{n} not in rating system')
                return

        delta_mu = sum(self.ratings[n].mu for n in team1) - sum(self.ratings[n].mu for n in team2)
        sum_sigma = sum(self.ratings[n].sigma ** 2 for n in itertools.chain(team1, team2))
        size = len(team1) + len(team2)
        denom = math.sqrt(size * (trueskill.BETA * trueskill.BETA) + sum_sigma)
        ts = trueskill.global_env()
        return ts.cdf(delta_mu / denom)

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


if __name__ == '__main__':
    trueskill_system = TrueSkillSystem()
    agent_names = [f'MP{idx}' for idx in range(3)]
    ranks = [100, 99, 101]
    for _ in range(100):
        random.shuffle(ranks)
        trueskill_system.update(agent_names=agent_names, rank_list=ranks)
    print(trueskill_system.leaderboard())
    print(trueskill_system.get_text())

    print(trueskill_system.get_win_probability(agent_names[0], agent_names[1]))
    # import pickle
    # x = pickle.dumps(trueskill_system)
    # y = pickle.loads(x)
