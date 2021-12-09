import os
import numpy as np
import copy
from tensorboardX import SummaryWriter
import sys
sys.path.append('..')

from ding.config import compile_config
from ding.worker import BaseLearner, BattleSampleSerialCollector, BattleInteractionSerialEvaluator, NaiveReplayBuffer
from ding.envs import SyncSubprocessEnvManager, BaseEnvManager
from policy.gobigger_policy import DQNPolicy
from ding.utils import set_pkg_seed
from gobigger.agents import BotAgent

from envs import GoBiggerSimpleEnv
from model import GoBiggerHybridActionSimple
from config.gobigger_no_spatial_config import main_config
import torch
import argparse

class RulePolicy:

    def __init__(self, team_id: int, player_num_per_team: int):
        self.collect_data = False  # necessary
        self.team_id = team_id
        self.player_num = player_num_per_team
        start, end = team_id * player_num_per_team, (team_id + 1) * player_num_per_team
        self.bot = [BotAgent(str(i)) for i in range(start, end)]

    def forward(self, data: dict, **kwargs) -> dict:
        ret = {}
        for env_id in data.keys():
            action = []
            for bot, raw_obs in zip(self.bot, data[env_id]['collate_ignore_raw_obs']):
                raw_obs['overlap']['clone'] = [[x[0], x[1], x[2], int(x[3]), int(x[4])]  for x in raw_obs['overlap']['clone']]
                action.append(bot.step(raw_obs))
            ret[env_id] = {'action': np.array(action)}
        return ret

    def reset(self, data_id: list = []) -> None:
        pass

def main(cfg, ckpt_path, seed=0):

    # Evaluator Setting 
    cfg.exp_name = 'gobigger_vsbot_eval'
    cfg.env.spatial = True  # necessary
    cfg.env.evaluator_env_num = 3
    cfg.env.n_evaluator_episode = 3

    cfg = compile_config(
        cfg,
        BaseEnvManager,
        DQNPolicy,
        BaseLearner,
        BattleSampleSerialCollector,
        BattleInteractionSerialEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    
    evaluator_env_num = cfg.env.evaluator_env_num

    rule_env_cfgs = []
    for i in range(evaluator_env_num):
        rule_env_cfg = copy.deepcopy(cfg.env)
        rule_env_cfg.train = False
        rule_env_cfg.save_video = True
        rule_env_cfg.save_quality = 'low'
        rule_env_cfg.save_path = './{}/rule'.format(cfg.exp_name)
        if not os.path.exists(rule_env_cfg.save_path):
            os.makedirs(rule_env_cfg.save_path)
        rule_env_cfgs.append(rule_env_cfg)

    print(rule_env_cfgs)
    rule_evaluator_env = BaseEnvManager(
        env_fn=[lambda: GoBiggerSimpleEnv(x) for x in rule_env_cfgs], cfg=cfg.env.manager
    )

    rule_evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = GoBiggerHybridActionSimple(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)
    policy.eval_mode.load_state_dict(torch.load(ckpt_path))
    team_num = cfg.env.team_num
    rule_eval_policy = [RulePolicy(team_id, cfg.env.player_num_per_team) for team_id in range(1, team_num)]

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))

    rule_evaluator = BattleInteractionSerialEvaluator(
        cfg.policy.eval.evaluator,
        rule_evaluator_env, [policy.eval_mode] + rule_eval_policy,
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='rule_evaluator'
    )
    rule_evaluator.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--ckpt', help='checkpoint for evaluation')
    args = parser.parse_args()
    main(main_config,ckpt_path = args.ckpt)