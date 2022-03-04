import sys
sys.path.append('..')
sys.path.append('.')
from easydict import EasyDict
import argparse
import yaml

from sl_learner import SLLearner


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.safe_load(f)
    config = EasyDict(config)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='')
    args = parser.parse_args()

    assert args.config != '', 'Please set config!'

    cfg = parse_config(args.config)
    sl_learner = SLLearner(cfg)
    sl_learner.train()
