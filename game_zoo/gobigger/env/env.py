import os

from bigrl.core.utils.config_helper import read_config, deep_merge_dicts
from gobigger.envs import GoBiggerEnv

default_config = read_config(os.path.join(os.path.dirname(__file__), 'default_env_config.yaml'))


def make_env(env_cfg=None):
    if env_cfg is None:
        env_cfg = default_config.env
    else:
        env_cfg = deep_merge_dicts(default_config.env, env_cfg)
    step_mul = env_cfg.get('step_mul', 5)
    env = GoBiggerEnv(env_cfg,step_mul=step_mul)
    return env


if __name__ == '__main__':
    env = make_env()
    obs = env.reset()
    obs, rew, done, info = env.step(0)
