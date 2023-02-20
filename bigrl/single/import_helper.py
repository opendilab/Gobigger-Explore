import importlib
import os
from typing import List


def import_module(modules: List[str]) -> None:
    """
    Overview:
        Import several module as a list
    Args:
        - modules (:obj:`list` of `str`): List of module names
    """
    for name in modules:
        importlib.import_module(name)


MODULE_PATHS = {
    'RLLearner': 'rl_learner',
    'SLLearner': 'sl_learner',
    'Agent': 'agent',
    'Model': 'model.model',
    'Actor': 'actor',
    'SLActor': 'sl_actor',
    'ReinforcementLoss': 'loss.rl_loss',
    'SupervisedLoss': 'loss.sl_loss',
    'RLDataLoader': 'rl_dataloader',
    'SLDataLoader': 'sl_dataloader',
    'League': 'league',
    'ReplayDecoder': 'replay_decoder',
    'Features': 'features',
    'send_data': 'send_data',
    'make_env': 'env.env',
}

DEFAULT_MODULE_PATH_NAME = {
    'RLLearner': ['bigrl.single.worker.learner.rl_learner', 'BaseRLLearner'],
    'SLLearner': ['bigrl.single.worker.learner.sl_learner', 'BaseSLLearner'],
    'Actor': ['bigrl.single.worker.actor.actor', 'Actor'],
    'ReinforcementLoss': ['bigrl.single.policy.impala.rl_loss', 'ReinforcementLoss'],
    'SupervisedLoss': ['bigrl.single.worker.learner.sl_loss', 'SupervisedLoss'],
    'RLDataLoader': ['bigrl.single.worker.learner.rl_dataloader', 'RLDataLoader'],
    'SLDataLoader': ['bigrl.single.worker.learner.sl_dataloader', 'SLDataLoader'],
    'League': ['bigrl.single.worker.league.base_league', 'BaseLeague'],
    'send_data': ['bigrl.single.worker.learner.send_data', 'send_data'],
}

DEFAULT_POLICY_MODULE_PATH_NAME = {
    'Agent': ['bigrl.single.policy', 'agent', 'BaseAgent'],
    'ReinforcementLoss': ['bigrl.single.policy', 'rl_loss', 'ReinforcementLoss'],
    'SupervisedLoss': ['bigrl.single.policy','sl_loss', 'SupervisedLoss'],
    'RLLearner': ['bigrl.single.policy','rl_learner', 'RLLearner'],
    'RLDataLoader': ['bigrl.single.policy', 'rl_dataloader', 'RLDataLoader'],
}
def import_pipeline_agent(env_name, pipeline, name='Agent'):
    try:
        abs_path = f'game_zoo.{env_name}.pipelines.{pipeline}.{MODULE_PATHS[name]}'
        module = getattr(importlib.import_module(abs_path), name)
        return module
    except Exception as e:
        pass
        # import traceback
        # print(f'[Import Error]{e}', flush=True)
        # print(''.join(traceback.format_tb(e.__traceback__)), flush=True)

    try:
        dir_path, file_name, module_name = DEFAULT_POLICY_MODULE_PATH_NAME[name]
        abs_path = '.'.join([dir_path, pipeline, file_name])
        module = getattr(importlib.import_module(abs_path),module_name)
        return module
    except Exception as e:
        pass
        # import traceback
        # print(f'[Import Error]{e}', flush=True)
        # print(''.join(traceback.format_tb(e.__traceback__)), flush=True)
    if name in DEFAULT_MODULE_PATH_NAME.keys():
        abs_path, module_name = DEFAULT_MODULE_PATH_NAME[name]
        module = getattr(importlib.import_module(abs_path), module_name)
        print(f'use default module {module_name} from {abs_path}')
        return module
    else:
        print(f'cant load module {name} for env {env_name}')
        raise ImportError

    return module


def import_pipeline_module(env_name, pipeline, name, ):
    try:
        abs_path = f'game_zoo.{env_name}.pipelines.{pipeline}.{MODULE_PATHS[name]}'
        module = getattr(importlib.import_module(abs_path), name)
        return module
    except Exception as e:
        import traceback
        print(f'[Import Error]{e}', flush=True)
        print(''.join(traceback.format_tb(e.__traceback__)), flush=True)
        # pass
    return import_env_module(env_name, name)


def import_env_module(env_name, name, ) -> None:
    """
    Overview:
        Import several module as a list
    Args:
        - modules (:obj:`list` of `str`): List of module names
    """
    try:
        abs_path = f'game_zoo.{env_name}.{MODULE_PATHS[name]}'
        module = getattr(importlib.import_module(abs_path), name)
        return module
    except Exception as e:
        import traceback
        print(f'[Import Error]{e}', flush=True)
        print(''.join(traceback.format_tb(e.__traceback__)), flush=True)
        # pass
    if name in DEFAULT_MODULE_PATH_NAME.keys():
        abs_path, module_name = DEFAULT_MODULE_PATH_NAME[name]
        module = getattr(importlib.import_module(abs_path), module_name)
        print(f'use default module {module_name} from {abs_path}')
        return module
    else:
        print(f'cant load module {name} for env {env_name}')
        raise ImportError


if __name__ == '__main__':
    Model = import_pipeline_module('default', 'Model')
