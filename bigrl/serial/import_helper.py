import importlib
import warnings
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
    'Agent': 'agent',
    'Model': 'model.model',
    'Actor': 'actor',
    'Features': 'features',
    'make_env': 'env.env',
    'Learner': 'learner',
    'Evaluator': 'evaluator',
}


DEFAULT_POLICY_MODULE_PATH_NAME = {
    'Agent': ['bigrl.serial.policy', 'agent', 'BaseAgent'],
    'Learner': ['bigrl.serial.worker', 'learner', 'BaseLearner'],
    'Evaluator': ['bigrl.serial.worker', 'evaluator', 'BaseEvaluator'],
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

    dir_path, file_name, module_name = DEFAULT_POLICY_MODULE_PATH_NAME[name]
    abs_path = '.'.join([dir_path, pipeline, file_name])
    module = getattr(importlib.import_module(abs_path),module_name)
    return module

def import_pipeline_worker(env_name, pipeline, name='Learner'):
    try:
        abs_path = f'game_zoo.{env_name}.pipelines.{pipeline}.{MODULE_PATHS[name]}'
        module = getattr(importlib.import_module(abs_path), name)
        return module
    except Exception as e:
        pass
        # import traceback
        # print(f'[Import Error]{e}', flush=True)
        # print(''.join(traceback.format_tb(e.__traceback__)), flush=True)

    dir_path, file_name, module_name = DEFAULT_POLICY_MODULE_PATH_NAME[name]
    abs_path = '.'.join([dir_path, file_name])
    module = getattr(importlib.import_module(abs_path),module_name)
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
        raise ImportError


if __name__ == '__main__':
    Model = import_pipeline_module('default', 'Model')
