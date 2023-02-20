import importlib
import os
from typing import List


MODULE_PATHS = {
    'Agent': 'agent',
    'Model': 'model.model',
    'League': 'league',
    'make_env': 'env.env',
}


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

if __name__ == '__main__':
    Model = import_pipeline_module('default', 'Model')
