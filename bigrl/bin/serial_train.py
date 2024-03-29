import argparse

from bigrl.core.utils import read_config
from bigrl.serial.import_helper import import_pipeline_worker

def get_args():
    parser = argparse.ArgumentParser(description="serial_train")
    parser.add_argument("--config", type=str,required=True)
    parser.add_argument("--type", type=str,default='train',)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    config = read_config(args.config)
    Learner = import_pipeline_worker(config.env.name,config.agent.pipeline, 'Learner')
    learner = Learner(config)
    learner.run()