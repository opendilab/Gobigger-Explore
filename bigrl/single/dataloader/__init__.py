from .rl_max_use_redis_dataloader import RLDataLoader as MaxUseRedisDataloader
from .rl_queue_redis_dataloader import RLDataLoader as QueueRedisDataloader



def get_dataloader_class(type='max_use'):
    if type == 'max_use':
        return MaxUseRedisDataloader
    elif type == 'queue':
        return QueueRedisDataloader
    else:
        return MaxUseRedisDataloader
