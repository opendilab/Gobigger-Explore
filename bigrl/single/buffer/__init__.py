from .dqn_redis_buffer import RedisBuffer
from .max_use_redis_buffer import MaxUseRedisBuffer
from .queue_redis_buffer import QueueRedisBuffer


def get_buffer_class(type='max_use'):
    if type == 'max_use':
        return MaxUseRedisBuffer
    elif type == 'queue':
        return QueueRedisBuffer
    elif type == 'normal':
        return RedisBuffer
    else:
        return RedisBuffer
