import sys
import uuid

import redis
import time
from bigrl.core.utils.file_helper import dumps,loads

from bigrl.core.utils.redis_utils import start_redis_server, shutdown_redis


class MaxUseRedisBuffer:
    def __init__(self, cfg, type='server'):
        self.whole_cfg = cfg
        self.type = type  # chosen from ['sender','server','receiver']

        # communication setting
        self.redis_ip = self.whole_cfg.communication.get('redis_ip','127.0.0.1')
        self.redis_port = self.whole_cfg.communication.redis_port
        self.redis_pass = self.whole_cfg.communication.get('redis_pass','4Str0ngP@ss')
        self.traj_fs_type = self.whole_cfg.communication.get('traj_fs_type','nppickle')

        # buffer setting
        self.max_use = self.whole_cfg.learner.data.get('max_use', 3)
        self.max_buffer_size = self.whole_cfg.learner.data.get('max_buffer_size', 2)
        self.start_sample_size = self.whole_cfg.learner.data.get('start_sample_size', 2)
        self.min_sample_size = self.whole_cfg.learner.data.get('min_sample_size', 1)

    def launch(self):
        if self.type == 'server':
            start_redis_server(ip=self.redis_ip,redis_port=self.redis_port, redis_pass=self.redis_pass)
        elif self.type == 'sender':
            self.setup_redis()
            self.setup_sender()
        elif self.type == 'receiver':
            self.setup_redis()
            self.setup_receiver()
        else:
            print(f"dont support type:{self.type}")
            raise NotImplementedError

    def setup_redis(self):
        self.redis_connect = redis.StrictRedis(host=self.redis_ip, db=0, port=self.redis_port,
                                               password=self.redis_pass)

    def setup_sender(self):
        self.sender_lua_script = """
                    local collect_data_size =  tonumber(redis.call('incr', 'collect_data_size'))
                    if collect_data_size > tonumber(KEYS[4]) then
                        local buffer_size = tonumber(redis.call('scard', 'traj_paths'))
                        if buffer_size >= tonumber(KEYS[5]) then
                            local randkey = redis.call('srandmember', 'traj_paths')
                            redis.call('del',randkey)
                            redis.call('srem', 'traj_paths', randkey)
                            redis.call('hdel', 'counts', randkey)
                        end
                        redis.call('set', KEYS[1], KEYS[2])
                        redis.call('sadd','traj_paths', KEYS[1])
                        redis.call('hset','counts', KEYS[1],KEYS[3])
                    end
                    """
        self.sender_cmd = self.redis_connect.register_script(self.sender_lua_script)

    def setup_receiver(self):
        self.receiver_lua_script = """
                local buffer_size = tonumber(redis.call('scard', 'traj_paths'))
                local min_sample_size = tonumber(KEYS[1])
                if buffer_size < min_sample_size then
                    return 0
                end
                local randkey = redis.call('srandmember', 'traj_paths')
                local return_data = redis.call('get',randkey)
                local count = tonumber(redis.call('hget', 'counts', randkey))
                if count == 1 then
                    redis.call('del',randkey)
                    redis.call('srem', 'traj_paths', randkey)
                    redis.call('hdel', 'counts', randkey)
                else
                    redis.call('hset', 'counts', randkey, count-1)
                end
                return return_data
                """
        self.receiver_cmd = self.redis_connect.register_script(self.receiver_lua_script)

    def push(self, data):
        traj_path = str(uuid.uuid1())
        trajdata = dumps(data, fs_type=self.traj_fs_type, compress=True)
        try:
            self.sender_cmd([traj_path, trajdata, self.max_use, self.start_sample_size, self.max_buffer_size])
        except Exception as e:
            import traceback
            print(f'[receiver redis Error]{e}', flush=True)
            print(''.join(traceback.format_tb(e.__traceback__)), flush=True)
        return True

    def get_data(self,):
        while True:
            try:
                get_data = self.receiver_cmd([self.min_sample_size])
            except Exception as e:
                import traceback
                print(f'[sender redis Error]{e}', flush=True)
                print(''.join(traceback.format_tb(e.__traceback__)), flush=True)
                if 'Connection refused' in f'{e}':
                    start_redis_server(ip=self.redis_ip, redis_port=self.redis_port, redis_pass=self.redis_pass)
                    get_data = None
            if not get_data:
                time.sleep(0.001)
            else:
                break
        trajdata = loads(get_data, fs_type=self.traj_fs_type, compress=True)
        return trajdata

    def close(self):
        shutdown_redis(ip=self.redis_ip,redis_port=self.redis_port,redis_pass=self.redis_pass)

if __name__ == '__main__':
    from bigrl.core.utils import read_config
    cfg = read_config('default_buffer_config.yaml')
    server_buffer = MaxUseRedisBuffer(cfg, type='server')
    server_buffer.launch()

    buffer = MaxUseRedisBuffer(cfg, type='sender')
    buffer.launch()
    for i in range(10):
        buffer.push('123')


    receiver_buffer = MaxUseRedisBuffer(cfg, type='receiver')
    receiver_buffer.launch()
    data = receiver_buffer.get_data()
    print(data)

