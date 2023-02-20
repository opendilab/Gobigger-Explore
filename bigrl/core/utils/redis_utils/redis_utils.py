import os
import random
import subprocess
import time

try:
    import redis
except:
    redis = None
import torch

REDIS_SERVER = os.path.join(os.path.dirname(__file__), 'redis-server')


def start_redis_server(redis_port, redis_address_dir=None, ip=None, redis_pass='4Str0ngP@ss'):
    if ip is None:
        ip = os.environ.get('SLURMD_NODENAME') if 'SLURMD_NODENAME' in os.environ else '127.0.0.1'
    torch.set_num_threads(1)
    cmd = [REDIS_SERVER, f'--port', str(redis_port), '--requirepass', str(redis_pass)]
    if ip == "127.0.0.1":
        cmd += ["--bind", "127.0.0.1"]
    proc = subprocess.Popen(cmd, )
    redis_conn = redis.StrictRedis(host='127.0.0.1', db=0, port=redis_port, password=redis_pass)
    success = False
    while True:
        try:
            success = redis_conn.ping()
        except redis.exceptions.ConnectionError:
            pass
        if success:
            print(f'successfully start redis server at port:{redis_port}')
            break
        else:
            time.sleep(1)
    if redis_address_dir is not None:
        try:
            os.makedirs(redis_address_dir)
        except FileExistsError:
            pass
        with open(os.path.join(redis_address_dir, f"{ip}:{redis_port}"), 'w') as f:
            f.write(f'ip:{ip}, redis_port:{redis_port}')
    return proc


def shutdown_redis(ip, redis_port, redis_pass='4Str0ngP@ss'):
    if redis is None:
        return True
    redis_conn = redis.StrictRedis(host=ip, db=0, port=redis_port, password=redis_pass)
    redis_conn.shutdown(nosave=True)
    print(f'close redis server at ip: {ip} port: {redis_port}!')


def get_redis_ip_port_connect(ip, redis_port, redis_pass='4Str0ngP@ss'):
    return redis.StrictRedis(host=ip, db=0, port=redis_port, password=redis_pass)


def get_redis_connect_from_address(redis_address_dir, try_times=0, redis_pass='4Str0ngP@ss'):
    while True:
        try:
            redis_addresses = os.listdir(redis_address_dir)
            if len(redis_addresses) > 0:
                break
            else:
                time.sleep(1)
        except FileNotFoundError:
            if try_times == 1:
                return None
            elif try_times > 1:
                try_times -= 1
            time.sleep(1)

    assert len(redis_addresses), 'redis addresses cant be empty'
    chosen_redis_address = random.choice(redis_addresses)
    host, port = chosen_redis_address.split(':')
    redis_connect = redis.StrictRedis(host=host,
                                      port=port,
                                      password=redis_pass)
    return redis_connect

def get_redis_address(redis_address_dir, try_times=0):
    while True:
        try:
            redis_address = os.listdir(redis_address_dir)
            if len(redis_address) > 0:
                break
            else:
                time.sleep(1)
        except FileNotFoundError:
            if try_times == 1:
                return None
            elif try_times > 1:
                try_times -= 1
            time.sleep(1)
    assert len(redis_address_dir), 'rpc addresses cant be empty'
    chosen_redis_address = random.choice(redis_address)
    ip, port = chosen_redis_address.split(':')
    return {'ip': ip, 'port': port}