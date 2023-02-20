from typing import Union, Any, List

from .random_acess_queue import RandomAccessQueue


class ReplayBuffer:
    """Experience Replay Buffer

    As described in
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf.

    Args:
        capacity (int): capacity in terms of number of transitions
        num_steps (int): Number of timesteps per stored transition
            (for N-step updates)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.max_buffer_size = self.cfg.get('max_buffer_size', 100)
        self.batch_size = self.cfg.batch_size
        self.memory = RandomAccessQueue(maxlen=self.max_buffer_size)

    def push_data(self, data: Union[List[Any], Any]) -> None:
        r"""
        Overview:
            Push a data into buffer.
        Arguments:
            - data (:obj:`Union[List[Any], Any]`): The data which will be pushed into buffer. Can be one \
                (in `Any` type), or many(int `List[Any]` type).
            - cur_collector_envstep (:obj:`int`): Collector's current env step. \
                Not used in naive buffer, but preserved for compatiblity.
        """
        if isinstance(data, list):
            self.extend(data)
        else:
            self.append(data)

    def append(self, data, **kwargs):
        self.memory.append(data)

    def available_sample(self, batch_size):
        return len(self.memory) >= batch_size

    def extend(self, datas, **kwargs):
        self.memory.extend(datas)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if not self.available_sample(batch_size):
            return None
        else:
            data = self.memory.sample(batch_size)
            return data

    def __len__(self):
        return len(self.memory)
    
    def reset(self):
        del self.memory
        self.memory = RandomAccessQueue(maxlen=self.max_buffer_size)
    
    def clear(self):
        del self.memory
        self.memory = RandomAccessQueue(maxlen=self.max_buffer_size)
        
        
