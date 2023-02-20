from abc import abstractmethod

class BaseReplayDecoder:
    def __init__(self, cfg={}):
        self.whole_cfg = cfg
        self.launch()
        self.reset()

    def launch(self):
        pass

    @abstractmethod
    def decode_replay(self, path,):
        # yield path
        # return True
        raise NotImplementedError

    def reset(self,):
        pass

    def close(self):
        pass