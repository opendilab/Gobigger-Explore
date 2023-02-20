import platform

from bigrl.single.dataloader import get_dataloader_class


class RLDataLoader(object):
    def __init__(self, learner, ) -> None:
        self.whole_cfg = learner.whole_cfg
        self.dataloader_type = self.whole_cfg.learner.data.get('type', 'max_use')

        dataloader_class = get_dataloader_class(self.dataloader_type)
        self.dataloader = dataloader_class(learner)
        self.batch_size = self.dataloader.batch_size
        self.unroll_len = self.dataloader.unroll_len

    def get_data(self):
        return self.dataloader.get_data()

    def close(self):
        self.dataloader.close()
