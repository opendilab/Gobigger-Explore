from .scheduler import Step, StepDecay, Cosine, Poly, eCosine, CosineCycle, MultiStep,Linear,Progress # noqa F401
from copy import deepcopy
import torch

# def lr_scheduler_entry(config):
#     return globals()[config["type"] + "LRScheduler"](**config["kwargs"])

def build_lr_scheduler(cfg,optimizer):
    if cfg and cfg.get('type',None):
        cfg = deepcopy(cfg)
        lr_scheduler_type = cfg.pop('type')
        return globals()[lr_scheduler_type](optimizer=optimizer,**cfg)
    else:
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1)
