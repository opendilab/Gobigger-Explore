from collections import defaultdict

import torch
from torch._six import inf
from torch.nn.utils import clip_grad_norm_, clip_grad_value_


def build_grad_clip(cfg):
    clip_type = cfg.get('type', 'clip_norm')
    clip_threshold = cfg.get('threshold', 1)
    clip_norm_type = cfg.get('norm_type', 2)
    if clip_norm_type == 'inf':
        clip_norm_type = inf
    return GradClipper(clip_type, clip_threshold, clip_norm_type)


class GradClipper(object):
    def __init__(self, clip_type, threshold, norm_type, begin_step=100, ignore_threshold=3):
        assert (clip_type in ['clip_value', 'none', 'clip_norm', 'momentum_norm'])
        self.clip_type = clip_type
        self.threshold = threshold
        self.norm_type = norm_type
        self.step = 0
        self.beta1 = 0.95
        self.beta2 = 0.999
        self.state = defaultdict(dict)
        if self.clip_type == 'momentum_norm':
            self.norm_mom = None
            self.flag = 0

    def apply(self, parameters):
        self.step += 1
        with torch.no_grad():
            if self.clip_type == 'none':
                if isinstance(parameters, torch.Tensor):
                    parameters = [parameters]
                parameters = list(filter(lambda p: p.grad is not None, parameters))
                norm_type = float(self.norm_type)
                total_norm = 0
                for p in parameters:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm.item() ** norm_type
                total_norm = total_norm ** (1. / norm_type)

            elif self.clip_type == 'momentum_norm':
                bias_correction1 = 1 - self.beta1 ** self.step
                if isinstance(parameters, torch.Tensor):
                    parameters = [parameters]
                parameters = list(filter(lambda p: p.grad is not None, parameters))
                norm_type = float(self.norm_type)
                total_norm = 0
                norm_scale = self.threshold
                for idx, p in enumerate(parameters):
                    g_norm = p.grad.data.norm(norm_type)
                    if self.norm_mom is not None:
                        m_norm = self.norm_mom[idx]
                        if g_norm < norm_scale * m_norm:
                            s_temp = 1.0
                        else:
                            s_temp = norm_scale * m_norm / (g_norm + 1e-6)
                    else:
                        s_temp = 1.0

                    p.grad.data.mul_(s_temp)
                if self.flag == 0:
                    self.norm_mom = []

                for idx, p in enumerate(parameters):
                    g_norm = p.grad.data.norm(norm_type)
                    if self.flag == 0:
                        self.norm_mom.append(float(g_norm))
                    else:
                        self.norm_mom[idx] = self.norm_mom[idx] * 0.99 + float(g_norm) * 0.01
                    total_norm += g_norm.item() ** norm_type
                total_norm = total_norm ** (1. / norm_type)
                self.flag = 1


            elif self.clip_type == 'clip_value':
                total_norm = clip_grad_value_(parameters, self.threshold)
                if isinstance(total_norm, torch.Tensor):
                    total_norm = total_norm.item()


            elif self.clip_type == 'clip_norm':
                total_norm = clip_grad_norm_(parameters, self.threshold, self.norm_type)
                if isinstance(total_norm, torch.Tensor):
                    total_norm = total_norm.item()

        return total_norm
