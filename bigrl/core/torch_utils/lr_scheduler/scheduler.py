import math
from bisect import bisect_right

import torch


class _LRScheduler(object):
    def __init__(self, optimizer, last_iter=0):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_iter == 0:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_iter

    def get_last_lr(self):
        return self._last_lr

    def _get_new_lr(self):
        raise NotImplementedError

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.last_epoch + 1
        self.last_epoch = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_new_lr()):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class _WarmUpLRScheduler(_LRScheduler):
    r"""Adjust learning rate with warmup strategy.
    Set the learning rate as a linear function of the minibatch size and apply a simple
    warmup phase for the first few epochs of training.
    Paper: Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour
    Link: https://arxiv.org/pdf/1706.02677.pdf

    Arguments:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float): initial learning rate.
        warmup_lr (float): target learning rate after warmup.
        warmup_steps (int): total interations of warmup phrase.
        last_iter (int): the index of last iteration.

    Example:
    # >>> # Assuming optimizer uses lr = 0.1 for all groups
    # >>> # base_lr = 0.1
    # >>> # warmup_lr = 0.4
    # >>> # warmup_steps = 2500
    # >>> # ...
    # >>> scheduler = _WarmUpLRScheduler(optimizer, base_lr = 0.1, warmup_lr = 0.4, warmup_steps = 2500)
    # >>> for iteration in range(max_iteration):
    # >>>     scheduler.step(iteration)
    # >>>     scheduler.get_lr()[0]

    """

    def __init__(self, optimizer, base_lr, warmup_lr, warmup_steps, last_iter=0):
        assert warmup_steps >= 2 or warmup_steps == 0
        if warmup_steps == 0:
            assert base_lr == warmup_lr
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        if warmup_steps == 0:
            self.warmup_lr = base_lr
        else:
            self.warmup_lr = warmup_lr
        super(_WarmUpLRScheduler, self).__init__(optimizer, last_iter)

    def _get_warmup_lr(self):
        if self.warmup_steps >= 2 and self.last_epoch < self.warmup_steps:
            target_lr = (self.warmup_lr - self.base_lr) / (self.warmup_steps - 1) * (self.last_epoch - 1) + self.base_lr
            scale = target_lr / self.base_lr
            return [scale * base_lr for base_lr in self.base_lrs]
        else:
            return None


class StepLRScheduler(_WarmUpLRScheduler):
    r"""Decays the learning rate of each parameter group by lr_multi every
    step_size iterations.

    Arguments:
        - optimizer (:obj:`Optimizer`): Wrapped optimizer.
        - lr_steps (:obj:`list`): milestones to adjust learning rate.
        - lr_mults (:obj:`list`): coefficients to decay learning rate.
        - base_lr (:obj:`float`): initial learning rate.
        - warmup_lr (:obj:`float`): target learning rate after warmup.
        - warmup_steps (:obj:`int`): total interations of warmup phrase.
        - max_iter (:obj:`int`): maximum/total interrations of training.
        - last_iter (:obj:`int`): the index of last iteration.

    Example:
        # >>> # Assuming optimizer uses lr = 0.1 for all groups
        # >>> # base_lr = 0.1
        # >>> # warmup_lr = 0.4
        # >>> # warmup_steps = 2500
        # >>> # lr_steps = [37500, 75000, 112500]
        # >>> # lr_mults = [0.1, 0.1, 0.1]
        # >>> # max_iter = 125000
        # >>> # ...
        # >>> scheduler = StepLRScheduler(optimizer, [37500, 75000, 112500], [0.1, 0.1, 0.1],
        # >>>                 base_lr = 0.1, warmup_lr = 0.4, max_iter = 125000, warmup_steps = 2500)
        # >>> for iteration in range(max_iteration):
        # >>>     scheduler.step(iteration)
        # >>>     scheduler.get_lr()[0]

    """

    def __init__(self, optimizer, lr_steps, lr_mults, base_lr, warmup_lr, warmup_steps, max_iter, last_iter=0):
        super(StepLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_steps, last_iter)

        assert len(lr_steps) == len(lr_mults), "{} vs {}".format(lr_steps, lr_mults)
        for x in lr_steps:
            assert isinstance(x, int)
        if not list(lr_steps) == sorted(lr_steps):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                lr_steps,
            )
        self.lr_steps = lr_steps
        self.lr_mults = [1.0]
        for x in lr_mults:
            self.lr_mults.append(self.lr_mults[-1] * x)

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        pos = bisect_right(self.lr_steps, self.last_epoch)
        scale = self.warmup_lr * self.lr_mults[pos] / self.base_lr
        return [base_lr * scale for base_lr in self.base_lrs]


Step = StepLRScheduler


class StepDecayLRScheduler(_WarmUpLRScheduler):
    r"""Decays the learning rate of each parameter group by decay every
    definited step_size iterations.
    Paper: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
    Link: https://arxiv.org/abs/1905.11946

    Arguments:
        - optimizer (:obj:`Optimizer`): Wrapped optimizer.
        - step_size (:obj:`int`): iterations to decay.
        - decay (:obj:`float`): coefficients to decay learning rate.
        - base_lr (:obj:`float`): initial learning rate.
        - warmup_lr (:obj:`float`): target learning rate after warmup.
        - warmup_steps (:obj:`int`): total interations of warmup phrase.
        - max_iter (:obj:`int`): maximum/total interrations of training.
        - last_iter (:obj:`int`): the index of last iteration.

    Example:
        # >>> # Assuming optimizer uses lr = 0.1 for all groups
        # >>> # base_lr = 0.1
        # >>> # warmup_lr = 0.4
        # >>> # warmup_steps = 2500
        # >>> # step_size = 3500
        # >>> # decay = 0.97
        # >>> # max_iter = 437500
        # >>> # ...
        # >>> scheduler = StepDecayLRScheduler(optimizer, 3500, 0.97,
        # >>>                 base_lr = 0.1, warmup_lr = 0.4, max_iter = 437500, warmup_steps = 2500)
        # >>> for iteration in range(max_iteration):
        # >>>     scheduler.step(iteration)
        # >>>     scheduler.get_lr()[0]

    """

    def __init__(self, optimizer, step_size, decay, base_lr, warmup_lr, warmup_steps, max_iter, last_iter=0):
        super(StepDecayLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_steps, last_iter)

        self.step_size = step_size
        self.decay = decay

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        num = (self.last_epoch - self.warmup_steps) // self.step_size
        scale = self.decay ** num * self.warmup_lr / self.base_lr
        return [base_lr * scale for base_lr in self.base_lrs]


StepDecay = StepDecayLRScheduler


class CosineLRScheduler(_WarmUpLRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule.

    Arguments:
        - optimizer (:obj:`Optimizer`): Wrapped optimizer.
        - max_iter (:obj:`int`): maximum/total interrations of training.
        - min_lr (:obj:`float`): minimum learning rate. Default: 0.
        - base_lr (:obj:`float`): initial learning rate.
        - warmup_lr (:obj:`float`): target learning rate after warmup.
        - warmup_steps (:obj:`int`): total interations of warmup phrase.
        - last_iter (:obj:`int`): the index of last iteration.

    Example:
        # >>> # Assuming optimizer uses lr = 0.1 for all groups
        # >>> # base_lr = 0.1
        # >>> # warmup_lr = 0.4
        # >>> # warmup_steps = 2500
        # >>> # min_lr = 0.0
        # >>> # max_iter: 125000
        # >>> # ...
        # >>> scheduler = CosineLRScheduler(optimizer, 125000, 0.0,
        # >>>                 base_lr = 0.1, warmup_lr = 0.4, max_iter = 125000, warmup_steps = 2500)
        # >>> for iteration in range(max_iteration):
        # >>>     scheduler.step(iteration)
        # >>>     scheduler.get_lr()[0]
    """

    def __init__(self, optimizer, max_iter, min_lr, base_lr, warmup_lr, warmup_steps, last_iter=0):
        super(CosineLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_steps, last_iter)
        self.max_iter = max_iter
        self.min_lr = min_lr

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        step_ratio = (self.last_epoch - self.warmup_steps) / (self.max_iter - self.warmup_steps)
        target_lr = self.min_lr + (self.warmup_lr - self.min_lr) * (1 + math.cos(math.pi * step_ratio)) / 2
        scale = target_lr / self.base_lr
        return [scale * base_lr for base_lr in self.base_lrs]


Cosine = CosineLRScheduler


class CosineLREpochScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, max_epoch, max_iter, min_lr, base_lr, warmup_lr, warmup_steps, last_iter=0):
        super(CosineLREpochScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_steps, last_iter)
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.max_epoch = max_epoch

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        step_per_epoch = self.max_iter // self.max_epoch
        last_epoch = self.last_epoch // step_per_epoch
        epoch_ratio = last_epoch / self.max_epoch
        target_lr = self.min_lr + (self.warmup_lr - self.min_lr) * (1 + math.cos(math.pi * epoch_ratio)) / 2
        scale = target_lr / self.base_lr

        return [scale * base_lr for base_lr in self.base_lrs]


eCosine = CosineLREpochScheduler


class PolynomialLRScheduler(_WarmUpLRScheduler):
    r"""Set the learning rate of each parameter group using a polynomial annealing
    schedule.
    Paper: Large Batch Training of Convolutional Networks
    Link: https://arxiv.org/abs/1708.03888

    Args:
        - optimizer (:obj:`Optimizer`): Wrapped optimizer.
        - power (:obj:`float`): polynomial coefficient.
        - max_iter (:obj:`int`): maximum/total interrations of training.
        - base_lr (:obj:`float`): initial learning rate.
        - warmup_lr (:obj:`float`): target learning rate after warmup.
        - warmup_steps (:obj:`int`): total interations of warmup phrase.
        - last_iter (:obj:`int`): the index of last iteration.

    Example:
        # >>> # Assuming optimizer uses lr = 0.1 for all groups
        # >>> # power = 2.0
        # >>> # base_lr = 0.1
        # >>> # warmup_lr = 0.4
        # >>> # warmup_steps = 2500
        # >>> # max_iter: 125000
        # >>> # ...
        # >>> scheduler = PolynomialLRScheduler(optimizer, 2.0,
        # >>>                 base_lr = 0.1, warmup_lr = 0.4, max_iter = 125000, warmup_steps = 2500)
        # >>> for iteration in range(max_iteration):
        # >>>     scheduler.step(iteration)
        # >>>     scheduler.get_lr()[0]

    """

    def __init__(self, optimizer, power, max_iter, base_lr, warmup_lr, warmup_steps, last_iter=0):
        super(PolynomialLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_steps, last_iter)
        self.max_iter = max_iter
        self.power = power

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        factor = (1 - (self.last_epoch - self.warmup_steps) / float(self.max_iter)) ** self.power
        target_lr = factor * self.warmup_lr
        scale = target_lr / self.base_lr
        return [scale * base_lr for base_lr in self.base_lrs]


Poly = PolynomialLRScheduler


class MultiStepLRScheduler(_WarmUpLRScheduler):
    def __init__(
            self,
            optimizer,
            max_iter,
            interval,
            lr_mults,
            base_lr,
            warmup_lr,
            warmup_steps,
            last_iter=-1,
    ):
        super(MultiStepLRScheduler, self).__init__(
            optimizer, base_lr, warmup_lr, warmup_steps, last_iter
        )
        self.interval = interval  # step interval
        self.max_iter = max_iter
        self.milestones = []
        self.lr_mults = [1.0]
        for i in range(1, math.ceil(max_iter / interval)):
            self.milestones.append(interval * i + self.warmup_steps)
            self.lr_mults.append(self.lr_mults[-1] * lr_mults)

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        pos = bisect_right(self.milestones, self.last_epoch)
        scale = self.warmup_lr * self.lr_mults[pos] / self.base_lr
        return [base_lr * scale for base_lr in self.base_lrs]


MultiStep = MultiStepLRScheduler


class CosineCycleLRScheduler(_LRScheduler):
    def __init__(
            self,
            optimizer,
            lr_steps,
            eta_min,
            base_lr,
            warmup_lr,
            warmup_steps,
            cycles=1,
            last_iter=-1,
            **kwargs
    ):
        super(CosineCycleLRScheduler, self).__init__(optimizer, last_iter)
        # lr_steps: list or int, steps for each cycle
        # eta_min: list or float, min lr for each cycle
        # warmup_steps: list or int, warmup steps for each cycle
        # warmup_lr: list or float, warmup lr for each cycle
        if isinstance(lr_steps, int):
            lr_steps = [lr_steps] * cycles
            print("new lr_steps for each cycle: {}".format(lr_steps))
        else:
            assert isinstance(lr_steps, list), "lr_steps should be int or list"

        if isinstance(eta_min, float) or isinstance(eta_min, int):
            eta_min = [eta_min] * cycles
            print("new eta_min for each cycle: {}".format(eta_min))
        else:
            assert isinstance(eta_min, list), "eta_min should be int/float or list"

        if isinstance(warmup_steps, int):
            warmup_steps = [warmup_steps] * cycles
            print("new warmup_steps for each cycle: {}".format(warmup_steps))
        else:
            assert isinstance(warmup_steps, list), "warmup_steps should be int or list"

        if isinstance(warmup_lr, float):
            warmup_lr = [warmup_lr] * cycles
            print("[warning]new warmup_lr for each cycle: {}".format(warmup_lr))
        else:
            assert isinstance(warmup_lr, list), "warmup_lr should be float or list"

        self.optimizer = optimizer
        self.lr_steps = lr_steps
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.cycles = cycles

        self.current_stage = 0
        step_count = 0
        for step in self.lr_steps:
            step_count += step
            if last_iter >= step_count:
                self.current_stage += 1
        self._update_stage(self.current_stage)

    def _update_stage(self, stage_iter):
        # optimizer, max_iter, eta_min, base_lr, warmup_lr, warmup_steps, last_iter=-1

        max_iter = self.lr_steps[stage_iter]
        eta_min = self.eta_min[stage_iter]
        if stage_iter == 0:
            base_lr = self.base_lr
        else:
            # base_lr = self.scheduler._get_new_lr()[0]
            base_lr = self.eta_min[stage_iter - 1]
        warmup_lr = self.warmup_lr[stage_iter]
        warmup_steps = self.warmup_steps[stage_iter]

        if stage_iter == 0:
            last_iter = self.last_epoch
        else:
            last_iter = self.last_epoch - sum(self.lr_steps[:stage_iter]) - 1
        # import pdb; pdb.set_trace()

        self.scheduler = CosineLRScheduler(
            self.optimizer,
            max_iter,
            eta_min,
            base_lr,
            warmup_lr,
            warmup_steps,
            last_iter,
        )

    def _get_new_lr(self):
        if self.last_epoch >= sum(self.lr_steps[: self.current_stage + 1]):
            self.current_stage += 1
            self._update_stage(self.current_stage)
        return self.scheduler._get_new_lr()

    def get_lr(self):
        return self.scheduler.get_lr()

    def step(self, this_iter=None):
        # if not self.scheduler.has_base_lrs:
        #     # called when optimizer is recovered after lr_scheduler
        #     # then at __init__ there is no 'initial_lr'
        #     self.scheduler._get_base_lrs_later()

        if this_iter is None:
            this_iter = self.last_epoch + 1
        self.last_epoch = this_iter
        # for param_group, lr in zip(self.optimizer.param_groups, self._get_new_lr()):
        #     param_group['lr'] = lr
        if self.last_epoch >= sum(self.lr_steps[: self.current_stage + 1]):
            self.current_stage += 1
            self._update_stage(self.current_stage)
        self.scheduler.step()


CosineCycle = CosineCycleLRScheduler


class LinearLRScheduler(_LRScheduler):
    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, max_iter=5, last_iter=0, ):
        super(LinearLRScheduler, self).__init__(optimizer, last_iter)

        if start_factor > 1.0 or start_factor < 0:
            raise ValueError('Starting multiplicative factor expected to be between 0 and 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError('Ending multiplicative factor expected to be between 0 and 1.')

        self.max_iter = max_iter
        self.start_factor = start_factor
        self.end_factor = end_factor

    def _get_new_lr(self):
        if self.last_epoch == 0:
            scale = self.start_factor
        elif self.last_epoch > self.max_iter:
            scale = self.end_factor
        else:
            scale = (self.start_factor +
                (self.end_factor - self.start_factor) * min(self.max_iter, self.last_epoch) / self.max_iter)
        return [scale * base_lr for base_lr in self.base_lrs]


Linear = LinearLRScheduler

class ProgressLRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_iter=0, ):
        super(ProgressLRScheduler, self).__init__(optimizer, last_iter)
        self._progress_remaining = 1

    def _get_new_lr(self):
        scale = self._progress_remaining
        return [scale * base_lr for base_lr in self.base_lrs]

    def set_progress(self, progress_remaining):
        self._progress_remaining = progress_remaining

Progress = ProgressLRScheduler

if __name__ == '__main__':
    optimizer = torch.optim.SGD([torch.rand((2, 2), requires_grad=True)], lr=1)
    scheduler = ProgressLRScheduler(optimizer, )

    for epoch in range(101):
        scheduler.set_progress(epoch/100)
        scheduler.step()
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        print(scheduler.get_last_lr())
