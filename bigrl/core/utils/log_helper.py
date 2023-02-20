'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. log helper, used to help to save logger on terminal, tensorboard or save file.
    2. CountVar, to help counting number.
'''
import json
import logging
import numbers
import os
import sys
from collections import  deque

import numpy as np
import torch
import yaml
from tabulate import tabulate


class TextLogger(object):
    r"""
    Overview:
        Logger that save terminal output to file

    Interface: __init__, info
    """

    def __init__(self, path, name=None, level=logging.INFO):
        r"""
        Overview:
            initialization method, create logger.
        Arguments:
            - path (:obj:`str`): logger's save dir
            - name (:obj:`str`): logger's name
            - level (:obj:`int` or :obj:`str`): Set the logging level of logger, reference Logger class setLevel method.
        """
        if name is None:
            name = 'default_logger'
        # ensure the path exists
        os.makedirs(path,exist_ok=True)

        self.logger = self._create_logger(name, os.path.join(path, name + '.txt'), level=level)

    def _create_logger(self, name, path, level=logging.INFO):
        r"""
        Overview:
            create logger using logging
        Arguments:
            - name (:obj:`str`): logger's name
            - path (:obj:`str`): logger's save dir
        Returns:
            - (:obj`logger`): new logger
        """
        logger = logging.getLogger(name)
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format='[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
        if not logger.handlers:
            formatter = logging.Formatter('[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
            fh = logging.FileHandler(path, 'a')
            fh.setFormatter(formatter)
            logger.setLevel(level)
            logger.addHandler(fh)
            # handler = logging.StreamHandler()
            # handler.setFormatter(formatter)
            # logger.addHandler(handler)
            # logger.propagate = False
        return logger

    def info(self, s):
        r"""
        Overview:
            add message to logger
        Arguments:
            - s (:obj:`str`): message to add to logger
        Notes:
            you can reference Logger class in the python3 /logging/__init__.py
        """
        self.logger.info(s)

    def bug(self, s):
        r"""
        Overview:
            call logger.debug
        Arguments:
            - s (:obj:`str`): message to add to logger
        Notes:
            you can reference Logger class in the python3 /logging/__init__.py
        """
        self.logger.debug(s)

    def error(self, s):
        self.logger.error(s)

class ShareEmaMeter(object):
    def __init__(self, decay=0.99, warm_up_size=100,ignore_counts=0):
        assert (warm_up_size > 0)
        assert 0 <= decay <= 1
        self._decay = decay
        self._warm_up_size = warm_up_size
        self._ignore_counts = ignore_counts
        self._remain_ignores = ignore_counts
        self.reset()

    def reset(self):
        self._count = torch.tensor([0],dtype=torch.int).share_memory_()
        self._val = torch.tensor([0],dtype=torch.float).share_memory_()
        self._remain_ignores = self._ignore_counts

    def update(self, val):
        if self._remain_ignores > 0:
            self._remain_ignores -= 1
            return
        if isinstance(val, torch.Tensor):
            val = val.item()
        assert isinstance(val, numbers.Integral) or isinstance(val, numbers.Real)
        if self._count >= self._warm_up_size:
            self._val.copy_(self._decay * self._val + (1 - self._decay) * val)
        else:
            self._val.copy_((self._val * self._count + val) / (self._count + 1))
        self._count += 1

    @property
    def val(self):
        return self._val.item()

    @property
    def count(self):
        return self._count.item()

    @val.setter
    def val(self):
        raise NotImplementedError

    @count.setter
    def count(self):
        raise NotImplementedError

class ShareVariableRecord(object):
    r"""
    Overview:
        logger that record variable for further process

    Interface:
        __init__, register_var, update_var, get_var_names, get_var_text, get_vars_tb_format, get_vars_text
    """

    def __init__(self, decay=0.99, warm_up_size=100,ignore_counts=10):
        r"""
        Overview:
            init the VariableRecord
        Arguments:
            - length (:obj:`int`): the length to average across, if less than 10 then will be set to 10
        """
        self.var_dict = {}
        self.decay = decay # at least average across 10 iteration
        self.warm_up_size = warm_up_size
        self.ignore_counts = ignore_counts
        self.update_counts = torch.tensor([0],dtype=torch.long).share_memory_()
    def register_var(self, name, decay=None, warm_up_size=None,ignore_counts=None):
        r"""
        Overview:
            add var to self_var._names, calculate it's average value
        Arguments:
            - name (:obj:`str`): name to add
            - length (:obj:`int` or :obj:`None`): length of iters to average, default set to self.length
            - var_type (:obj:`str`): the type of var to add to, defalut set to 'scalar'
        """
        decay = self.decay if decay is None else decay
        warm_up_size = self.warm_up_size if warm_up_size is None else warm_up_size
        ignore_counts = self.ignore_counts if ignore_counts is None else ignore_counts
        self.var_dict[name] = ShareEmaMeter(decay, warm_up_size,ignore_counts)

    def update_var(self, info):
        r"""
        Overview:
            update vars
        Arguments:
            - info (:obj:`dict`): key is var type and value is the corresponding variable name
        """
        assert isinstance(info, dict)
        for k, v in info.items():
            self.var_dict[k].update(v)

    def get_var_text(self, name, ):
        r"""
        Overview:
            get the text discroption of var
        Arguments:
            - name (:obj:`str`): name of the var to query
        Returns:
            - text(:obj:`str`): the corresponding text description
        """
        handle_var = self.var_dict[name]
        return '{}: val({:.6f})|avg({:.6f})'.format(name, handle_var.val,)

    def get_vars_text(self):
        r"""
        Overview:
            get the string description of var
        Returns:
            - ret (:obj:`list` of :obj:`str`): the list of text description of vars queried
        """
        headers = ["Name", "Value", ]
        data = []
        for k in self.var_dict.keys():
            handle_var = self.var_dict[k]
            data.append([k, "{:.6f}".format(handle_var.val),])
        s = "\n" + tabulate(data, headers=headers, tablefmt='grid')
        return s




class MoveAverageMeter(object):
    def __init__(self,length=1000):
        self.length = length
        self.reset()
    def reset(self):
        self.history = deque(maxlen=self.length)
        self._val = 0.0
        self._count = 0

    def update(self, val):
        if isinstance(val, torch.Tensor):
            val = val.item()
        assert (isinstance(val, list) or isinstance(val, numbers.Integral) or isinstance(val, numbers.Real))
        self._count += 1
        s1 = len(self.history)
        if s1 < self.length:
            self._val = (1-1.0/(s1+1))*self._val+1.0/(s1+1)*val
            self.history.append(val)
        else:
            left_val = self.history.popleft()
            self._val = self._val+(val-left_val)/self.length
            self.history.append(val)

    @property
    def val(self):
        return self._val

    @property
    def count(self):
        return self._count

    @val.setter
    def val(self):
        raise NotImplementedError

    @count.setter
    def count(self):
        raise NotImplementedError



class AverageMeter(object):
    r"""
    Overview:
        Computes and stores the average and current value, scalar and 1D-array
    Interface:
        __init__, reset, update
    """

    def __init__(self, length=0):
        r"""
        Overview:
            init AverageMeter class
        Arguments:
            - length (:obj:`int`) : set the default length of iters to average
        """
        assert (length > 0)
        self.length = length
        self.reset()

    def reset(self):
        r"""
        Overview:
            reset AverageMeter class
        """
        self.history = []
        self.val = 0.0
        self.avg = 0.0

    def update(self, val):
        r"""
        Overview:
            update AverageMeter class, append the val to the history and calculate the average
        Arguments:
            - val (:obj:`numbers.Integral` or :obj:`list` or :obj:`numbers.Real` ) : the latest value
        """
        if isinstance(val, torch.Tensor):
            val = val.item()
        assert isinstance(val, numbers.Integral) or isinstance(val, numbers.Real)
        self.history.append(val)
        if len(self.history) > self.length:
            del self.history[0]

        self.val = self.history[-1]
        self.avg = np.mean(self.history, axis=0)

class AvgMeter(object):
    def __init__(self,):
        self.reset()

    def reset(self):
        self._count = 0
        self._total_val = 0

    def update(self, val):
        if isinstance(val, list):
            for v in val:
                self.update(v)
            return
        if isinstance(val, torch.Tensor):
            val = val.item()
        assert isinstance(val, numbers.Integral) or isinstance(val, numbers.Real)
        self._total_val += val
        self._count += 1

    @property
    def val(self):
        if self._count <= 0:
            return 0
        else:
            return self._total_val / self._count

    @property
    def count(self):
        return self._count

    @val.setter
    def val(self):
        raise NotImplementedError

    @count.setter
    def count(self):
        raise NotImplementedError


class EmaMeter(object):
    def __init__(self, decay, warm_up_size):
        assert (warm_up_size > 0)
        assert 0 <= decay <= 1
        self._decay = decay
        self._warm_up_size = warm_up_size
        self.reset()

    def reset(self):
        self._count = 0
        self._val = 0

    def update(self, val):
        if isinstance(val, torch.Tensor):
            val = val.item()
        assert isinstance(val, numbers.Integral) or isinstance(val, numbers.Real)
        if self._count >= self._warm_up_size:
            self._val = self._decay * self._val + (1 - self._decay) * val
        else:
            self._val = (self._val * self._count + val) / (self._count + 1)
        self._count += 1

    @property
    def val(self):
        return self._val

    @property
    def count(self):
        return self._count

    @val.setter
    def val(self):
        raise NotImplementedError

    @count.setter
    def count(self):
        raise NotImplementedError

def pretty_print(result, direct_print=True):
    r"""
    Overview:
        print the result in a pretty way
    Arguments:
        - result (:obj:`dict`): the result to print
        - direct_print (:obj:`bool`): whether to print directly
    Returns:
        - string (:obj:`str`): the printed result in str format
    """
    result = result.copy()
    out = {}
    for k, v in result.items():
        if v is not None:
            out[k] = v
    cleaned = json.dumps(out)
    string = yaml.safe_dump(json.loads(cleaned), default_flow_style=False)
    if direct_print:
        print(string)
    return string


class LossVariableRecord(object):
    r"""
    Overview:
        logger that record variable for further process

    Interface:
        __init__, register_var, update_var, get_var_names, get_var_text, get_vars_tb_format, get_vars_text
    """

    def __init__(self, length, sort=True, *args,**kwargs):
        r"""
        Overview:
            init the VariableRecord
        Arguments:
            - length (:obj:`int`): the length to average across, if less than 10 then will be set to 10
        """
        self.var_dict = {}
        self.sort = sort
        self.length = max(length, 10)  # at least average across 10 iteration

    def register_var(self, name,):
        r"""
        Overview:
            add var to self_var._names, calculate it's average value
        Arguments:
            - name (:obj:`str`): name to add
            - length (:obj:`int` or :obj:`None`): length of iters to average, default set to self.length
            - var_type (:obj:`str`): the type of var to add to, defalut set to 'scalar'
        """
        self.var_dict[name] = AverageMeter(self.length)

    def update_var(self, info):
        r"""
        Overview:
            update vars
        Arguments:
            - info (:obj:`dict`): key is var type and value is the corresponding variable name
        """
        assert isinstance(info, dict)
        for k, v in info.items():
            if k not in self.var_dict:
                self.register_var(k)
            self.var_dict[k].update(v)

    def get_vars_info_text(self):
        var_info = self.get_var_info()

        headers = ["Name", "Value", ]
        data = []
        for k, val in var_info.items():
            data.append([k, "{:.6f}".format(val), ])
        vars_text = "\n" + tabulate(data, headers=headers, tablefmt='grid')
        return var_info, vars_text

    def get_vars_text(self):
        r"""
        Overview:
            get the string description of var
        Returns:
            - ret (:obj:`list` of :obj:`str`): the list of text description of vars queried
        """
        headers = ["Name", "Value", ]
        data = []
        var_info = self.get_var_info()
        if self.sort:
            var_info = dict(sorted(var_info.items()))
        for k,val in var_info.items():
            data.append([k, "{:.6f}".format(val), ])
        s = "\n" + tabulate(data, headers=headers, tablefmt='grid')
        return s

    def get_var_info(self):
        var_info = {k:val.avg  for k,val in self.var_dict.items()}
        return var_info

class LogVariableRecord(object):
    r"""
    Overview:
        logger that record variable for further process

    Interface:
        __init__, register_var, update_var, get_var_names, get_var_text, get_vars_tb_format, get_vars_text
    """

    def __init__(self,):
        r"""
        Overview:
            init the VariableRecord
        Arguments:
            - length (:obj:`int`): the length to average across, if less than 10 then will be set to 10
        """
        self.var_dict = {}

    def register_var(self, name,):
        r"""
        Overview:
            add var to self_var._names, calculate it's average value
        Arguments:
            - name (:obj:`str`): name to add
            - length (:obj:`int` or :obj:`None`): length of iters to average, default set to self.length
            - var_type (:obj:`str`): the type of var to add to, defalut set to 'scalar'
        """
        self.var_dict[name] = AvgMeter()

    def update_var(self, info):
        r"""
        Overview:
            update vars
        Arguments:
            - info (:obj:`dict`): key is var type and value is the corresponding variable name
        """
        assert isinstance(info, dict)
        for k, v in info.items():
            if k not in self.var_dict:
                self.register_var(k)
            self.var_dict[k].update(v)

    def get_vars_text(self):
        r"""
        Overview:
            get the string description of var
        Returns:
            - ret (:obj:`list` of :obj:`str`): the list of text description of vars queried
        """
        headers = ["Name", "Value", ]
        data = []
        var_info = self.get_var_info()
        for k,val in var_info.items():
            data.append([k, "{:.6f}".format(val), ])
        s = "\n" + tabulate(data, headers=headers, tablefmt='grid')
        return s

    def get_var_info(self):
        var_info = {k:val.val  for k,val in self.var_dict.items()}
        return var_info

    def reset(self):
        for k in self.var_dict:
            self.var_dict[k].reset()