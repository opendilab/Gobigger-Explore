import numbers
from collections.abc import Sequence
from typing import Iterable, Any
import time
from threading import Thread
from queue import Queue

import numpy as np
import torch


def to_device(item, device, ignore_keys=[]):
    r"""
    Overview:
        transfer data to certain device

    Arguments:
        Note:
            Now supported item type :obj:`torch.nn.Module`, :obj:`torch.Tensor`, :obj:`Sequence`, :obj:`dict`,
            :obj:`numbers.Integral`, :obj:`numbers.Real`, :obj:`np.ndarray`, :obj:`str` and :obj:`None`.

        - item (:obj:`object`): the item to be transfered
        - device (:obj:`torch.divice`): the device wanted
        - ignore_keys (:obj:`list` of `item.keys()`): the keys to be ignored in transfer, defalut set to empty

    Returns:
        - item (:obj:`object`): the transfered item
    """
    if isinstance(item, torch.nn.Module):
        return item.to(device)
    elif isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, Sequence):
        if isinstance(item, str):
            return item
        else:
            return [to_device(t, device) for t in item]
    elif isinstance(item, dict):
        new_item = {}
        for k in item.keys():
            if k in ignore_keys:
                new_item[k] = item[k]
            else:
                new_item[k] = to_device(item[k], device)
        return new_item
    elif isinstance(item, numbers.Integral) or isinstance(item, numbers.Real):
        return item
    elif isinstance(item, np.ndarray) or isinstance(item, np.bool_):
        return item
    elif item is None or isinstance(item, str):
        return item
    else:
        raise TypeError("not support item type: {}".format(type(item)))

def to_contiguous(item, ignore_keys=[]):
    r"""
    Overview:
        transfer data to certain device

    Arguments:
        Note:
            Now supported item type :obj:`torch.nn.Module`, :obj:`torch.Tensor`, :obj:`Sequence`, :obj:`dict`,
            :obj:`numbers.Integral`, :obj:`numbers.Real`, :obj:`np.ndarray`, :obj:`str` and :obj:`None`.

        - item (:obj:`object`): the item to be transfered
        - device (:obj:`torch.divice`): the device wanted
        - ignore_keys (:obj:`list` of `item.keys()`): the keys to be ignored in transfer, defalut set to empty

    Returns:
        - item (:obj:`object`): the transfered item
    """
    if isinstance(item, torch.nn.Module):
        return item.contiguous()
    elif isinstance(item, torch.Tensor):
        return item.contiguous()
    elif isinstance(item, Sequence):
        if isinstance(item, str):
            return item
        else:
            return [to_contiguous(t) for t in item]
    elif isinstance(item, dict):
        new_item = {}
        for k in item.keys():
            if k in ignore_keys:
                new_item[k] = item[k]
            else:
                new_item[k] = to_contiguous(item[k])
        return new_item
    elif isinstance(item, numbers.Integral) or isinstance(item, numbers.Real):
        return item
    elif isinstance(item, np.ndarray) or isinstance(item, np.bool_):
        return item
    elif item is None or isinstance(item, str):
        return item
    else:
        raise TypeError("not support item type: {}".format(type(item)))

def to_pin_memory(item, ignore_keys=[]):
    r"""
    Overview:
        transfer data to certain device

    Arguments:
        Note:
            Now supported item type :obj:`torch.nn.Module`, :obj:`torch.Tensor`, :obj:`Sequence`, :obj:`dict`,
            :obj:`numbers.Integral`, :obj:`numbers.Real`, :obj:`np.ndarray`, :obj:`str` and :obj:`None`.

        - item (:obj:`object`): the item to be transfered
        - device (:obj:`torch.divice`): the device wanted
        - ignore_keys (:obj:`list` of `item.keys()`): the keys to be ignored in transfer, defalut set to empty

    Returns:
        - item (:obj:`object`): the transfered item
    """
    if isinstance(item, torch.nn.Module):
        return item.pin_memory()
    elif isinstance(item, torch.Tensor):
        return item.pin_memory()
    elif isinstance(item, Sequence):
        if isinstance(item, str):
            return item
        else:
            return [to_pin_memory(t) for t in item]
    elif isinstance(item, dict):
        new_item = {}
        for k in item.keys():
            if k in ignore_keys:
                new_item[k] = item[k]
            else:
                new_item[k] = to_pin_memory(item[k])
        return new_item
    elif isinstance(item, numbers.Integral) or isinstance(item, numbers.Real):
        return item
    elif isinstance(item, np.ndarray) or isinstance(item, np.bool_):
        return item
    elif item is None or isinstance(item, str):
        return item
    else:
        raise TypeError("not support item type: {}".format(type(item)))

def to_share(item, ignore_keys=[]):
    r"""
    Overview:
        transfer data to certain device

    Arguments:
        Note:
            Now supported item type :obj:`torch.nn.Module`, :obj:`torch.Tensor`, :obj:`Sequence`, :obj:`dict`,
            :obj:`numbers.Integral`, :obj:`numbers.Real`, :obj:`np.ndarray`, :obj:`str` and :obj:`None`.

        - item (:obj:`object`): the item to be transfered
        - device (:obj:`torch.divice`): the device wanted
        - ignore_keys (:obj:`list` of `item.keys()`): the keys to be ignored in transfer, defalut set to empty

    Returns:
        - item (:obj:`object`): the transfered item
    """
    if isinstance(item, torch.nn.Module):
        item.share_memory_()
        return item
    elif isinstance(item, torch.Tensor):
        item.share_memory_()
        return item
    elif isinstance(item, Sequence):
        if isinstance(item, str):
            return item
        else:
            return [to_share(t,) for t in item]
    elif isinstance(item, dict):
        new_item = {}
        for k in item.keys():
            if k in ignore_keys:
                new_item[k] = item[k]
            else:
                new_item[k] =to_share(item[k],)
        return new_item
    elif isinstance(item, numbers.Integral) or isinstance(item, numbers.Real):
        return item
    elif isinstance(item, np.ndarray) or isinstance(item, np.bool_):
        return item
    elif item is None or isinstance(item, str):
        return item
    else:
        raise TypeError("not support item type: {}".format(type(item)))

def to_dtype(item, dtype):
    r"""
    Overview:
        transfer data to certain dtype

    Arguments:
        Note:
            Now supported item type: :obj:`torch.Tensor`, :obj:`Sequence`, :obj:`dict`

        - item (:obj:`object`): the item to be transfered
        - dtype (:obj:`type`): the type wanted

    Returns:
        - item (:obj:`object`): the transfered item
    """
    if isinstance(item, torch.Tensor):
        return item.to(dtype=dtype)
    elif isinstance(item, Sequence):
        return [to_dtype(t, dtype) for t in item]
    elif isinstance(item, dict):
        return {k: to_dtype(item[k], dtype) for k in item.keys()}
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def to_tensor(item, dtype):
    r"""
    Overview:
        transfer data to certain dtype tensor

    Arguments:
        Note:
            Now supported item type: :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`

        - item (:obj:`object`): the item to be transfered
        - dtype (:obj:`type`): the type of wanted tensor

    Returns:
        - item (:obj:`object`): the transfered item
    """

    def transform(d):
        return torch.tensor(d, dtype=dtype)

    if isinstance(item, dict):
        new_data = {}
        for k, v in item.items():
            new_data[k] = to_tensor(v, dtype)
        return new_data
    elif isinstance(item, list) or isinstance(item, tuple):
        if len(item) == 0:
            return None
        elif isinstance(item[0], numbers.Integral) or isinstance(item[0], numbers.Real):
            return transform(item)
        else:
            new_data = []
            for t in item:
                new_data.append(to_tensor(t, dtype))
            return new_data
    elif isinstance(item, np.ndarray):
        return torch.from_numpy(item).to(dtype)
    elif np.isscalar(item):
        return torch.as_tensor([item]).to(dtype)
    elif item is None:
        return None
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def tensor_to_list(item):
    r"""
    Overview:
        transfer data to certain dtype

    Arguments:
        Note:
            Now supported item type: :obj:`torch.Tensor`, :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`

        - item (:obj:`object`): the item to be transfered

    Returns:
        - item (:obj:`list`): the transfered list
    """
    if item is None:
        return item
    elif isinstance(item, torch.Tensor):
        if item.shape == (1, ):
            return item.item()
        else:
            return item.tolist()
    elif isinstance(item, list) or isinstance(item, tuple):
        return [tensor_to_list(t) for t in item]
    elif isinstance(item, dict):
        return {k: tensor_to_list(v) for k, v in item.items()}
    elif np.isscalar(item):
        return item
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def same_shape(data):
    r"""
    Overview:
        whether a list of data have same shapes

    Arguments:
        - data (:obj:`list`): the list of data

    Returns:
        - same (:obj:`bool`): whether the list of data all have same shapes
    """
    assert (isinstance(data, list))
    shapes = [t.shape for t in data]
    return len(set(shapes)) == 1

def get_tensor_data(data: Any) -> Any:
    """
    Overview:
        get pure tensor data from the given data(avoiding disturbing grad computation graph)
    """
    if isinstance(data, torch.Tensor):
        return data.data.clone()
    elif data is None:
        return None
    elif isinstance(data, Sequence):
        return [get_tensor_data(d) for d in data]
    elif isinstance(data, dict):
        return {k: v for k, v in data.items()}
    else:
        raise TypeError("not support type in get_tensor_data: {}".format(type(data)))

def flat(data):
    if isinstance(data, torch.Tensor):
        return torch.flatten(data, start_dim=0, end_dim=1)  # (1, (T+1) * B)
    elif isinstance(data, dict):
        new_data = {}
        for k, val in data.items():
            new_data[k] = flat(val)
        return new_data
    elif isinstance(data, Sequence):
        new_data = [flat(v) for v in data]
        return new_data
    else:
        print(type(data))