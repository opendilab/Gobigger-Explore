import io
import os
import pickle
from typing import NoReturn, Union

import _pickle as cPickle
import lz4.frame
import torch

from .data_helper import to_tensor, to_ndarray


def read_from_file(path: str) -> object:
    """
    Overview:
        read file from local file system
    Arguments:
        - path (:obj:`str`): file path in local file system
    Returns:
        - (:obj`data`): deserialized data
    """
    with open(path, "rb") as f:
        value = pickle.load(f)

    return value


def read_file(path: str, fs_type: Union[None, str] = None) -> object:
    r"""
    Overview:
        read file from path
    Arguments:
        - path (:obj:`str`): the path of file to read
        - fs_type (:obj:`str` or :obj:`None`): the file system type, support 'normal' and 'ceph'
    """
    data = torch.load(path, map_location='cpu')
    return data


def save_file(path: str, data: object, fs_type: Union[None, str] = None) -> NoReturn:
    r"""
    Overview:
        save data to file of path
    Arguments:
        - path (:obj:`str`): the path of file to save to
        - data (:obj:`object`): the data to save
        - fs_type (:obj:`str` or :obj:`None`): the file system type, support 'normal' and 'ceph'
    """

    torch.save(data, path)


def remove_file(path: str, fs_type: Union[None, str] = None) -> NoReturn:
    r"""
    Overview:
        remove file
    Arguments:
        - path (:obj:`str`): the path of file you want to remove
        - fs_type (:obj:`str` or :obj:`None`): the file system type, support 'normal' and 'ceph'
    """
    if fs_type is None:
        fs_type = 'ceph' if path.lower().startswith('s3') else 'normal'
    assert fs_type in ['normal', 'ceph']
    if fs_type == 'ceph':
        pass
        os.popen("aws s3 rm --recursive {}".format(path))
    elif fs_type == 'normal':
        os.popen("rm -rf {}".format(path))


def save_traj_file(data: object, path: str, fs_type: Union[None, str] = 'pickle', compress=True) -> NoReturn:
    if fs_type == 'torch':
        torch.save(data, path, _use_new_zipfile_serialization=False)
    elif fs_type == 'torchnp':
        data = to_ndarray(data)
        torch.save(data, path, _use_new_zipfile_serialization=False)
    else:
        data = to_ndarray(data)
        with open(path, 'wb', buffering=0) as f:
            pickle.dump(data, f)


def load_traj_file(path: str, fs_type: Union[None, str] = 'pickle', compress=True) -> object:
    if fs_type == 'torch':
        data = torch.load(path, map_location='cpu')
    elif fs_type == 'torchnp':
        data = torch.load(path, map_location='cpu')
        data = to_tensor(data)
    else:
        with open(path, "rb") as f:
            data = pickle.load(f)
        data = to_tensor(data)
    return data


def dumps(data, fs_type: Union[None, str] = 'cPickle', compress=True):
    # return cPickle.dumps(data)
    if fs_type == 'torch':
        b = io.BytesIO()
        torch.save(data, b)
        data = b.getvalue()
    elif fs_type == 'cPickle':
        data = cPickle.dumps(data)
    elif fs_type == 'npcPickle':
        data = cPickle.dumps(to_ndarray(data))
    elif fs_type == 'pickle':
        data = pickle.dumps(data)
    elif fs_type == 'nppickle':
        data = pickle.dumps(to_ndarray(data))
    else:
        print(f'not support fs_type:{fs_type}')
        raise NotImplementedError
    if compress:
        data = lz4.frame.compress(data)
    return data


def loads(data, fs_type: Union[None, str] = 'cPickle', compress=True):
    if compress:
        data = lz4.frame.decompress(data)
    if fs_type == 'torch':
        data = io.BytesIO(data)
        data = torch.load(data, map_location='cpu')
    elif fs_type == 'cPickle':
        data = cPickle.loads(data)
    elif fs_type == 'npcPickle':
        data = to_tensor(cPickle.loads(data))
    elif fs_type == 'pickle':
        data = pickle.loads(data)
    elif fs_type == 'nppickle':
        data = to_tensor(pickle.loads(data))

    else:
        print(f'not support fs_type:{fs_type}')
        raise NotImplementedError
    return data
