import re
from collections.abc import Sequence, Mapping
from typing import List, Dict, Union, Any

import torch
import collections.abc as container_abcs
from torch._six import string_classes
#from torch._six import int_classes as _int_classes
int_classes = int

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def default_collate(batch: Sequence) -> Union[torch.Tensor, Mapping, Sequence]:
    """
    Overview:
        Put each data field into a tensor with outer dimension batch size.
    Example:
        >>> # a list with B tensors shaped (m, n) -->> a tensor shaped (B, m, n)
        >>> a = [torch.zeros(2,3) for _ in range(4)]
        >>> default_collate(a).shape
        torch.Size([4, 2, 3])
        >>>
        >>> # a list with B lists, each list contains m elements -->> a list of m tensors, each with shape (B, )
        >>> a = [[0 for __ in range(3)] for _ in range(4)]
        >>> default_collate(a)
        [tensor([0, 0, 0, 0]), tensor([0, 0, 0, 0]), tensor([0, 0, 0, 0])]
        >>>
        >>> # a list with B dicts, whose values are tensors shaped :math:`(m, n)` -->>
        >>> # a dict whose values are tensors with shape :math:`(B, m, n)`
        >>> a = [{i: torch.zeros(i,i+1) for i in range(2, 4)} for _ in range(4)]
        >>> print(a[0][2].shape, a[0][3].shape)
        torch.Size([2, 3]) torch.Size([3, 4])
        >>> b = default_collate(a)
        >>> print(b[2].shape, b[3].shape)
        torch.Size([4, 2, 3]) torch.Size([4, 3, 4])
    Arguments:
        - batch (:obj:`Sequence`): a data sequence, whose length is batch size, whose element is one piece of data
    Returns:
        - ret (:obj:`Union[torch.Tensor, Mapping, Sequence]`): the collated data, with batch size into each data field.\
            the return dtype depends on the original element dtype, can be [torch.Tensor, Mapping, Sequence].
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, directly concatenate into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        if elem.shape == (1,):
            # reshape (B, 1) -> (B)
            return torch.cat(batch, 0, out=out)
        else:
            return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    default_collate_err_msg_format.format(
                        elem.dtype))
            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float32)
    elif isinstance(elem, int_classes):
        dtype = torch.bool if isinstance(elem, bool) else torch.int64
        return torch.tensor(batch, dtype=dtype)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples)
                           for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def default_collate_with_dim(batch,device='cpu',dim=0, k=None,cat=False):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    #if k is not None:
    #    print(k)

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        try:
            if cat == True:
                return torch.cat(batch, dim=dim, out=out).to(device=device)
            else:
                return torch.stack(batch, dim=dim, out=out).to(device=device)
        except:
            print(batch)
            if k is not None:
                print(k)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate_with_dim([torch.as_tensor(b,device=device) for b in batch],device=device,dim=dim,cat=cat)
        elif elem.shape == ():  # scalars
            try:
                return torch.as_tensor(batch,device=device)
            except:
                print(batch)
                if k is not None:
                    print(k)
    elif isinstance(elem, float):
        try:
            return torch.tensor(batch,device=device)
        except:
            print(batch)
            if k is not None:
                print(k)
    elif isinstance(elem, int_classes):
        try:
            return torch.tensor(batch,device=device)
        except:
            print(batch)
            if k is not None:
                print(k)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate_with_dim([d[key] for d in batch if key in d.keys()],device=device,dim=dim, k=key, cat=cat) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate_with_dim(samples,device=device,dim=dim,cat=cat) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate_with_dim(samples,device=device,dim=dim,cat=cat) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def timestep_collate(batch: List[Dict[str, Any]]
                     ) -> Dict[str, Union[torch.Tensor, list]]:
    """
    Overview:
        Put each timestepped data field into a tensor with outer dimension batch size using ``default_collate``.
        For short, this process can be represented by:
        [len=B, ele={dict_key: [len=T, ele=Tensor(any_dims)]}] -> {dict_key: Tensor([T, B, any_dims])}
    Arguments:
        - batch (:obj:`List[Dict[str, Any]]`): a list of dicts with length B, each element is {some_key: some_seq} \
            ('prev_state' should be a key in the dict); \
            some_seq is a sequence with length T, each element is a torch.Tensor with any shape.
    Returns:
        - ret (:obj:`Dict[str, Union[torch.Tensor, list]]`): the collated data, with timestep and batch size \
            into each data field. By using ``default_collate``, timestep would come to the first dim. \
            So the final shape is :math:`(T, B, dim1, dim2, ...)`
    """

    def stack(data):
        if isinstance(data, container_abcs.Mapping):
            return {k: stack(data[k]) for k in data}
        elif isinstance(data, container_abcs.Sequence) and isinstance(data[0], torch.Tensor):
            return torch.stack(data)
        else:
            return data

    elem = batch[0]
    assert isinstance(elem, container_abcs.Mapping), type(elem)
    prev_state = [b.pop('prev_state') for b in batch]
    # -> {some_key: T lists}, each list is [B, some_dim]
    batch = default_collate(batch)
    batch = stack(batch)  # -> {some_key: [T, B, some_dim]}
    batch['prev_state'] = list(zip(*prev_state))
    return batch


def diff_shape_collate(
        batch: Sequence) -> Union[torch.Tensor, Mapping, Sequence]:
    """
    Overview:
        Similar to ``default_collate``, put each data field into a tensor with outer dimension batch size.
        The main difference is that, ``diff_shape_collate`` allows tensors in the batch have `None`,
        which is quite common StarCraft observation.
    Arguments:
        - batch (:obj:`Sequence`): a data sequence, whose length is batch size, whose element is one piece of data
    Returns:
        - ret (:obj:`Union[torch.Tensor, Mapping, Sequence]`): the collated data, with batch size into each data field.\
            the return dtype depends on the original element dtype, can be [torch.Tensor, Mapping, Sequence].
    """
    elem = batch[0]
    elem_type = type(elem)
    assert not any([isinstance(batch[elem_idx], type(None)) for elem_idx in range(len(batch)-1)]),\
        f'none object must at the end of sequence'
    if isinstance(elem, torch.Tensor):
        shapes = [e.shape for e in batch]
        if len(set(shapes)) != 1:
            return batch
        else:
            return torch.stack(batch, 0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray':
            return diff_shape_collate([torch.as_tensor(b)
                                       for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float32)
    elif isinstance(elem, int_classes):
        dtype = torch.bool if isinstance(elem, bool) else torch.int64
        return torch.tensor(batch, dtype=dtype)
    elif isinstance(elem, Mapping):
        return {key: diff_shape_collate(
            [d[key] for d in batch if key in d]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(diff_shape_collate(samples)
                           for samples in zip(*batch)))
    elif isinstance(elem, Sequence):
        transposed = zip(*batch)
        return [diff_shape_collate(samples) for samples in transposed]

    raise TypeError('not support element type: {}'.format(elem_type))

def default_decollate(batch: Union[torch.Tensor, Sequence, Mapping], ignore: List[str] = ['prev_state']) -> List[Any]:
    """
    Overview:
        Drag out batch_size collated data's batch size to decollate it,
        which is the reverse operation of ``default_collate``.
    Arguments:
        - batch (:obj:`Union[torch.Tensor, Sequence, Mapping]`): can reference the Returns of ``default_collate``
        - ignore(:obj:`List[str]`): a list of names to be ignored, only function if input ``batch`` is a dict. \
            If key is in this list, its value would stay the same with no decollation.
    Returns:
        - ret (:obj:`List[Any]`): a list with B elements.
    """
    if isinstance(batch, torch.Tensor):
        batch = torch.split(batch, 1, dim=0)
        # squeeze if original batch's shape is like (B, dim1, dim2, ...);
        # otherwise directly return the list.
        if len(batch[0].shape) > 1:
            batch = [elem.squeeze(0) for elem in batch]
        return list(batch)
    elif isinstance(batch, Sequence):
        return list(zip(*[default_decollate(e) for e in batch]))
    elif isinstance(batch, Mapping):
        tmp = {k: v if k in ignore else default_decollate(v) for k, v in batch.items()}
        B = len(list(tmp.values())[0])
        return [{k: tmp[k][i] for k in tmp.keys()} for i in range(B)]

    raise TypeError("not support batch type: {}".format(type(batch)))

def default_decollate_with_dim(batch: Union[torch.Tensor, Sequence, Mapping], ignore: List[str] = [],dim=0) -> List[Any]:
    """
    Overview:
        Drag out batch_size collated data's batch size to decollate it,
        which is the reverse operation of ``default_collate``.
    Arguments:
        - batch (:obj:`Union[torch.Tensor, Sequence, Mapping]`): can reference the Returns of ``default_collate``
        - ignore(:obj:`List[str]`): a list of names to be ignored, only function if input ``batch`` is a dict. \
            If key is in this list, its value would stay the same with no decollation.
    Returns:
        - ret (:obj:`List[Any]`): a list with B elements.
    """
    if isinstance(batch, torch.Tensor):
        batch = torch.split(batch, 1, dim=dim)
        # squeeze if original batch's shape is like (B, dim1, dim2, ...);
        # otherwise directly return the list.
        batch = [elem.squeeze(dim) for elem in batch]
        return list(batch)
    elif isinstance(batch, Sequence):
        return list(zip(*[default_decollate_with_dim(e,dim=dim) for e in batch]))
    elif isinstance(batch, Mapping):
        tmp = {k: v if k in ignore else default_decollate_with_dim(v,dim=dim) for k, v in batch.items()}
        B = len(list(tmp.values())[0])
        return [{k: tmp[k][i] for k in tmp.keys()} for i in range(B)]

    raise TypeError("not support batch type: {}".format(type(batch)))

def default_collate_data_batch(data):
    data_batch = default_collate_with_dim(data)
    data_batch_collated = default_collate_with_dim(data_batch)
    if 'prev_state' in data_batch_collated.keys():
        for k in range(2):
            data_batch_collated['prev_state'][k] = data_batch_collated['prev_state'][k].permute(2, 0, 1, 3)
    return data_batch_collated
