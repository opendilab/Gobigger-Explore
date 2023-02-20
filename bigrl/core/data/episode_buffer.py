import torch
import numpy as np
import numbers
from collections.abc import Sequence
from types import SimpleNamespace as SN
from bigrl.core.torch_utils.data_helper import to_device, to_share, to_contiguous, to_pin_memory
import random


class EpisodeBatch:
    def __init__(self,
                 batch_size,
                 player_num,
                 max_seq_length,
                 data=None,
                 device="cpu",
                 features=None):
        self.scheme = {
            "obs": {"vshape": (1,), "group": player_num},
            "action": {"vshape": (1,), "group": player_num, "dtype": torch.long},
            "reward": {"vshape": (1,), "group": player_num,},
            "terminated": {"vshape": (1,), "dtype": torch.uint8},
        }
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.player_num = player_num
        self.device = device
        self.features = features

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, batch_size, max_seq_length, player_num)

    def _setup_data(self, scheme, batch_size, max_seq_length, player_num):

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": torch.long},
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", torch.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:
                shape = (group, *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = torch.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            else:
                if field_key == 'obs':
                    self.data.transition_data[field_key] = to_contiguous(self.features.get_marl_batch_data(max_seq_length - 1, batch_size, player_num))
                else:
                    self.data.transition_data[field_key] = torch.zeros((batch_size, max_seq_length, *shape), dtype=dtype, device=self.device)

    def extend(self, scheme):
        self._setup_data(scheme, self.batch_size, self.max_seq_length)
    
    def keys(self):
        return self.scheme.keys()

    def to(self, device):
        for k, v in self.data.transition_data.items():
            if k == 'obs':
                self.data.transition_data[k] = to_device(self.data.transition_data[k], device)
            else:
                self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        slices = self._parse_slices((bs, ts))
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            if k == 'obs':
                copy_data(_slices, v, self.data.transition_data[k])
                continue
            dtype = self.scheme[k].get("dtype", torch.float32)
            v = torch.tensor(v, dtype=dtype, device=self.device)
            target[k][_slices] = v.view_as(target[k][_slices])

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            ret = EpisodeBatch(self.batch_size, self.player_num, self.max_seq_length, data=new_data, device=self.device)
            return ret
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = get_data_from_indices(v, item)
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(ret_bs, self.player_num, ret_max_t, data=new_data, device=self.device)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            or (isinstance(items, (list, np.ndarray, torch.LongTensor, torch.cuda.LongTensor)))  # [a,b,c]
            ):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            #TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item+1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        return torch.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{}".format(self.batch_size,
                                                                                     self.max_seq_length,
                                                                                     self.scheme.keys())


class ReplayBuffer(EpisodeBatch):
    def __init__(self, buffer_size, player_num, max_seq_length, device="cpu", features=None):
        super(ReplayBuffer, self).__init__(buffer_size, player_num, max_seq_length, device=device, features=features)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0

    def insert_episode_batch(self, ep_batch):
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            return self[ep_ids]

    def uni_sample(self, batch_size):
        return self.sample(batch_size)

    def sample_latest(self, batch_size):
        assert self.can_sample(batch_size)
        if self.buffer_index - batch_size < 0:
            #Uniform sampling
            return self.uni_sample(batch_size)
        else:
            # Return the latest
            return self[self.buffer_index - batch_size : self.buffer_index]

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys())



def copy_data(batch_idx, src_data, dest_data, k=None):
    if isinstance(src_data, dict):
        for k in src_data:
            copy_data(batch_idx, src_data[k], dest_data[k], k=k)
    if isinstance(src_data, torch.Tensor):
        if dest_data.dtype != src_data.dtype:
            print(f'{k} dtype not same, dest: {dest_data.dtype}, src: {src_data.dtype}', flush=True)
        try:
            if isinstance(batch_idx[0], list):
                idx, s = batch_idx
                for i, j in enumerate(idx):
                    dest_data[j][s].copy_(src_data.view_as(dest_data[batch_idx])[i])
            else:
                dest_data[batch_idx].copy_(src_data.view_as(dest_data[batch_idx]))
            # src_shape = src_data.shape[1:]
            # dest_shape = dest_data.shape[2:]
            # assert len(src_shape) == len(dest_shape)
            # final_shape = [min(a, b) for a, b in zip(src_shape, dest_shape)]
            # if len(src_shape) == 0:
            #     dest_data[batch_idx].copy_(src_data)
            # elif len(src_shape) == 1:
            #     dest_data[batch_idx, :final_shape[0]].copy_(src_data[:, :dest_shape[0]])
            # elif len(src_shape) == 2:
            #     dest_data[batch_idx, :final_shape[0], :final_shape[1]].copy_(
            #         src_data[:, :dest_shape[0], :dest_shape[1]])
            # elif len(src_shape) == 3:
            #     dest_data[batch_idx, :final_shape[0], :final_shape[1], :final_shape[2]].copy_(
            #         src_data[:, :dest_shape[0], :dest_shape[1], :dest_shape[2]])
            # elif len(src_shape) == 4:
            #     dest_data[batch_idx, :final_shape[0], :final_shape[1], :final_shape[2], :final_shape[3]].copy_(
            #         src_data[:, :dest_shape[0], :dest_shape[1], :dest_shape[2], :dest_shape[3]])
            # else:
            #     raise NotImplementedError
        except Exception as e:
            print(k, dest_data.shape, src_data.shape)
            print(k, dest_data.dtype, src_data.dtype)
            raise e


def flatten_data(data):
    if isinstance(data, dict):
        return {k: flatten_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return torch.flatten(data, start_dim=0, end_dim=1)


def get_data_from_indices(data, indices):
    if isinstance(data, dict):
        return {k: get_data_from_indices(v, indices) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data[indices]

def to_device(item, device, ignore_keys=[]):
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