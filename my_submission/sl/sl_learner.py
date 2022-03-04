import os
import sys
import random
import torch
import torch.nn.functional as F
import pickle
import time
from easydict import EasyDict
from collections import OrderedDict
import numpy as np
import copy
import logging
import datetime
import multiprocessing as mp
from tensorboardX import SummaryWriter
sys.path.append('..')
sys.path.append('.')

from data import SLDataLoader, SLShareDataLoader
from model import GoBiggerHybridActionSimpleV3
from ding.model import model_wrap
from ding.torch_utils import to_device
from utils.misc import AverageMeter, accuracy, create_logger, get_logger

logging.basicConfig(level=logging.INFO)


class SLLearner:

    @staticmethod
    def default_config():
        cfg = dict(
            use_cuda=True,
            learning_rate=0.01,
            milestones=[20000,40000,60000,80000],
            gamma=0.8,
            weight_decay=0.00005,
            max_iterations=100000,
            save_frequency=5000,
            exp_path='exp',
            exp_name='sample',
            print_freq=1,
            resume_ckpt='',
            model=dict(
                scalar_shape=5,
                food_shape=2,
                food_relation_shape=150,
                thorn_relation_shape=12,
                clone_shape=17,
                clone_relation_shape=12,
                hidden_shape=128,
                encode_shape=32,
                action_type_shape=16, # 6 * 16
            ),
            data=dict(
                team_num=4,
                player_num_per_team=3,
                batch_size=40,
                cache_size=120,
                train_data_prefix='PATH/replays',
                train_data_file='PATH/replays.txt.train',
                worker_num=40,
                angle_split_num=4,
                action_type_num=4,
            ),
        )
        return EasyDict(cfg)

    def __init__(self, cfg):
        self.cfg = cfg
        # model
        assert self.cfg.model.action_type_shape == self.cfg.data.angle_split_num * self.cfg.data.action_type_num
        self.model = GoBiggerHybridActionSimpleV3(**self.cfg.model)
        self.model = model_wrap(self.model, wrapper_name='argmax_sample')
        if self.cfg.use_cuda:
            self.model.cuda()
        # data
        self.loader = SLShareDataLoader(self.cfg.data)
        # loss
        self.criterion = torch.nn.CrossEntropyLoss()
        # optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay
        )
        # lr scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=self.cfg.milestones, 
            gamma=self.cfg.gamma
        )
        if not os.path.isdir(self.cfg.exp_path):
            os.mkdir(self.cfg.exp_path)
        self.root_path = os.path.join(self.cfg.exp_path, self.cfg.exp_name)
        if not os.path.isdir(self.root_path):
            os.mkdir(self.root_path)
        self.log_path = os.path.join(self.root_path, 'log.txt')
        self.save_path = os.path.join(self.root_path, 'ckpts')
        self.event_path = os.path.join(self.root_path, 'events')
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        if not os.path.isdir(self.event_path):
            os.mkdir(self.event_path)
        # logger
        create_logger(self.log_path, level=logging.INFO)
        self.logger = get_logger(self.log_path)
        self.logger.info(self.cfg)
        # tensorboard logger
        self.tb_logger = SummaryWriter(self.event_path)

    def pre_train(self):
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.cfg.print_freq)
        self.meters.forward_time = AverageMeter(self.cfg.print_freq)
        self.meters.data_time = AverageMeter(self.cfg.print_freq)
        self.meters.losses = AverageMeter(self.cfg.print_freq)
        self.meters.top1 = AverageMeter(self.cfg.print_freq)
        self.model.train()

    def load_ckpt(self):
        if self.cfg.resume_ckpt and os.path.isfile(self.cfg.resume_ckpt):
            self.state = torch.load(self.cfg.resume_ckpt, map_location='cpu')
            self.logger.info("Recovering from {}".format(self.cfg.resume_ckpt))
            self.model.load_state_dict(self.state['model'], strict=False)

    def train(self):
        self.load_ckpt()
        self.pre_train()
        for i in range(self.cfg.max_iterations):
            t1 = time.time()
            data, label = next(self.loader)
            self.lr_scheduler.step(i)
            current_lr = self.lr_scheduler.get_lr()[0]
            self.meters.data_time.update(time.time() - t1)
            label = torch.tensor(label, dtype=torch.long)
            if self.cfg.use_cuda:
                data = to_device(data, 'cuda:0')
                label = to_device(label, 'cuda:0').view(-1)
            else:
                data = to_device(data, 'cpu')
                label = to_device(label, 'cpu').view(-1)
            t3 = time.time()
            output = self.model.forward(data)
            self.meters.forward_time.update(time.time() - t3)
            logit = output['logit'].view(-1, self.cfg.model.action_type_shape)
            action = output['action']
            loss = self.criterion(logit, label)
            loss = loss.sum()
            prec1 = accuracy(logit, label, topk=(1,))[0]
            reduced_loss = loss.clone()
            self.meters.losses.reduce_update(reduced_loss)
            self.meters.top1.update(prec1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.meters.batch_time.update(time.time() - t1)

            if i % self.cfg.print_freq == 0:
                self.tb_logger.add_scalar('loss_train', self.meters.losses.avg, i)
                self.tb_logger.add_scalar('acc1_train', self.meters.top1.avg, i)
                self.tb_logger.add_scalar('lr', current_lr, i)
                remain_secs = (self.cfg.max_iterations - i) * self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()+remain_secs))
                log_msg = 'Iter: [{}/{}]\t'.format(i, self.cfg.max_iterations)
                log_msg += 'Time {:.3f} ({:.3f})\t'.format(self.meters.batch_time.val, self.meters.data_time.avg)
                log_msg += 'Data {:.3f} ({:.3f})\t'.format(self.meters.data_time.val, self.meters.data_time.avg)
                log_msg += 'Forward {:.3f} ({:.3f})\t'.format(self.meters.forward_time.val, self.meters.forward_time.avg)
                log_msg += 'Loss {:.4f} ({:.4f})\t'.format(self.meters.losses.val, self.meters.losses.avg)
                log_msg += 'Prec@1 {:.3f} ({:.3f})\t'.format(self.meters.top1.val.item(), self.meters.top1.avg.item())
                log_msg += 'LR {:.4f}\t'.format(current_lr)
                log_msg += 'Remaining Time {} ({})'.format(remain_time, finish_time)
                self.logger.info(log_msg)

            if i % self.cfg.save_frequency == 0:
                state = {}
                state['model'] = self.model.state_dict()
                state['optimizer'] = self.optimizer.state_dict()
                state['last_iter'] = i
                ckpt_name = os.path.join(self.save_path, '{}.pth.tar'.format(i))
                torch.save(state, ckpt_name)
                # if i > 0:
                #     import pdb; pdb.set_trace()


if __name__ == '__main__':
    sl_learner = SLLearner(SLLearner.default_config())
    sl_learner.train()