import math
import random
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .lstm import script_lnlstm
from .nn_module import fc_block
from .rnn import sequence_mask



class PointerNetwork(nn.Module):
    def __init__(self, cfg):
        super(PointerNetwork, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model
        self.activation_type = self.cfg.activation
        self.entity_reduce_type = self.cfg.entity_reduce_type

        self.features_cfg = self.whole_cfg.agent.get('features', {})
        self.MAX_SELECTED_UNITS_NUM = self.features_cfg.get('max_selected_units_num', 64)
        self.MAX_ENTITY_NUM = self.features_cfg.get('max_entity_num', 512)

        self.key_fc = fc_block(self.cfg.entity_embedding_dim, self.cfg.key_dim, activation=None, norm_type=None)

        self.query_mlp = nn.Sequential(*[
            fc_block(self.cfg.input_dim, self.cfg.func_dim, activation=self.activation_type),
            fc_block(self.cfg.func_dim, self.cfg.key_dim, activation=None),
        ])
        self.embed_mlp = nn.Sequential(
            *[fc_block(self.cfg.key_dim, self.cfg.func_dim, activation=self.activation_type, norm_type=None),
              fc_block(self.cfg.func_dim, self.cfg.input_dim, activation=None, norm_type=None)])

        self.lstm_num_layers = self.cfg.lstm_num_layers
        self.lstm_hidden_dim = self.cfg.lstm_hidden_dim
        self.lstm = script_lnlstm(self.cfg.key_dim, self.lstm_hidden_dim, self.lstm_num_layers)
        self._setup_end_embedding()

    def _setup_end_embedding(self):
        self.end_embedding = torch.nn.Parameter(torch.FloatTensor(1, self.cfg.key_dim))
        stdv = 1. / math.sqrt(self.end_embedding.size(1))
        self.end_embedding.data.uniform_(-stdv, stdv)

    def _get_key_mask(self, entity_embedding, entity_num):
        bs = entity_embedding.shape[0]  # batch size
        padding_end = torch.zeros(1, self.end_embedding.shape[1]).repeat(bs, 1,
                                                                         1).to(entity_embedding.device)  # b, 1, c
        key = self.key_fc(entity_embedding)  # b, n, c
        key = torch.cat([key, padding_end], dim=1)  # b, (n+1), c

        # end_embeddings = torch.ones(key.shape, dtype=key.dtype, device=key.device) * self.end_embedding.squeeze(
        #     dim=0)  # b, (n+1), c
        #
        # flag = torch.ones(key.shape[:2], dtype=torch.bool, device=key.device).unsqueeze(dim=2)  # b, (n+1), 1
        # flag[torch.arange(bs), entity_num] = 0
        #
        # key = key * flag + end_embeddings * ~flag

        key[torch.arange(bs),entity_num] = self.end_embedding

        new_entity_num = entity_num + 1  # add end entity
        entity_mask = sequence_mask(new_entity_num, max_len=entity_embedding.shape[1] + 1)

        if self.entity_reduce_type == 'entity_num':
            key_reduce = torch.div(key, entity_num.reshape(-1, 1, 1))
            key_embeddings = self.embed_mlp(key_reduce)
        elif self.entity_reduce_type == 'constant':
            key_reduce = torch.div(key, self.MAX_ENTITY_NUM)
            key_embeddings = self.embed_mlp(key_reduce)
        elif self.entity_reduce_type == 'selected_units_num':
            key_embeddings = key
        else:
            raise NotImplementedError

        return key, entity_mask, key_embeddings

    def _get_pred_with_logit(self, logit):
        dist = torch.distributions.Categorical(logits=logit)
        units = dist.sample()
        return units

    def _query(
            self, key: Tensor, entity_num: Tensor, autoregressive_embedding: Tensor, entity_mask: Tensor,
            key_embeddings: Tensor, temperature: float = 1):

        ae = autoregressive_embedding
        bs = ae.shape[0]  # batch size

        entity_mask[torch.arange(bs), entity_num] = False  # bs, n+1 cant choose entity with idx entity_num
        # cant choose end flag in the first iter
        # entity_mask[torch.arange(bs), entity_num] = torch.tensor([0], dtype=torch.bool, device=ae.device) # bs, n+1

        end_flag = torch.zeros(bs, dtype=torch.bool).to(ae.device)
        results_list, logits_list = [], []
        result: Optional[Tensor] = None
        selected_units_num = torch.ones(bs, dtype=torch.long, device=ae.device) * self.MAX_SELECTED_UNITS_NUM

        # initialize hidden state
        state = [(torch.zeros(bs, self.lstm_hidden_dim, device=ae.device),
                  torch.zeros(bs, self.lstm_hidden_dim, device=ae.device))
                 for _ in range(self.lstm_num_layers)]

        selected_units_one_hot = torch.zeros(*key_embeddings.shape[:2], device=ae.device)  # bs, n+1,1
        for i in range(self.MAX_SELECTED_UNITS_NUM):
            if i > 0:
                if i == 1:  # end flag can be selected at second selection
                    entity_mask[torch.arange(bs), entity_num] = True
                if result is not None:
                    entity_mask[torch.arange(bs), result.detach()] = False  # mask selected units
            lstm_input = self.query_mlp(ae).unsqueeze(0)  # 1, bs, lstm_hidden_size
            lstm_output, state = self.lstm(lstm_input, state)

            # dot product of queries and key -> logits
            queries = lstm_output.permute(1, 0, 2)  # 1, bs, lstm_hidden_size -> bs, 1,lstm_hidden_size
            # get logits
            step_logits = (queries * key).sum(dim=-1)  # b, n
            step_logits.div_(temperature)
            step_logits = step_logits.masked_fill(~entity_mask, -1e9)
            logits_list.append(step_logits)

            result = self._get_pred_with_logit(step_logits, )
            # if not end and choose end flag set selected units_num
            selected_units_num[(result == entity_num) * ~(end_flag)] = i + 1
            end_flag[result == entity_num] = True
            results_list.append(result)
            if self.entity_reduce_type == 'selected_units_num':
                # put selected_units in cut step to selected_units_on_hot
                selected_units_one_hot[torch.arange(bs)[~end_flag], result[~end_flag],] = 1
                slected_num = selected_units_one_hot.sum(dim=1)

                # take average of selected_units_embedding according to selected_units_num
                selected_units_emebedding = (key_embeddings * selected_units_one_hot.unsqueeze(-1)).sum(dim=1)
                selected_units_emebedding[slected_num != 0] = selected_units_emebedding[slected_num != 0] / \
                                                              slected_num[slected_num != 0].unsqueeze(dim=1)

                selected_units_emebedding = self.embed_mlp(selected_units_emebedding)
                ae = autoregressive_embedding + selected_units_emebedding
            else:
                ae = ae + key_embeddings[torch.arange(bs), result] * ~end_flag.unsqueeze(dim=1)
            if end_flag.all():
                break
        results = torch.stack(results_list, dim=1)
        logits = torch.stack(logits_list, dim=1)

        return logits, results, ae, selected_units_num

    def _train_query(
            self, key: Tensor, entity_num: Tensor, autoregressive_embedding: Tensor, entity_mask: Tensor,
            key_embeddings: Tensor, selected_units: Tensor, selected_units_num: Tensor, temperature: float = 1):
        ae = autoregressive_embedding
        bs = ae.shape[0]
        seq_len = selected_units_num.max()
        end_flag = torch.zeros(bs, dtype=torch.bool).to(ae.device)

        logits_list = []
        # end flag is not available at first selection
        entity_mask[torch.arange(bs), entity_num] = 0
        # entity_mask = entity_mask.repeat(max(seq_len, 1), 1, 1)  # b, n -> s, b, n
        selected_units_one_hot = torch.zeros(*key_embeddings.shape[:2], device=ae.device).unsqueeze(dim=2)

        # initialize hidden state
        state = [(torch.zeros(bs, self.lstm_hidden_dim, device=ae.device),
                  torch.zeros(bs, self.lstm_hidden_dim, device=ae.device))
                 for _ in range(self.lstm_num_layers)]

        for i in range(max(seq_len, 1)):
            if i > 0:
                # entity_mask[i] = entity_mask[i - 1]
                if i == 1:  # enable end flag
                    entity_mask[torch.arange(bs), entity_num] = 1
                entity_mask[torch.arange(bs), selected_units[:, i - 1]] = 0  # mask selected units
            lstm_input = self.query_mlp(ae).unsqueeze(0)
            lstm_output, state = self.lstm(lstm_input, state)

            queries = lstm_output.permute(1, 0, 2)  # 1, bs, lstm_hidden_size -> bs, 1,lstm_hidden_size
            step_logits = (queries.squeeze(0) * key).sum(dim=-1)  # b, n
            step_logits.div_(temperature)
            step_logits = step_logits.masked_fill(~entity_mask, -1e9)

            logits_list.append(step_logits)
            end_flag[selected_units[:, i] == entity_num] = 1

            if self.entity_reduce_type == 'selected_units_num' in self.entity_reduce_type:
                new_selected_units_one_hot = selected_units_one_hot.clone()  # inplace operation can not backward !!!

                # end flag is not included in selected_units_one_hot
                new_selected_units_one_hot[torch.arange(bs)[~end_flag], selected_units[:, i][~end_flag], :] = 1

                # take average of selected_units_embedding according to selected_units_nu
                selected_units_emebedding = (key_embeddings * new_selected_units_one_hot).sum(dim=1)
                selected_units_emebedding[selected_units_num != 0] = selected_units_emebedding[
                                                                         selected_units_num != 0] / \
                                                                     new_selected_units_one_hot.sum(dim=1)[
                                                                         selected_units_num != 0]
                selected_units_emebedding = self.embed_mlp(selected_units_emebedding)
                ae = autoregressive_embedding + selected_units_emebedding
                selected_units_one_hot = new_selected_units_one_hot.clone()
            else:
                ae = ae + key_embeddings[torch.arange(bs),
                                         selected_units[:, i]] * ((i +1)<= selected_units_num).unsqueeze(1)*(selected_units[:, i]!=entity_num).unsqueeze(1)
        logits = torch.stack(logits_list, dim=1)
        return logits, None, ae, selected_units_num

    def forward(
            self,
            embedding,
            entity_embedding,
            entity_num,
            selected_units=None,
            selected_units_num=None,
            temperature=1,
    ):
        key, entity_mask, key_embeddings = self._get_key_mask(entity_embedding, entity_num)
        if selected_units is not None and selected_units_num is not None:  # train
            logits, units, embedding, selected_units_num = self._train_query(
                key, entity_num, embedding, entity_mask, key_embeddings, selected_units, selected_units_num,
                temperature=temperature)
        else:
            logits, units, embedding, selected_units_num = self._query(
                key, entity_num, embedding, entity_mask, key_embeddings, temperature=temperature)
        return logits, units, embedding, selected_units_num


import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    from easydict import EasyDict

    default_model_config = EasyDict({'agent': {'features': {'max_selected_units_num': 64, 'max_entity_num': 512}},
                                     'model': {'input_dim': 1024,
                                               'entity_embedding_dim': 256, 'key_dim': 32, 'func_dim': 256,
                                               'lstm_hidden_dim': 32, 'lstm_num_layers': 1, 'max_entity_num': 64,
                                               'activation': 'relu', 'entity_reduce_type': 'selected_units_num',
                                               # ['constant', 'entity_num', 'selected_units_num'] only 'selected_units_num' works for now
                                               }
                                     }
                                    )
    net = PointerNetwork(default_model_config)
    input_dim = net.cfg.entity_embedding_dim
    batch_size = 10
    embedding = torch.rand(size=(batch_size, 1024,))
    MaxEntityNum = 512
    entity_num = torch.ones(size=(batch_size,)).long() * 512
    entity_embedings = torch.rand(size=(batch_size, MaxEntityNum, input_dim,))

    setup_seed(20)
    logits0, units, embedding0, selected_units_num = net.forward(embedding, entity_embedings, entity_num,
                                                                 temperature=0.8)

    setup_seed(20)
    logits1, _, embedding1, _ = net.forward(embedding, entity_embedings, entity_num, selected_units=units,
                                            selected_units_num=selected_units_num, temperature=0.8)

    print((logits1 - logits0).abs().max())
    print((embedding1 - embedding0).abs().max())
