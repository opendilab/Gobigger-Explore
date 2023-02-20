import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_module import fc_block
from .normalization import build_normalization


class Attention(nn.Module):
    def __init__(self, input_dim, head_dim, output_dim, head_num, dropout):
        super(Attention, self).__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.dropout = dropout
        self.attention_pre = fc_block(input_dim, head_dim * head_num * 3)  # query, key, value
        self.project = fc_block(head_dim * head_num, output_dim)

    def split(self, x, T: bool = False):
        B, N = x.shape[:2]
        x = x.view(B, N, self.head_num, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()  # B, head_num, N, head_dim
        if T:
            x = x.permute(0, 1, 3, 2).contiguous()
        return x

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Overview:
            x: [batch_size, seq_len, embeddding_size]
        """
        assert (len(x.shape) == 3)
        B, N = x.shape[:2]
        x = self.attention_pre(x)
        query, key, value = torch.chunk(x, 3, dim=2)
        query, key, value = self.split(query), self.split(key, T=True), self.split(value)

        score = torch.matmul(query, key)  # B, head_num, N, N
        score /= math.sqrt(self.head_dim)
        if mask is not None:
            score.masked_fill_(~mask, value=-1e9)

        score = F.softmax(score, dim=-1)
        if self.dropout:
            score = self.dropout(score)
        attention = torch.matmul(score, value)  # B, head_num, N, head_dim

        attention = attention.permute(0, 2, 1, 3).contiguous()  # B, N, head_num, head_dim
        attention = self.project(attention.view(B, N, -1))  # B, N, output_dim
        return attention


class TransformerLayer(nn.Module):
    def __init__(self, input_dim, head_dim, hidden_dim, output_dim, head_num, mlp_num, dropout, activation, ln_type):
        super(TransformerLayer, self).__init__()
        self.attention = Attention(input_dim, head_dim, output_dim, head_num, dropout)
        self.layernorm1 = build_normalization('LN')(output_dim)
        self.dropout = dropout
        layers = []
        dims = [output_dim] + [hidden_dim] * (mlp_num - 1) + [output_dim]
        for i in range(mlp_num):
            layers.append(fc_block(dims[i], dims[i + 1], activation=activation))
        if self.dropout:
            layers.append(self.dropout)
        self.mlp = nn.Sequential(*layers)
        self.layernorm2 = build_normalization('LN')(output_dim)
        self.ln_type = ln_type

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        if self.ln_type == 'post':
            a = self.attention(x, mask)
            if self.dropout:
                a = self.dropout(a)
            x = self.layernorm1(x + a)
            m = self.mlp(x)
            if self.dropout:
                m = self.dropout(m)
            x = self.layernorm2(x + m)
        elif self.ln_type == 'pre':
            a = self.attention(self.layernorm1(x), mask)
            if self.dropout:
                a = self.dropout(a)
            x = x + a
            m = self.mlp(self.layernorm2(x))
            if self.dropout:
                m = self.dropout(m)
            x = x + m
        else:
            raise NotImplementedError(self.ln_type)
        return x, mask


class Transformer(nn.Module):
    '''
        Note:
          Input has passed through embedding
    '''

    def __init__(
            self,
            input_dim,
            head_dim=128,
            hidden_dim=1024,
            output_dim=256,
            head_num=2,
            mlp_num=2,
            layer_num=3,
            pad_val=0,
            dropout_ratio=0.0,
            activation=nn.ReLU(),
            ln_type='pre'
    ):
        super(Transformer, self).__init__()
        self.embedding = fc_block(input_dim, output_dim, activation=activation)
        self.pad_val = pad_val
        self.act = activation
        layers = []
        dims = [output_dim] + [output_dim] * layer_num
        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)
        else:
            self.dropout = None
        for i in range(layer_num):
            layers.append(
                TransformerLayer(
                    dims[i], head_dim, hidden_dim, dims[i + 1], head_num, mlp_num, self.dropout, self.act, ln_type
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            mask = mask.unsqueeze(dim=1).repeat(1, mask.shape[1], 1).unsqueeze(dim=1)
        x = self.embedding(x)
        if self.dropout:
            x = self.dropout(x)
        for m in self.layers:
            x, mask = m(x, mask)
        return x
