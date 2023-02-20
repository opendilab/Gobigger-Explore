import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_module import fc_block


class AttentionPool(nn.Module):
    def __init__(self, key_dim, head_num, output_dim, max_num=None):
        super(AttentionPool, self).__init__()
        self.queries = torch.nn.Parameter(torch.zeros(1, 1, head_num, key_dim))
        torch.nn.init.xavier_uniform_(self.queries)
        self.head_num = head_num
        self.add_num = False
        if max_num is not None:
            self.add_num = True
            self.num_ebed = torch.nn.Embedding(num_embeddings=max_num, embedding_dim=output_dim)
        self.embed_fc = fc_block(key_dim * self.head_num, output_dim)

    def forward(self, x, num=None, mask=None):
        assert len(x.shape) == 3  # batch size, tokens, channels
        x_with_head = x.unsqueeze(dim=2)  # add head dim
        score = x_with_head * self.queries
        score = score.sum(dim=3)  # b, t, h
        if mask is not None:
            assert len(mask.shape) == 3 and mask.shape[-1] == 1
            mask = mask.repeat(1, 1, self.head_num)
            score.masked_fill_(~mask.bool(), value=-1e9)
        score = F.softmax(score, dim=1)
        x = x.unsqueeze(dim=3).repeat(1, 1, 1, self.head_num)  # b, t, c, h
        score = score.unsqueeze(dim=2)  # b, t, 1, h
        x = x * score
        x = x.sum(dim=1)  # b, c, h
        x = x.view(x.shape[0], -1)  # b, c * h
        x = self.embed_fc(x)  # b, c
        if self.add_num:
            x = x + F.relu(self.num_ebed(num.long()))
        x = F.relu(x)
        return x