import math

import torch
import torch.nn.functional as F
from torch import nn

from modules.affine import Affine


class MultiHeadAttention(nn.Module):
    def __init__(self, head, model_dim, dropout=0.0):
        """
        :param head: number of heads
        :param model_dim: model size
        :param dropout:
        """
        super(MultiHeadAttention, self).__init__()
        assert model_dim % head == 0
        # We assume d_v always equals d_k
        self.d_k = model_dim // head
        self.h = head
        self.q_linear = Affine(model_dim, model_dim)
        self.k_linear = Affine(model_dim, model_dim)
        self.v_linear = Affine(model_dim, model_dim)
        self.out_linear = Affine(model_dim, model_dim)
        self.attn = None
        self.attn_dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self, initializer="orthogonal"):
        if initializer == "orthogonal":
            self.q_linear.orthogonal_initialize()
            self.k_linear.orthogonal_initialize()
            self.v_linear.orthogonal_initialize()
            self.out_linear.orthogonal_initialize()
        else:
            nn.init.xavier_uniform_(self.q_linear.weight)
            nn.init.xavier_uniform_(self.k_linear.weight)
            nn.init.xavier_uniform_(self.v_linear.weight)
            nn.init.xavier_uniform_(self.out_linear.weight)
            nn.init.constant_(self.q_linear.bias, 0.0)
            nn.init.constant_(self.k_linear.bias, 0.0)
            nn.init.constant_(self.v_linear.bias, 0.0)
            nn.init.constant_(self.out_linear.bias, 0.0)

    def attention(self, query, key, value, attn_mask=None):
        """
        Compute 'Scaled Dot Product Attention'
        :param query:
        :param key:
        :param value:
        :param attn_mask: pad mask. 1(is pad) / 0(not pad)
        :return:
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        return torch.matmul(attn, value), attn

    def forward(self, query, key, value, attn_mask=None):
        """

        :param query:
        :param key:
        :param value:
        :param attn_mask: (batch * 1 * seq_len
        :return:
        """
        if attn_mask is not None:
            # Same mask applied to all h heads.
            attn_mask = attn_mask.unsqueeze(1)
        batch = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.q_linear(query).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        key = self.k_linear(key).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        value = self.v_linear(value).view(batch, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, attn_mask=attn_mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_k)
        return self.out_linear(x)
