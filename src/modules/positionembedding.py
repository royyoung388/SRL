import math

import torch
from torch import nn


class PositionEmbedding(nn.Module):

    def __init__(self, model_dim, dropout, max_seq=500):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_seq, model_dim)
        position = torch.arange(0, max_seq).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * -(math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1 * seq_index * dim_index)
        pe = pe.unsqueeze(0)
        pe.requires_grad_(False)
        self.register_buffer('pe', pe)

    def forward(self, embeddings: torch.Tensor):
        """
        
        :param embeddings: word embeddings. (batch * seq_len * model_dim)
        :return:
        """
        embeddings = embeddings + self.pe[:, :embeddings.size(1)]
        return self.dropout(embeddings)
