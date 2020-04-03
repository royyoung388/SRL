from torch import nn

from config import *
from modules.attention import MultiHeadAttention
from modules.feedforward import FeedForward
from modules.container import SublayerContainer


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, filter_dim):
        super(EncoderLayer, self).__init__()
        self.layers = nn.ModuleList([SublayerContainer(model_dim, residual_dropout) for i in range(2)])
        self.ff = FeedForward(model_dim, filter_dim, model_dim, relu_dropout)
        self.self_attn = MultiHeadAttention(head_num, model_dim, attention_dropout)

    def forward(self, inputs, attn_mask):
        inputs = self.layers[0](inputs, self.ff)
        inputs = self.layers[1](inputs, lambda x: self.self_attn(x, x, x, attn_mask))
        return inputs
