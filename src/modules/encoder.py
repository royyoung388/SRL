from torch import nn

from modules.encoderlayer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, model_dim, filter_dim, layer_num):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(model_dim, filter_dim) for i in range(layer_num)])

    def forward(self, inputs, attn_mask):
        for layer in self.layers:
            inputs = layer(inputs, attn_mask)
        return inputs