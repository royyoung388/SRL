import math

import torch
import torch.nn as nn


class Affine(nn.Module):

    def __init__(self, input_dim, output_dim, bias=True):
        super(Affine, self).__init__()
        self.in_features = input_dim
        self.out_features = output_dim

        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def orthogonal_initialize(self, gain=1.0):
        nn.init.orthogonal_(self.weight, gain)
        nn.init.zeros_(self.bias)

    def forward(self, inputs):
        return nn.functional.linear(inputs, self.weight, self.bias)
