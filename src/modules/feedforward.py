import torch.nn.functional as F
from torch import nn

from modules.affine import Affine


class FeedForward(nn.Module):

    def __init__(self, input_dim, filter_dim, output_dim, dropout=0.0):
        super(FeedForward, self).__init__()
        self.in_linear = Affine(input_dim, filter_dim)
        self.out_linear = Affine(filter_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self, initializer="orthogonal"):
        if initializer == "orthogonal":
            self.in_linear.orthogonal_initialize()
            self.out_linear.orthogonal_initialize()
        else:
            nn.init.xavier_uniform_(self.in_linear.weight)
            nn.init.xavier_uniform_(self.out_linear.weight)
            nn.init.constant_(self.in_linear.bias, 0.0)
            nn.init.constant_(self.out_linear.bias, 0.0)

    def forward(self, x):
        return self.out_linear(self.dropout(F.relu(self.in_linear(x), inplace=True)))
