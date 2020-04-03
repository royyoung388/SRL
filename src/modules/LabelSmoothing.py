import torch
from torch import nn
from torch.autograd import Variable


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, logits, target):
        """
        
        :param logits: predict.sh tensor. (batch * seq_len * label_size)
        :param target: list of label tensor. every element is label id
                [tensor([0,2,6,4]),
                  tensor([5,2]),
                  tensor([1,5,4,2,1])]
        :return: 
        """
        assert logits.size(-1) == self.size
        true_dist = logits.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, :, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(logits, Variable(true_dist, requires_grad=False))
