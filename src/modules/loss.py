import math

import torch
import torch.nn.functional as F
from torch import nn


class SmoothedCrossEntropyLoss(nn.Module):

    def __init__(self, smoothing=0.0, normalize=True):
        super(SmoothedCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.normalize = normalize

    def forward(self, logits, labels):
        """

        :param logits: (batch * seq_len * label_size). model output.
        :param labels: label tensor after padded. every element is label id. (batch * seq_len)
                tensor([[7,2,6,4,0]),
                        [5,2,0,0,0],
                        [1,5,4,2,1]])
        :return:
        """
        # # smoothed probability distribution. dist_i = 0 when dist_i is padded
        # # if i==true label, dist_i=1-smoothing, else dist_i=smoothing/size-1.
        # dist = torch.full(logits.shape, self.smoothing / (logits.size(-1) - 1))
        # for batch, seq in enumerate(dist):
        #     for i, word in enumerate(seq):
        #         label = labels[batch]
        #         if i < len(label):
        #             # true label, 1-smoothing
        #             word[label[i]] = 1 - self.smoothing
        #         else:
        #             # padded, 0
        #             word[:] = 0.0
        #
        # loss = F.log_softmax(logits)
        # loss =

        shape = labels.shape
        # (batch * seq_len, label_size)
        logits = torch.reshape(logits, [-1, logits.shape[-1]])
        # (batch * seq_len)
        labels = torch.reshape(labels, [-1])

        log_probs = F.log_softmax(logits, dim=-1)

        batch_idx = torch.arange(labels.shape[0], device=logits.device)
        loss = log_probs[batch_idx, labels]

        if not self.smoothing:
            return -torch.reshape(loss, shape)

        n = logits.shape[-1] - 1.0
        p = 1.0 - self.smoothing
        q = self.smoothing / n

        sum_probs = torch.sum(log_probs, dim=-1)
        loss = p * loss + q * (sum_probs - loss)

        loss = -torch.reshape(loss, shape)

        if self.normalize:
            normalizing = -(p * math.log(p) + n * q * math.log(q + 1e-20))
            return loss - normalizing
        else:
            return loss
