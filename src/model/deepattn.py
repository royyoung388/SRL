import torch
from torch import nn

from config import *
from modules.affine import Affine
from modules.encoder import Encoder
from modules.loss import SmoothedCrossEntropyLoss
from modules.positionembedding import PositionEmbedding


class DeepAttn(nn.Module):
    def __init__(self, vocab_size, label_size, feature_dim, model_dim, filter_dim):
        super(DeepAttn, self).__init__()
        self.feature_dim = feature_dim
        self.model_dim = model_dim

        self.word_embed = nn.Embedding(vocab_size, feature_dim)
        self.pred_embed = nn.Embedding(2, feature_dim)
        self.position_embed = PositionEmbedding(model_dim, residual_dropout)

        self.encoder = Encoder(model_dim, filter_dim, layer_num)

        self.bias = torch.nn.Parameter(torch.zeros([model_dim]), requires_grad=True)
        self.project = Affine(model_dim, label_size)

        self.criterion = SmoothedCrossEntropyLoss(label_smoothing)

    def load_pre_trained(self, path, feature_dim:int):
        with open(path, encoding='utf-8') as f:
            size = f.readline().strip()
            del self.word_embed
            self.word_embed = nn.Embedding(int(size), feature_dim)
            for line in f.readlines():
                self.word_embed.from_pretrained()


    def reset_parameters(self):
        nn.init.normal_(self.word_embed, mean=0.0,
                        std=self.feature_dim ** -0.5)
        nn.init.normal_(self.pred_embed, mean=0.0,
                        std=self.feature_dim ** -0.5)
        nn.init.normal_(self.project.weight, mean=0.0,
                        std=self.model_dim ** -0.5)
        nn.init.zeros_(self.project.bias)

    def encode(self, words, preds):
        # 1. attention pad mask. 1(is padded) / 0(not padded)
        # (batch * 1 * seq_len)
        attn_mask = torch.ne(words, WORD_PAD_ID).float().unsqueeze(-2)

        # 2. ids to embedding
        word_embed = self.word_embed(words)
        pred_embed = self.pred_embed(preds)
        inputs = torch.cat([word_embed, pred_embed], -1)
        inputs = inputs * model_dim ** 0.5
        # todo bias有什么用
        inputs = inputs + self.bias

        # 3. position embedding
        inputs = self.position_embed(inputs)

        # 4. forward
        output = self.encoder(inputs, attn_mask)
        logits = self.project(output)
        return logits

    def forward(self, words, preds, labels):
        """
        :param words: word ids after pad. (batch * seq_len)
        :param preds: predicate mask. 1(is predicate) / 0(not predicate)
        :param labels: label ids after pad. (batch * seq_len)
        :return:
        """
        # not pad mask (true=1, false=0)
        mask = torch.ne(words, 0).float()
        logits = self.encode(words, preds)
        loss = self.criterion(logits, labels)
        return torch.sum(loss * mask) / torch.sum(mask)

    def argmax_decode(self, words, preds):
        logits = self.encode(words, preds)
        return torch.argmax(logits, -1)