from typing import List

import torch
from torch.utils.data import Dataset, DataLoader

# todo 优化内存
from dataset.vocab import Vocab


def read_word_data(path, word_vocab: Vocab):
    sentence = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            sentence.append(torch.LongTensor(word_vocab.toID(words)))
    return sentence


def read_label_data(path, label_vocab: Vocab):
    sentence = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            labels = line.strip().split()
            sentence.append(torch.LongTensor(label_vocab.toID(labels)))
    return sentence


class DataReader(Dataset):
    def __init__(self, word_path, label_path, word_vocab, label_vocab):
        self.word_tensor = read_word_data(word_path, word_vocab)
        self.label_tensor = read_label_data(label_path, label_vocab)

    def __getitem__(self, index):
        return self.word_tensor[index], self.label_tensor[index]

    def __len__(self) -> int:
        return len(self.word_tensor)


def pad_tensor(vec, pad, pad_id):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to

    return:
        a new tensor padded to 'pad'
    """
    return torch.cat([vec, torch.full([pad - len(vec)], pad_id, dtype=torch.long)], dim=0).data.numpy()


class Collate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, pred_id: List, word_pad=0, label_pad=0, sort=True):
        """

        :param pred_id: id of predicate label. 'B-v' or 'I-v'
        :param word_pad:
        :param label_pad:
        """
        self.word_pad = word_pad
        self.label_pad = label_pad
        self.pred_id = pred_id
        self.sort = sort

    def _collate(self, batch):
        """
        :param batch: list of (word, label). [(w,l),(w,l)].
            word/label: torch.LongTensor
        :return:
            xs/ys - a tensor of all labels in batch after padding like like:
                '''
                 [tensor([1,2,3,4,0]),
                  tensor([1,2,0,0,0]),
                  tensor([1,2,3,4,5])]
                '''
        """
        xs = [v[0] for v in batch]
        ys = [v[1] for v in batch]
        # 获得每个样本的序列长度
        lengths = [len(v) for v in xs]
        seq_lengths = torch.IntTensor(lengths)
        max_len = max(lengths)
        # 每个样本都padding到当前batch的最大长度
        xs = torch.LongTensor([pad_tensor(v, max_len, self.word_pad) for v in xs])
        ys = torch.LongTensor([pad_tensor(v, max_len, self.label_pad) for v in ys])
        # 把xs和ys按照序列长度从大到小排序
        if self.sort:
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            xs = xs[perm_idx]
            ys = ys[perm_idx]
        # predicate mask. set to 1 when the label is 'B-v', else 0
        preds = torch.LongTensor([[1 if id in self.pred_id else 0 for id in v] for v in ys])
        return xs, preds, ys, seq_lengths

    def __call__(self, batch):
        return self._collate(batch)


if __name__ == '__main__':
    word_vocab = Vocab('../../data/train/word_vocab.txt', 1)
    label_vocab = Vocab('../../data/train/label_vocab.txt')
    pred_id = [label_vocab.toID('B-v'), label_vocab.toID('I-v')]

    dataset = DataReader('../../data/train/word.txt', '../../data/train/label.txt', word_vocab, label_vocab)
    dataLoader = DataLoader(dataset=dataset, batch_size=32, num_workers=4, collate_fn=Collate(pred_id, 0, 0))
    for xs, preds, ys, lengths in dataLoader:
        print(ys)
