import argparse
import time

import torch
from seqeval.metrics import *
from torch.utils.data import DataLoader

import config
from config import *
from dataset.datareader import DataReader, Collate
from dataset.vocab import Vocab
from model.deepattn import DeepAttn


class Predictor(object):
    def __init__(self, model, word_vocab, label_vocab, word, label):
        # load vocab
        self.word_vocab = Vocab(word_vocab)
        self.label_vocab = Vocab(label_vocab)

        self.word_vocab.unk_id = self.word_vocab.toID(UNK)
        # todo just for test
        self.label_vocab.unk_id = 1
        config.WORD_PAD_ID = self.word_vocab.toID(PAD)
        config.WORD_UNK_ID = self.word_vocab.toID(UNK)
        config.LABEL_PAD_ID = self.label_vocab.toID(PAD)
        pred_id = [self.label_vocab.toID('B-v')]

        # load data
        dataset = DataReader(word, label, self.word_vocab, self.label_vocab)
        self.dataLoader = DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     pin_memory=True,
                                     shuffle=False,
                                     collate_fn=Collate(pred_id, WORD_PAD_ID, LABEL_PAD_ID, False))

        self.model = DeepAttn(self.word_vocab.size(), self.label_vocab.size(), feature_dim, model_dim, filter_dim)
        self.model.load_state_dict(torch.load(model))

    def save(self, path, labels):
        with open(path, 'a', encoding='utf-8') as f:
            for label in labels:
                f.write(' '.join(label))
                f.write('\n')

    def predict(self, save_path):
        start = time.time()
        print('start predict')

        self.model.eval()
        with torch.no_grad():
            y_pred = []
            y_true = []
            for step, (xs, preds, ys, lengths) in enumerate(self.dataLoader):
                y_true.extend(convert_to_string(ys.squeeze().tolist(), self.label_vocab, lengths))

                labels = self.model.argmax_decode(xs, preds)
                labels = convert_to_string(labels.squeeze().tolist(), self.label_vocab, lengths)
                y_pred.extend(labels)

                print('predict step: %d, time: %.2f M' % (step, (time.time() - start) / 60))
                print('current F1 score: %.2f' % f1_score(y_true, y_pred))

            self.save(save_path, y_pred)

            score = f1_score(y_true, y_pred)

            print('finish predict: %.2f' % ((time.time() - start) / 60))
            print('F1 score: %.2f' % score)
            return score


def convert_to_string(labels, label_vocab: Vocab, lengths: torch.Tensor):
    """

    :param labels: (batch * seq)
    :param label_vocab:
    :param lengths: IntTensor. batch size
    :return:
    """
    result = []
    for label, length in zip(labels, lengths):
        result.append(label_vocab.toToken(label)[:length])
    return result


def parse_args():
    msg = "predict labels"
    parser = argparse.ArgumentParser(description=msg)

    msg = "model path"
    parser.add_argument("--model", default='result/3/model.pt', help=msg)
    msg = 'word vocab path'
    parser.add_argument("--word_vocab", default='data/train/word_vocab.txt', help=msg)
    msg = 'label vocab path'
    parser.add_argument("--label_vocab", default='data/train/label_vocab.txt', help=msg)
    msg = 'word path'
    parser.add_argument("--word", default='data/dev/word.txt', help=msg)
    msg = 'label path'
    parser.add_argument("--label", default='data/dev/label.txt', help=msg)
    msg = 'label output path'
    parser.add_argument("--output", default='data/dev/label_out.txt', help=msg)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    predictor = Predictor(args.model, args.word_vocab, args.label_vocab, args.word, args.label)
    predictor.predict(args.output)
