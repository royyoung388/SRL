import argparse

import torch
from torch.utils.data import DataLoader

from config import *
from dataset.datareader import DataReader, Collate
from dataset.vocab import Vocab
from model.deepattn import DeepAttn


def convert_to_string(labels, label_vocab: Vocab):
    return [label_vocab.toToken(label) for label in labels]

def parse_args():
    msg = "predict.sh labels"
    parser = argparse.ArgumentParser(description=msg)

    msg = "model path"
    parser.add_argument("--model", default='result/30/model.pt', help=msg)
    msg = 'word vocab path'
    parser.add_argument("--word_vocab", default='data/train/word_vocab.txt', help=msg)
    msg = 'label vocab path'
    parser.add_argument("--label_vocab", default='data/train/label_vocab.txt', help=msg)
    msg = 'word path'
    parser.add_argument("--word", default='data/test/word.txt', help=msg)
    msg = 'label path'
    parser.add_argument("--label", default='data/test/label.txt', help=msg)
    msg = 'label output path'
    parser.add_argument("--output", default='data/test/label_out.txt', help=msg)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # load vocab
    word_vocab = Vocab(args.word_vocab)
    word_vocab.unk_id = word_vocab.toID(UNK)
    label_vocab = Vocab(args.label_vocab)
    WORD_PAD_ID = word_vocab.toID(PAD)
    WORD_UNK_ID = word_vocab.toID(UNK)
    LABEL_PAD_ID = label_vocab.toID(PAD)
    pred_id = [label_vocab.toID('B-v'), label_vocab.toID('I-v')]

    # load data
    dataset = DataReader(args.word, args.label, word_vocab, label_vocab)
    dataLoader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            shuffle=False,
                            collate_fn=Collate(pred_id, WORD_PAD_ID, LABEL_PAD_ID, False))

    # model = DeepAttn(word_vocab.size(), label_vocab.size(), feature_dim, model_dim, filter_dim)
    # model.load_state_dict(torch.load(args.model))
    model = torch.load(args.model)

    model.eval()
    with torch.no_grad(), open(args.output, 'a', encoding='utf-8') as f:
        for step, (xs, preds, ys, lengths) in enumerate(dataLoader):
            labels = model.argmax_decode(xs, preds)
            labels = convert_to_string(labels, label_vocab)
            f.write(' '.join(label) for label in labels)
