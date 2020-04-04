import argparse
import collections
import time
from typing import List


class BuildVocab(object):
    def __init__(self, data_path, out_path, char_path):
        self.data_path = data_path
        self.out_path = out_path
        self.char_path = char_path

    def count(self, data_path, limit=0):
        counter = collections.Counter()
        char = collections.Counter()

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                counter.update(parts)

        vocab = []
        for key, value in counter.items():
            if value >= limit:
                vocab.append(key)
            elif self.char_path is not None:
                char.update([k for k in key])
        return vocab, len(counter), list(char.keys())

    def split_label(self, label: str):
        parts = label.split('-')
        if len(parts) >= 3:
            return '-'.join(parts[:2])
        return label

    def save_file(self, out_path, vocab: List):
        with open(out_path, 'w', encoding='utf-8') as f:
            for v in vocab:
                f.write(v)
                f.write('\n')

    def build(self, special=None, limit=None):
        print('build vocab: ' + self.data_path)
        start = time.time()
        if special is None:
            special = list()

        vocab, size, char = self.count(self.data_path, limit)
        vocab = special.copy() + vocab

        self.save_file(self.out_path, vocab)
        if self.char_path is not None:
            self.save_file(self.char_path, char)
        print('finish vocab, coverage: %.2f%%' % ((len(vocab) - len(special)) / size * 100))


def parse_args():
    msg = "build vocabulary"
    parser = argparse.ArgumentParser(description=msg)

    msg = "input data path"
    parser.add_argument("input", help=msg)
    msg = "output vocabulary file path"
    parser.add_argument("output", help=msg)
    msg = "the min occurrence of vocabulary"
    parser.add_argument("--limit", default=0, type=int, help=msg)
    msg = "add special token, separated by colon"
    parser.add_argument("--special", default=None, help=msg)
    msg = "path to save character level embedding"
    parser.add_argument("--char", default=None, help=msg)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    special = args.special
    if special is not None:
        special = [s for s in special.strip().split(':')]
    build = BuildVocab(args.input, args.output, args.char)
    build.build(special, args.limit)

# if __name__ == '__main__':
#     build = BuildVocab('data/train/label.txt', 'data/train/label_vocab.txt')
#     build.build(['<pad>'], 0, True)
