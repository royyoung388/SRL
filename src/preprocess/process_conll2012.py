import argparse
import os
from typing import List

from subword import SubWord


class Process_conll2012(object):
    def __init__(self, out_word_file, out_pos_file, out_srl_file):
        self.out_word_path = out_word_file
        self.out_pos_path = out_pos_file
        self.out_srl_path = out_srl_file

        self.check_and_clear_file(out_word_file)
        self.check_and_clear_file(out_pos_file)
        self.check_and_clear_file(out_srl_file)

    def check_and_clear_file(self, path):
        if not os.path.isfile(path):
            open(path, 'w', encoding='utf-8').close()
        else:
            with open(path, 'r+')as f:
                f.truncate()

    def read_file(self, file):
        sentences = []
        with open(file, 'r', encoding='utf-8') as f:
            word, pos, label = [], [], []

            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue

                if len(line) <= 0:
                    # end of a sentence
                    if len(word) > 0:
                        label_num = len(label[0])
                        # don't have labels
                        if label_num == 0:
                            label = [['*'] * len(word)]
                        else:
                            label = [[l[i].lower() for l in label] for i in range(label_num)]

                        label = self.convert2BIO(label)
                        sentences.append({'word': word, 'pos': pos, 'label': label})
                    word, pos, label = [], [], []
                    continue

                parts = line.split()
                assert len(parts) >= 12

                # skip the 'empty' line
                if parts[3] == bytes.fromhex('efbca5efbcadefbcb0efbcb4efbcb9').decode('utf-8'):
                    continue

                w = parts[3]
                if args.lowcase:
                    w = w.lower()
                if args.subword:
                    w = SubWord.subword(w)
                word.append(w)
                pos.append(parts[4].lower())
                label.append(parts[11:-1])
        return sentences

    def process_label(self, label: str):
        """
        process for special label
        :param label:
        :return:
        """
        if label.startswith('c-') or label.startswith('r-'):
            return label[2:]
        elif label.startswith('rel-'):
            return label[:3]
        return label

    def convert2BIO(self, label):
        """

        :param label: list of list
        :return:
        """
        bio = []
        for t in label:
            item = []
            flag = '*'
            tag = self.process_label

            for i in t:
                if i.startswith('('):
                    if i.endswith(')'):
                        item.append('B-' + tag(i[1:-2]))
                    else:
                        flag = i[1:-1]
                        item.append('B-' + tag(flag))
                elif i == '*':
                    if flag == '*':
                        item.append('O')
                    else:
                        item.append('I-' + tag(flag))
                elif i == '*)':
                    item.append('B-' + tag(flag))
                    flag = '*'
                else:
                    print('unexpected label: ' + i)

            bio.append(item)
        return bio

    def write_file(self, file, content: List, other: str = None):
        with open(file, 'a', encoding='utf-8') as f:
            if other is not None:
                f.write(other)
                f.write(' ')

            for i in content:
                f.write(i)
                f.write(' ')
            f.write('\n')

    def process(self, data_path, test, trial=-1):
        counter = 0

        for root, dirs, files in os.walk(data_path):
            postfix = 'gold_conll'
            if test:
                postfix = 'auto_conll'

            for file in files:
                if postfix in os.path.splitext(file)[-1]:
                    if counter == trial:
                        return
                    counter += 1
                    print(os.path.join(root, file))

                    name = os.path.join(root, file)
                    sentences = self.read_file(name)

                    for sentence in sentences:
                        for label in sentence['label']:
                            self.write_file(self.out_word_path, sentence['word'])
                            self.write_file(self.out_pos_path, sentence['pos'])
                            self.write_file(self.out_srl_path, label)


def parse_args():
    msg = "process conll2012 dataset"
    parser = argparse.ArgumentParser(description=msg)

    msg = "conll2012 data path"
    parser.add_argument("conll_path", help=msg)
    msg = "output dir path"
    parser.add_argument("output", default="data", help=msg)
    msg = 'test dataset tag'
    parser.add_argument('--test', action='store_true', help=msg)
    msg = 'count of files for trial'
    parser.add_argument('--trial', type=int, default=-1, help=msg)
    msg = 'subword'
    parser.add_argument('--subword', action='store_true', help=msg)
    msg = 'Ignore case '
    parser.add_argument('--lowcase', action='store_true', help=msg)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    processer = Process_conll2012(os.path.join(args.output, 'word.txt'),
                                  os.path.join(args.output, 'pos.txt'),
                                  os.path.join(args.output, 'label.txt'))
    processer.process(args.conll_path, args.test, trial=args.trial)

# if __name__ == '__main__':
#     processer = Process_conll2012('../data/word.txt', '../data/pos.txt', '../data/label.txt')
#     processer.process('E:/SRL/conll-2012/v4/data/train/data/chinese', debug=2)
