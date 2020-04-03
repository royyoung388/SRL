import os


def read(path):
    with open(path, 'r', encoding='utf-8') as f_in, open('special.txt', 'a', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            for c in line:
                # not chinese
                if c < u'\u4e00' or c > u'\u9fa5':
                    f_out.write(line)
                    f_out.write('\n')
                    break


def find(string, path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if 'conll' not in file:
                continue
            p = os.path.join(root, file)
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    if string in line:
                        print(p)
                        break


if __name__ == '__main__':
    # read('train/word_vocab.txt')
    # read('dev/word_vocab.txt')
    # read('test/word_vocab.txt')
    find('pali', 'E:/SRL/conll-2012/v4/data/train/data/chinese')
    find('pali', 'E:/SRL/conll-2012/v4/data/development/data/chinese')
    find('pali', 'E:/SRL/conll-2012/v9/data/test/data/chinese')
