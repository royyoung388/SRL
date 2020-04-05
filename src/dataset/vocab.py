class Vocab(object):
    def __init__(self, vocab_path, unk_id=None):
        self.token2id = {}
        self.id2token = {}
        self.unk_id = unk_id
        self.load_vocab(vocab_path)

    def size(self):
        return len(self.token2id)

    def load_vocab(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                cur_id = len(self.token2id)
                self.token2id[line] = cur_id
                self.id2token[cur_id] = line

    def toID(self, tokens):
        if type(tokens) == list:
            return [self.look_up(t, self.token2id) for t in tokens]
        else:
            return self.look_up(tokens, self.token2id)

    def toToken(self, ids):
        if type(ids) == list:
            return [self.look_up(i, self.id2token) for i in ids]
        else:
            return self.look_up(ids, self.id2token)

    def look_up(self, key, dict):
        if key not in dict:
            # print(key)
            assert self.unk_id is not None
            return self.unk_id
        else:
            return dict[key]
