from .data import JSONSerializable


class Vocabulary(JSONSerializable):
    pad_token, pad_token_id = '<pad>', 0
    unk_token, unk_token_id = '<unk>', 1
    start_token, start_token_id = pad_token, pad_token_id
    end_token, end_token_id = pad_token, pad_token_id

    def __init__(self):
        super().__init__()
        self.word2idx = {Vocabulary.pad_token: Vocabulary.pad_token_id,
                         Vocabulary.unk_token: Vocabulary.unk_token_id}
        self.idx2word = [Vocabulary.pad_token, Vocabulary.unk_token]

    def __len__(self):
        return len(self.word2idx)

    def get_index(self, word):
        if word not in self.word2idx:
            return Vocabulary.unk_token_id
        return self.word2idx[word]

    def get_word(self, index):
        index = int(index)
        if index < 0 or index >= len(self.idx2word):
            return Vocabulary.unk_token
        return self.idx2word[index]

    def _add_word(self, word):
        assert word not in self.word2idx
        assert len(self.word2idx) == len(self.idx2word)
        self.word2idx[word] = len(self.idx2word)
        self.idx2word.append(word)

JSONSerializable.register_cls(Vocabulary, 'Vocabulary')