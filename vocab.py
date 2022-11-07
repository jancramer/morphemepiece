from typing import List


class Vocab(object):
    

    def __init__(self, vocabulary: list, vocab_split, is_cased: bool) -> None:
        self.vocabulary = vocabulary
        self.vocabulary_set = set(vocabulary)
        self.vocab_split = vocab_split
        self.vocab_split['prefixes'] = set(self.vocab_split['prefixes'])
        self.vocab_split['suffixes'] = set(self.vocab_split['suffixes'])
        self.vocab_split['words'] = set(self.vocab_split['words'])
        self.is_cased = is_cased

    def process_vocab(self, raw_vocab):
        pass
