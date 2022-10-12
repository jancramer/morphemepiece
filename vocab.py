

from typing import List


class Vocab (object):

    def __init__(self,vocabulary:list, vocab_split, is_cased:bool) -> None:
        self.vocabulary= vocabulary 
        self.vocab_split=vocab_split
        self.is_cased=is_cased

    def process_vocab(self, raw_vocab):
        pass