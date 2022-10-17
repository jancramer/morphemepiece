from tokenizer import MorphemepieceTokenizer
from vocab import Vocab
import pandas as pd
import pytest
from transformers import BatchEncoding

# importing the data
vocabulary = pd.read_csv("./data/vocabulary.csv")["x"].to_list()
lookup = pd.read_csv("./data/lookup.csv").set_index("y").to_dict()["x"]
prefixes = pd.read_csv("./data/prefixes.csv")["x"].to_list()
words = pd.read_csv("./data/words.csv")["x"].to_list()
suffixes = pd.read_csv("./data/suffixes.csv")["x"].to_list()
vocab_split = {'prefixes': prefixes, 'words': words, 'suffixes': suffixes}
vocab = Vocab(vocabulary, vocab_split, True)

# creating the tokenizer
tokenizer = MorphemepieceTokenizer(vocab=vocab, lookup=lookup)


def test_call():
    sentence = "i use this sentence to check if everything works fine"
    expected = BatchEncoding(data={'overflowing_tokens': [5449, 3093, 6919, 3025, 4421, 7162, 4035, 7155],
                                   'num_truncated_tokens': 8,
                                   'input_ids': [3034, 3118, 12515, 10862, 3056],
                                   'token_type_ids': [0, 0, 0, 0, 0],
                                   'attention_mask': [1, 1, 1, 1, 1]})
    assert tokenizer.__call__(sentence, truncation=True, max_length=5, return_overflowing_tokens=True, vocab=vocab,
                              lookup=lookup) == expected


def test_tokenizer():
    sentence = "it is totally normal to be indistinguishable"
    expected = ['it', 'is', 'total', '##ly', 'normal', 'to', 'be', 'in##', 'distinguish', '##able']
    assert tokenizer.tokenize(sentence, vocab, lookup) == expected

    sentence = "let's test some compounds and puntuation here in this fine-grained testcase!?"
    expected = ['let', "##'", '##s', 'test', 'some', 'compound', '##s', 'and', 'punt', '##ua', '##tion', 'here', 'in',
                'this', 'fine', '-', 'grain', '##ed', 'test', '##', 'case', '!', '?']
    assert tokenizer.tokenize(sentence, vocab, lookup) == expected

def test_tokenize_irregular_plural():
    sentence = "foxes"
    expected = ['fox', '##s']
    assert tokenizer.tokenize(sentence, vocab, lookup) == expected

    sentence = "running"
    expected = ['run', '##ing']
    assert tokenizer.tokenize(sentence, vocab, lookup) == expected


def test_unknown_token():
    sentence = "longfakewordxzz"
    expected = ['[UNK]']
    assert tokenizer.tokenize(sentence, vocab, lookup, max_chars=7) == expected


def test_decode_irregular_plural():
    tokens = [3240, 4035]
    expected = "foxes"
    assert tokenizer.decode(tokens) == expected

def test_compound():
    sentence = "chairball"
    expected = ['chair', '##', 'ball']
    assert tokenizer.tokenize(sentence, vocab, lookup) == expected


def test_unarcher():
    sentence = "unarcher"
    expected = ["un##", "archer"]
    assert tokenizer.tokenize(sentence, vocab, lookup) == expected


def test_special_tokens_in_sequence():
    expected = ['[CLS]', 'hope', '##ful', '##ly', 'this', '[MASK]', 'work', '##s', 'as', 'intend', '##ed', 'to', 'get',
                '[MASK]', 'inform', '##ation', '[PAD]', '[PAD]']
    sentence = f"{tokenizer.cls_token} hopefully this {tokenizer.mask_token} works as intended to get {tokenizer.mask_token} information {tokenizer.pad_token} {tokenizer.pad_token}"

    assert tokenizer.tokenize(sentence, vocab, lookup) == expected


def test_empty_string():
    expected = BatchEncoding(data={'overflowing_tokens': [],
                        'num_truncated_tokens': -5,
                        'input_ids': [],
                        'token_type_ids': [],
                        'attention_mask': []})
    sentence = ""

    assert tokenizer(sentence, truncation=True, max_length=5, return_overflowing_tokens=True, vocab=vocab,
                         lookup=lookup) == expected

def test_special_tokens_from_vocab():
    # TODO: should not give an exception and the asserts should point to respective stable ids of these tokens

    assert tokenizer.sep_token_id == -1
    assert tokenizer.mask_token_id == -1
    assert tokenizer.pad_token_id == -1
    assert tokenizer.cls_token_id == -1


def test_relation_between_tokens_and_ids():
    sentence = "this is a random sentence to test the connection with the last word cosh"
    tokenized_string = tokenizer.tokenize(sentence, vocab, lookup)
    ids = [12515, 3059, 3026, 10282, 10862, 3056, 11677, 3052, 5847, 4040, 12530, 3052, 8464, 7257, 30000]
    assert tokenizer.convert_ids_to_tokens(ids) == tokenized_string
    assert tokenizer.convert_tokens_to_ids(tokenized_string) == ids

    assert tokenizer.decode(ids) == sentence
    assert tokenizer.encode(sentence) == ids
