import pandas as pd

from statistical_evaluation.stats_evaluation import word_boundary_mask, extract_word_sublists
from vocab import Vocab
from tokenizer import MorphemepieceTokenizer




vocabulary = pd.read_csv("../data/vocabulary.csv")["x"].to_list()
lookup = pd.read_csv("../data/lookup.csv").set_index("y").to_dict()["x"]
prefixes = pd.read_csv("../data/prefixes.csv")["x"].to_list()
words = pd.read_csv("../data/words.csv")["x"].to_list()
suffixes = pd.read_csv("../data/suffixes.csv")["x"].to_list()
vocab_split = {'prefixes': prefixes, 'words': words, 'suffixes': suffixes}
vocab = Vocab(vocabulary, vocab_split, True)

# Erstellen des Tokenizers
morpheme = MorphemepieceTokenizer(vocab,lookup)

word = "the notebook charger is there."
tokenized = morpheme.tokenize(word, vocab=vocab, lookup=lookup, unk_token="[UNK]", max_chars=200)
mask = word_boundary_mask(tokenized)
words = extract_word_sublists(tokenized, mask)
print(tokenized)
print(mask)
print(words)
assert words == [['the'], ['note', '##', 'book'], ['charge', '##er'], ['is'], ['there']]

assert mask == [0, 1, 1, 1, 2, 2, 3, 4, 5]

word = "we charge the notebooks"
tokenized = morpheme.tokenize(word, vocab=vocab, lookup=lookup, unk_token="[UNK]", max_chars=200)
mask = word_boundary_mask(tokenized)
words = extract_word_sublists(tokenized, mask)
print(tokenized)
print(mask)
print(words)
assert words == [['we'], ['charge'], ['the'], ['note', '##', 'book', '##s']]
assert mask == [0, 1, 2, 3, 3, 3, 3]

word = "we charge the prenotebooks"
tokenized = morpheme.tokenize(word, vocab=vocab, lookup=lookup, unk_token="[UNK]", max_chars=200)
mask = word_boundary_mask(tokenized)
words = extract_word_sublists(tokenized, mask)
print(tokenized)
print(mask)
print(words)
assert mask == [0, 1, 2, 3, 3, 3, 3, 3]
assert words == [['we'], ['charge'], ['the'], ['pre##', 'note', '##', 'book', '##s']]

word = "we charge the pre notebooks"
tokenized = morpheme.tokenize(word, vocab=vocab, lookup=lookup, unk_token="[UNK]", max_chars=200)
mask = word_boundary_mask(tokenized)
words = extract_word_sublists(tokenized, mask)
print(tokenized)
print(mask)
print(words)
assert mask == [0, 1, 2, 3, 4, 4, 4, 4]
assert words == [['we'], ['charge'], ['the'], ['pre'], ['note', '##', 'book', '##s']]
