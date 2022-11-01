from sys import prefix
from tokenizer import MorphemepieceTokenizer
from transformers import BertTokenizer
from vocab import Vocab
import pandas as pd

# vorbereiten des Vokabulars und des Lookup-Tables
vocabulary = pd.read_csv("./data/vocabulary.csv")["x"].to_list()
lookup = pd.read_csv("./data/lookup.csv").set_index("y").to_dict()["x"]
prefixes = pd.read_csv("./data/prefixes.csv")["x"].to_list()
words = pd.read_csv("./data/words.csv")["x"].to_list()
suffixes = pd.read_csv("./data/suffixes.csv")["x"].to_list()
vocab_split = {'prefixes': prefixes, 'words': words, 'suffixes': suffixes}
vocab = Vocab(vocabulary, vocab_split, True)

# Erstellen des Tokenizers
tokenizer = MorphemepieceTokenizer()
#tokenizer_bert=BertTokenizer.from_pretrained("bert-base-cased")
test_string = "let's test some compounds and punctuation here in this fine-grained testcase!?"
empty=""
tokenized_string = tokenizer.tokenize(test_string, vocab=tokenizer.vocab, lookup=tokenizer.lookup)
#print(tokenizer(empty, truncation=True, max_length=5, return_overflowing_tokens=True, vocab=vocab,
 #                        lookup=lookup))
print(tokenizer.encode(test_string,tokenizer.vocab,lookup=tokenizer.lookup))
print(tokenized_string)
#ids = []
#for i in range(len(tokenized_string)):
#    ids.append(tokenizer._convert_token_to_id(tokenized_string[i]))
#print(tokenizer.encode(test_string))
#print(tokenized_string)
#print(tokenizer.convert_tokens_to_string(tokenized_string))
#print(tokenizer.__call__(test_string, truncation=True, max_length=5, return_overflowing_tokens=True, vocab=vocab, lookup=lookup))
#print(tokenizer.unk_token_id()[0])
#print(tokenizer_bert.get_added_vocab())