from sys import prefix
from tokenizer import MorphemepieceTokenizer
from vocab import Vocab
import pandas as pd

#vorbereiten des Vokabulars und des Lookup-Tables
vocabulary=pd.read_csv("./data/vocabulary.csv")["x"].to_list()
lookup=pd.read_csv("./data/lookup.csv").set_index("y").to_dict()["x"]
prefixes=pd.read_csv("./data/prefixes.csv")["x"].to_list()
words=pd.read_csv("./data/words.csv")["x"].to_list()
suffixes= pd.read_csv("./data/suffixes.csv")["x"].to_list()
vocab_split={  'prefixes': prefixes, 'words': words, 'suffixes': suffixes}
vocab=Vocab(vocabulary,vocab_split, True)

#Erstellen des Tokenizers
tokenizer=MorphemepieceTokenizer(vocab,lookup)
test_string="we charge the prenotebooks"
tokenized_string=tokenizer.tokenize(test_string,vocab=vocab,lookup=lookup, unk_token="[UNK]", max_chars=200 )
ids=[]
for i in range(len(tokenized_string)):
    ids.append(tokenizer._convert_token_to_id(tokenized_string[i]))
print(tokenizer.decode(ids))
print(tokenizer.encode(test_string))
print(tokenized_string)
print(tokenizer.convert_tokens_to_string(tokenized_string))
