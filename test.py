from sys import prefix
from tokenizer import MorphemepieceTokenizer
from vocab import Vocab
import pandas as pd


vocabulary=pd.read_csv("C:/Users/janch/OneDrive/Desktop/Uni/BachelorThesis/thesis-morphemepiece-port/vocabulary.csv")["x"].to_list()
lookup=pd.read_csv("C:/Users/janch/OneDrive/Desktop/Uni/BachelorThesis/thesis-morphemepiece-port/lookup.csv").set_index("y").to_dict()["x"]
prefixes=pd.read_csv("C:/Users/janch/OneDrive/Desktop/Uni/BachelorThesis/thesis-morphemepiece-port/prefixes.csv")["x"].to_list()
words=pd.read_csv("C:/Users/janch/OneDrive/Desktop/Uni/BachelorThesis/thesis-morphemepiece-port/words.csv")["x"].to_list()
suffixes= pd.read_csv("C:/Users/janch/OneDrive/Desktop/Uni/BachelorThesis/thesis-morphemepiece-port/suffixes.csv")["x"].to_list()
vocab_split={  'prefixes': prefixes, 'words': words, 'suffixes': suffixes}
vocab=Vocab(vocabulary,vocab_split, True)

tokenizer=MorphemepieceTokenizer(vocab)
print(tokenizer.tokenize("When you create FIAT backed cryptocurrency using our tokenizer, the respective logic of the backing currency is embedded in the Smart contract automatically",vocab=vocab,lookup=lookup, unk_token="[UNK]", max_chars=200 ))
