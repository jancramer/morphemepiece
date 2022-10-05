from sys import prefix
from tokenizer import MorphemepieceTokenizer
from vocab import Vocab
import pandas as pd


vocabulary=pd.read_csv("C:/Users/janch/OneDrive/Desktop/Uni/BachelorThesis/thesis-morphemepiece-port/vocabulary.csv")["x"]
lookup=pd.read_csv("C:/Users/janch/OneDrive/Desktop/Uni/BachelorThesis/thesis-morphemepiece-port/lookup.csv").set_index("y").to_dict()
prefixes=pd.read_csv("C:/Users/janch/OneDrive/Desktop/Uni/BachelorThesis/thesis-morphemepiece-port/prefixes.csv")["x"].to_list()
words=pd.read_csv("C:/Users/janch/OneDrive/Desktop/Uni/BachelorThesis/thesis-morphemepiece-port/words.csv")["x"].to_list()
suffixes= pd.read_csv("C:/Users/janch/OneDrive/Desktop/Uni/BachelorThesis/thesis-morphemepiece-port/suffixes.csv")["x"].to_list()
vocab_split={  'prefixes': prefixes, 'words': words, 'suffixes': suffixes}
vocab=Vocab(vocab_split, False)

tokenizer=MorphemepieceTokenizer(vocab)
print(tokenizer.tokenize_word_bidirectional("unbelieveable",vocab_split=vocab.vocab_split, unk_token="[UNK]", max_chars=200 ))
