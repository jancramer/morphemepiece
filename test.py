from tokenizer import MorphemepieceTokenizer
from vocab import Vocab
import pandas as pd
from pandas import DataFrame


vocab_split={   'prefixes': ["re", "un","sub"],
                'words': ["see","break"], 
                'suffixes': ["able", "ing" , "loose"]}
lookup={


}
vocabulary=pd.read_csv("C:/Users/janch/OneDrive/Desktop/Uni/BachelorThesis/thesis-morphemepiece-port/vocabulary.csv")
voc_df=pd.DataFrame(vocabulary)
print(voc_df)
vocab=Vocab(vocab_split, False)
tokenizer=MorphemepieceTokenizer(vocab)
print(tokenizer.tokenize_word_bidirectional("unseebreakable",vocab_split=vocab.vocab_split, unk_token="[UNK]", max_chars=200 ))
