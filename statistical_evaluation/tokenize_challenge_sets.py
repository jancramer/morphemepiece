import pandas as pd
from glob import glob
from vocab import Vocab
from tokenizer import MorphemepieceTokenizer
from transformers import AutoTokenizer
from datetime import datetime as dt
from flota.src.flota_eval_tokenizer import FlotaEvalTokenizer
import os

def main():
    # get the absolute path of this file so that it does not matter from where we start the file
    path = os.path.dirname(os.path.abspath(__file__))
    files = glob(os.path.join(path, "..", "flota","data", "*_train.csv"))
    df = pd.concat([pd.read_csv(f) for f in files])

    vocabulary = pd.read_csv(os.path.join(path, "..","data", "vocabulary.csv"))["x"].to_list()
    lookup = pd.read_csv(os.path.join(path, "..","data", "lookup.csv")).set_index("y").to_dict()["x"]
    prefixes = pd.read_csv(os.path.join(path, "..", "data", "prefixes.csv"))["x"].to_list()
    words = pd.read_csv(os.path.join(path, "..", "data", "words.csv"))["x"].to_list()
    suffixes = pd.read_csv(os.path.join(path, "..", "data", "suffixes.csv"))["x"].to_list()
    vocab_split = {'prefixes': prefixes, 'words': words, 'suffixes': suffixes}
    vocab = Vocab(vocabulary, vocab_split, True)

    # Erstellen des Tokenizers
    morpheme = MorphemepieceTokenizer(vocab, lookup)

    bert = AutoTokenizer.from_pretrained("bert-base-uncased")

    flota = FlotaEvalTokenizer('bert-base-uncased', 3, False, "flota")

    print("morphemepiece tokenization")
    df['morphemepiece'] = df['text'].map(
        lambda x: " ".join(morpheme.tokenize(x, vocab=vocab, lookup=lookup, unk_token="[UNK]", max_chars=200)))
    
    print("bert-uncased tokenization")
    df['bert'] = df['text'].map(lambda x: " ".join(bert.tokenize(x)))

    print('flota')
    df['flota'] = df['text'].map(lambda x: " ".join(flota.tokenize(x)))
    filename = dt.now().strftime("%Y%m%d_%H%M_tokenized.csv")
    df.to_csv(os.path.join(path, "tokenized_datasets", filename))


if __name__ == '__main__':
    main()
