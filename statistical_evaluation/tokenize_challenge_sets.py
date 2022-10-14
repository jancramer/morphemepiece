import pandas as pd
from glob import glob
from vocab import Vocab
from tokenizer import MorphemepieceTokenizer
from transformers import AutoTokenizer
from datetime import datetime as dt


def main():
    files = glob("flota_challenge_sets/*_train.csv")
    df = pd.concat([pd.read_csv(f) for f in files])

    vocabulary = pd.read_csv("../data/vocabulary.csv")["x"].to_list()
    lookup = pd.read_csv("../data/lookup.csv").set_index("y").to_dict()["x"]
    prefixes = pd.read_csv("../data/prefixes.csv")["x"].to_list()
    words = pd.read_csv("../data/words.csv")["x"].to_list()
    suffixes = pd.read_csv("../data/suffixes.csv")["x"].to_list()
    vocab_split = {'prefixes': prefixes, 'words': words, 'suffixes': suffixes}
    vocab = Vocab(vocabulary, vocab_split, True)

    # Erstellen des Tokenizers
    morpheme = MorphemepieceTokenizer(vocab, lookup)

    bert = AutoTokenizer.from_pretrained("bert-base-uncased")

    print("morphemepiece tokenization")
    df['morphemepiece'] = df['text'].map(
        lambda x: " ".join(morpheme.tokenize(x, vocab=vocab, lookup=lookup, unk_token="[UNK]", max_chars=200)))
    print("bert-uncased tokenization")
    df['bert'] = df['text'].map(lambda x: " ".join(bert.tokenize(x)))

    filename = dt.now().strftime("%Y%m%d_%H%M_tokenized.csv")
    df.to_csv("tokenized_datasets/"+filename)


if __name__ == '__main__':
    main()
