import numpy as np


def morphological_coverage(predictions, morpheme_tokenizations):
    # returns the average coverage of morpheme tokens by the predictions
    scores = []
    for tokens, morphemes in zip(predictions, morpheme_tokenizations):
        match = 0
        combined_tokens = "".join(tokens)
        for morpheme in morphemes:
            match = match+1 if morpheme in combined_tokens else match
        scores.append(match/len(morphemes))
    return np.average(scores)

def stem_recall(predictions, morpheme_tokenizations, label_lists):
    # returns the percentage of exact stem matches
    # a stem is labeled as "("
    scores = []
    for tokens, morphemes, labels in zip(predictions, morpheme_tokenizations, label_lists):
        is_stem = False
        for morpheme, label in zip(morphemes, labels):
            if label == "(" and len(tokens) == 1 and  morpheme == tokens[0]:
                is_stem = True
        scores.append(int(is_stem))
    return np.average(scores)

def full_match(predictions, morpheme_tokenizations):
    # returns the percentage of exact matches between the predicted tokenization and the gold standard
    scores = []
    for tokens, morphemes in zip(predictions, morpheme_tokenizations):
        is_same = True
        if len(tokens) != len(morphemes):
            is_same = False
        for token, morpheme in zip(tokens, morphemes):
            is_same = is_same and token == morpheme
        scores.append(int(is_same))
    return np.average(scores)