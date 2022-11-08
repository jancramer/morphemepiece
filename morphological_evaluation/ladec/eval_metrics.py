import numpy as np


def morphological_coverage(predictions, compounds1, compounds2):
    # returns the average coverage of the two compounds from the prediction tokens
    # for a single prediction the coverage can be either 0, 0.5 or 1  when none, one or both compounds are contained in the prediction respectively
    scores = []
    for tokens, c1, c2 in zip(predictions, compounds1, compounds2):
        match = 0
        combined_tokens = "".join(tokens)
        if c1 in combined_tokens:
            match+=1
        if c2 in combined_tokens:
            match+=1
        scores.append(match/2)
    return np.average(scores)

def stem_recall(predictions, compounds1, compounds2):
    # returns the percentage of exact stem matches
    # we treat both compounds as valid stems
    scores = []
    for tokens, c1, c2 in zip(predictions, compounds1, compounds2):
        if c1 == tokens[0] or c2 == tokens[0]:
            scores.append(1)
        else:
            scores.append(0)
    return np.average(scores)

def full_match(predictions, compounds1, compounds2):
    # returns the percentage of exact matches between the predicted tokenization and the gold standard
    scores = []
    for tokens, c1, c2 in zip(predictions, compounds1, compounds2):
        if len(tokens) != 2:
            scores.append(0)
        elif c1 == tokens[0] and c2 == tokens[1]:
            scores.append(1)
        else:
            scores.append(0)
    return np.average(scores)