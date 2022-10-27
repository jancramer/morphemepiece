import pandas as pd
import numpy as np
import os
from eval_metrics import *
from flota.src.flota_eval_tokenizer import FlotaEvalTokenizer

def create_predictions(model, k, strict, words):
    flota_tok = FlotaEvalTokenizer(model, k, strict, 'flota')
    first_tok = FlotaEvalTokenizer(model, k, strict, 'first')
    longest_tok = FlotaEvalTokenizer(model, k, strict, 'longest')
    flota_predictions = flota_tok.tokenize_no_special_batch(words)
    first_predictions = first_tok.tokenize_no_special_batch(words)
    longest_predictions = longest_tok.tokenize_no_special_batch(words)
    return flota_predictions, first_predictions, longest_predictions

def get_morpheme_tokens(morphemes):
    # parses the string into the morpheme tokens
    # example morphemes string "<un<{<ob<(trude)}>ive>>ness>"
    
    # remove "{ }"because we do not need the grouping 
    morphemes = morphemes.replace("{","").replace("}","")
    morpheme_tokens = []
    morpheme_types = []
    idx = 0
    while idx < len(morphemes):
        char = morphemes[idx]
        # the special chars introduce a new morpheme
        if char == "<" or char == "(" or char == ">":
            morpheme = ""
            morpheme_types.append(char)
            idx += 1
            char = morphemes[idx]
            # as long as we do not have the closing symbol we continue to add up the chars of our morpheme
            while char != "<" and char != ")" and char != ">":
                morpheme+=char
                idx += 1
                char = morphemes[idx]
            idx += 1
            morpheme_tokens.append(morpheme)
        else:
            print("error with the following morpheme")
            print(morphemes)
            idx+=1
    return morpheme_tokens, morpheme_types

path = os.path.dirname(os.path.abspath(__file__))
morpholex = pd.read_csv(os.path.join(path, "data", "1-1-2.csv"))
words = morpholex["Word"]
morpheme_strings = morpholex["MorphoLexSegm"]
morphemes, labels = zip(*[get_morpheme_tokens(morpheme_string) for morpheme_string in morpheme_strings])

model = 'bert-base-uncased'
strict = False

# calculated the morphological coverage over different k
coverages = []
for k in range(1, 5):
    predictions = create_predictions(model, k, strict, words)
    scores = [morphological_coverage(prediction, morphemes) for prediction in predictions]
    coverages.append(scores)
coverages = np.array(coverages)
print("morphological coverage for k 1-4 \n flota, first, longest")
print(coverages)

# calculate the stem recall scores
predictions = create_predictions(model, 1, strict, words)
scores = [stem_recall(prediction, morphemes, labels) for prediction in predictions]
print(f"stem recall: \n flota: {scores[0]} \n first: {scores[1]} \n longest: {scores[2]}")


# calculate the full match scores
predictions = create_predictions(model, 4, strict, words)
scores = [full_match(prediction, morphemes) for prediction in predictions]
print(f"full match: \n flota: {scores[0]} \n first: {scores[1]} \n longest: {scores[2]}")