import pandas as pd
from eval_metrics import *
from flota.src.flota_eval_tokenizer import FlotaEvalTokenizer
import os


def create_predictions(model, k, strict, words):
    flota_tok = FlotaEvalTokenizer(model, k, strict, 'flota')
    first_tok = FlotaEvalTokenizer(model, k, strict, 'first')
    longest_tok = FlotaEvalTokenizer(model, k, strict, 'longest')
    flota_predictions = flota_tok.tokenize_no_special_batch(words)
    first_predictions = first_tok.tokenize_no_special_batch(words)
    longest_predictions = longest_tok.tokenize_no_special_batch(words)
    return flota_predictions, first_predictions, longest_predictions


# read in the datasets
path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(path, "ladec_only_correct_parse.csv")
ladec = pd.read_csv(dataset_path)

model = 'bert-base-uncased'
strict = False

# filter out the entries which can not be tokenized correctly by the model's vocab
flota_tok = FlotaEvalTokenizer(model, 1, strict, 'flota')
# filter only dependant on the model. We can use this data for different k and modes as long as the model stays the same.
ladec = flota_tok.filter_ladec(ladec)


# calculated the morphological coverage over different k
coverages = []
for k in range(1, 5):
    predictions = create_predictions(model, k, strict, ladec['stim'])
    scores = [morphological_coverage(prediction, ladec['c1'], ladec['c2']) for prediction in predictions]
    coverages.append(scores)
coverages = np.array(coverages)
print("morphological coverage for k 1-4 \n flota, first, longest")
print(coverages)


# calculate the stem recall scores
predictions = create_predictions(model, 1, strict, ladec['stim'])
scores = [stem_recall(prediction, ladec['c1'], ladec['c2']) for prediction in predictions]
print(f"stem recall: \n flota: {scores[0]} \n first: {scores[1]} \n longest: {scores[2]}")


# calculate the full match scores
predictions = create_predictions(model, 2, strict, ladec['stim'])
scores = [full_match(prediction, ladec['c1'], ladec['c2']) for prediction in predictions]
print(f"full match: \n flota: {scores[0]} \n first: {scores[1]} \n longest: {scores[2]}")
