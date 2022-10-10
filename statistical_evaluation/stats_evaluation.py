import numpy as np
import pandas as pd
import argparse
import collections
import itertools


def extract_word_sublists(tokens, word_boundary_mask):
    assert len(word_boundary_mask) == len(tokens)
    result = []
    last_mask = 0
    start = 0

    for i in range(1, len(word_boundary_mask)):
        current_mask = word_boundary_mask[i]
        if last_mask != current_mask:
            result.append(tokens[start:i])
            start = i
        last_mask = current_mask

    if start != len(word_boundary_mask) - 1:
        result.append(tokens[start:])

    return result


def word_boundary_mask(tokens):
    word_counter = 0
    result = []
    for i in range(len(tokens) - 1):
        current_t = tokens[i]
        next_t = tokens[i + 1]
        result.append(word_counter)

        if current_t == '##' or current_t.endswith("##"):
            continue

        if not next_t.startswith("##"):
            word_counter += 1

    result.append(word_counter)

    assert len(tokens) == len(result)

    return result


def report_token_stats(morphemepiece, bert):
    max_count = 40

    word_breakdowns_morphempiece = report_tokenizer_stats(max_count, morphemepiece, "morphemepiece")
    word_breakdowns_bert = report_tokenizer_stats(max_count, bert, "bert")

    overlapping = word_breakdowns_morphempiece.intersection(word_breakdowns_bert)

    print("Of the %d morphemepiece word tokenizations used, %d also occur in BERT. That is a ratio of %.3f" % (len(word_breakdowns_morphempiece), len(overlapping), len(overlapping)/ len(word_breakdowns_morphempiece)))
    print("Of the %d bert word tokenizations used, %d also occur in morphemepiece. That is a ratio of %.3f" % (
    len(word_breakdowns_bert), len(overlapping), len(overlapping) / len(word_breakdowns_bert)))


def report_tokenizer_stats(max_count, df, tokenizer_name):
    counter = count_tokens(df)
    most_common = counter.most_common(max_count)
    unique_tokens = len(set(counter))

    total_tokens = sum([s for _, s in counter.items()])

    avg_sent_token_lengths = df.map(
        lambda token_list: np.mean([compute_token_length(t) for t in token_list]))
    mean_token_length = avg_sent_token_lengths.mean()
    sd_token_length = avg_sent_token_lengths.std()

    word_boundaries = df.map(lambda tokens: extract_word_sublists(tokens, word_boundary_mask(tokens)))
    word_breakdown_strings = word_boundaries.map(
        lambda token_breakdown_list: [" ".join(x) for x in token_breakdown_list])
    complex_word_breakdown_strings = word_breakdown_strings.map(lambda strings: [x for x in strings if "##" in x])

    prefix_counter = collections.Counter(
        itertools.chain.from_iterable(df.map(lambda tokens: [x for x in tokens if x.endswith("##") and x != '##'])))
    suffix_counter = collections.Counter(
        itertools.chain.from_iterable(df.map(lambda tokens: [x for x in tokens if x.startswith("##") and x != '##'])))
    word_breakdown_string_counter = collections.Counter(itertools.chain.from_iterable(word_breakdown_strings))
    complex_word_breakdown_string_counter = collections.Counter(
        itertools.chain.from_iterable(complex_word_breakdown_strings))

    prefix_count = df.map(lambda tokens: len([x for x in tokens if x.endswith("##") and x != '##'])).sum()
    suffix_count = df.map(lambda tokens: len([x for x in tokens if x.startswith("##") and x != '##'])).sum()

    print("%s total tokens: %d" % (tokenizer_name, total_tokens))
    print("%s unique tokens: %d" % (tokenizer_name, unique_tokens))
    print("%s mean token length: %.3f (%.3f)" % (tokenizer_name, mean_token_length, sd_token_length))
    print("%s prefix token count: %d" % (tokenizer_name, prefix_count))
    print("%s suffix token count: %d" % (tokenizer_name, suffix_count))
    print("%s most_common_tokens: %s" % (tokenizer_name, most_common))
    print("%s most_common_prefixes: %s" % (tokenizer_name, prefix_counter.most_common(max_count)))
    print("%s most_common_suffixes: %s" % (tokenizer_name, suffix_counter.most_common(max_count)))

    print("%s most_common_word_strings: %s" % (tokenizer_name, word_breakdown_string_counter.most_common(max_count)))
    print("%s most_common_complex_strings: %s" % (
        tokenizer_name, complex_word_breakdown_string_counter.most_common(max_count)))

    return set(itertools.chain.from_iterable(word_breakdown_strings))

def compute_token_length(t: str):
    if t.startswith("#") or t.endswith("#"):
        return len(t) - 1
    return len(t)


def count_tokens(df):
    counter = collections.Counter()
    for l in df:
        counter.update(l)
    return counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    args = parser.parse_args()
    df = pd.read_csv(args.input_file)

    morphemepiece = df['morphemepiece'].map(lambda x: x.split())
    bert = df['bert'].map(lambda x: x.split())

    report_token_stats(morphemepiece, bert)


if __name__ == '__main__':
    main()
