# Morphological Evaluation

## Ladec Dataset

Contains words made up of exactly two compounds.
The relevant columns for us are `stim`, containing the word itself, `c1`, containing the first compound, and `c2`, containing the second compound.

## Metrics

### Morphological Coverage
The percentage of gold standard tokens covered by the tokenization

### Stem Recall
Only used with a single token tokenization.
1 if the single token is the stem of the word 0 otherwise.

### Full Match
1 if the tokenization is equal to the gold tokenization otherwise 0


## Evaluation Limitations and Concerns

### Gold Standard
Datasets defines a gold standard for the tokenization. This does however not necessarily mean that there is only one right solution and that the given tokenization makes the most sense.
In their paper they split undesirable into `un` and `desirable` here one could argue that the word is made up of three parts `un`, `desire` and `able`.

### Morphological Coverage
Having a high coverage with fewer tokens is seen as good.
Nevertheless, we have to keep in mind that sometimes more tokens also entail more information. As the ladec dataset is made up of words consisting of two compounds the ideal number of tokens to represent this information should also be two, this can however change if another dataset is used.

### Stem Recall
As we treat both compounds of words from the ladec dataset as word stem we loose a bit of the original meaning. The metric should be more expressive with a dataset where words also contain prefixes and suffixes or other non stem parts.

### Dataset and Used Words
Especially when only using the ladec dataset we do not have a rich variety of diverse words. They instead are rather similar. This also goes in the way of flota as the words we are looking at are specifically those where flota can provide an advantage. Whereas in a normal context not every word will be made up of two compounds or other morphemes. When only looking at the scores on the dataset we could assume a stronger improvement compared to the real usecase.