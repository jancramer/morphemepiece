# MorphoLex

### Link to the repository containing the dataset
https://github.com/hugomailhot/MorphoLex-en

### Link to the article
https://link.springer.com/article/10.3758%2Fs13428-017-0981-8


### Evaluation Limitations and Concerns
The transformation from word to its morphemes is irreversible, since we can not just combine them back together to the original input word.
Instead of simply splitting the word up into different substrings we also need to transform some parts into their base.
Those new tokens should provide us with a better representation of the word meaning compared to the word splits.
Nevertheless, when we want to compare different tokenizers based on this dataset we have to keep in mind that most of them are designed to split up the string and not to transform individual parts of it.
