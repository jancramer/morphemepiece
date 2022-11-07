# thesis-morphemepiece-port

Thesis Jan-Christoph

# Usage
- Basic implementation of the morphemepiece-tokenizer in Python from the R package morphemepiece https://github.com/macmillancontentscience/morphemepiece
    - use `from tokenizer import MorphemepieceTokenizer` to import the tokenizer into your project
    - to tokenizer create a `MorphemepieceTokenizer` object
        - call the tokenize function oft these objext with your text, tokenizer.vocab and tokenizer.lookup for the default data 
    - default data:
        - provided by the R package https://github.com/macmillancontentscience/morphemepiece.data 
        - this data for the default tokenization is stored in the `data` folder
        - these data is extracted from the R package 
    - if you want to use a custom vocabulary use the the class `vocab.py`
        - otherwise the standard vocabulary is used
    - in `test.py` are some test cases implemented, that test the functionality of this project
    - in `test_degug.py` are some examples of the usage implemented
    
    


# Evaluation Ideas

- The morphological measures from "Embarassingly simple paper" (Gianluca is implementing them)

- Variety of stats for a given corpus
    - how long are the tokens on average (implemented)
    - how many unique tokens do we need to tokenize a corpus (implemented)
    - how many total tokens do we need to tokenize a corpus (implemented)
    - how many suffixe / prefixe (implemented)
    - how many words without affix (implemented)
    - how many tokenizations in the vocabular overlap (implemented)

    - how long where the suffixe / prefixes
    - how many tokens are actually meaningful (words, common prefixes / affixes)


- Classification over the challenge sets of the "Embarassingly simple paper" (possibly Jan Cramer)
- SmallNLP Benchmark (possibly Jan Cramer)

### Exporting Python path
in order to solve importing of parent modules with respective path to the repo

`export PYTHONPATH="/home/gianluca/Documents/hiwi/thesis-morphemepiece-port"`