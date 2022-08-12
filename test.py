from lib2to3.pgen2.tokenize import tokenize
from tokenizer import MorphemepieceTokenizer

vocab_split={   'prefixes': ["re", "un","sub"],
                'words': ["see","break"], 
                'suffixes': ["able", "ing" , "loose"]}
tokenizer=MorphemepieceTokenizer()
print(tokenizer.tokenize_word("unseebreakable",vocab_split=vocab_split, dir=-1))
