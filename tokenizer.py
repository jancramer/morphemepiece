import re
from typing import Any, Dict, List, Union, Optional
from vocab import Vocab
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers.utils import PaddingStrategy, TensorType
from transformers.tokenization_utils_base import TruncationStrategy
from transformers import BasicTokenizer
import pandas as pd



class MorphemepieceTokenizer(PreTrainedTokenizer):
    r"""
        Construct a subword tokenizer, that obtains morphemes.

        This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
        this superclass for more information regarding those methods.

        This tokenizer is the ported version from the R package morphemepiece https://github.com/macmillancontentscience/morphemepiece

        Args:
            vocab (`Vocab`, *optional*, defaults to `morphemepiece_vocabulary` from R):
                Vocab object with the vocabulary.
            lookup (`Dict[str, List[str]]`, *optional*, defaults to 'morphemepiece_lookup' from R):
                Lookup for the tokenization process.
            do_lower_case (`bool`, *optional*, defaults to `True`):
                Whether or not to lowercase the input when tokenizing.
            do_basic_tokenize (`bool`, *optional*, defaults to `True`):
                Whether or not to do basic tokenization before WordPiece.
            never_split (`Iterable`, *optional*):
                Collection of tokens which will never be split during tokenization. Only has an effect when
                `do_basic_tokenize=True`
            unk_token (`str`, *optional*, defaults to `"[UNK]"`):
                The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
                token instead.
            sep_token (`str`, *optional*, defaults to `"[SEP]"`):
                The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
                sequence classification or for a text and a question for question answering. It is also used as the last
                token of a sequence built with special tokens.
            pad_token (`str`, *optional*, defaults to `"[PAD]"`):
                The token used for padding, for example when batching sequences of different lengths.
            cls_token (`str`, *optional*, defaults to `"[CLS]"`):
                The classifier token which is used when doing sequence classification (classification of the whole sequence
                instead of per-token classification). It is the first token of the sequence when built with special tokens.
            mask_token (`str`, *optional*, defaults to `"[MASK]"`):
                The token used for masking values. This is the token used when training this model with masked language
                modeling. This is the token which the model will try to predict.
            tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
                Whether or not to tokenize Chinese characters.
            

    """
    vocab_files_names: Dict[str, str] = {"morphemepiece_vocab": "./data/vocabulary.csv",
                                        "suffixes":"./data/suffixes.csv",
                                        "prefixes":"./data/prefixes.csv",
                                        "words": "./data/words.csv",
                                        "lookup": "./data/lookup.csv"}
    pretrained_vocab_files_map: Dict[str, Dict[str, str]]
    max_model_input_sizes: Dict[str, Optional[int]]
    pretrained_init_configuration: Dict[str, Dict[str, Any]]
    model_input_names: List[str]
    padding_side: str
    truncation_side: str
    
    def _prepare_vocab(self)->Vocab:
        """load and prepare vocabulary from morphemepiece_vocab"""
        vocabulary = pd.read_csv(self.vocab_files_names["morphemepiece_vocab"])["x"].to_list()
        prefixes = pd.read_csv(self.vocab_files_names["prefixes"])["x"].to_list()
        words = pd.read_csv(self.vocab_files_names["words"])["x"].to_list()
        suffixes = pd.read_csv(self.vocab_files_names["suffixes"])["x"].to_list()
        vocab_split = {'prefixes': prefixes, 'words': words, 'suffixes': suffixes}
        vocab = Vocab(vocabulary, vocab_split, True)
        return vocab

    def _prepare_lookup(self)-> Dict[str,str]:
        """load and prepare lookup from morphemepiece_vocab"""
        return pd.read_csv(self.vocab_files_names["lookup"]).set_index("y").to_dict()["x"]

    def __init__(self,
                 vocab: Vocab = None,
                 lookup: Dict[str, List[str]]=None,
                 do_lower_case=True,
                 do_basic_tokenize=True,
                 never_split=None,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 tokenize_chinese_chars=False,
                 strip_accents=None,
                 **kwargs):

        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs)
        if vocab is None: 
            vocab=self._prepare_vocab()
        if lookup is None:
            lookup=self._prepare_lookup()
        self.vocab = vocab
        self.lookup = lookup
       


    def tokenize_word(self, word: str, vocab_split, dir=1, allow_compounds=True, unk_token="[UNK]", max_chars=100) -> List[str]:
        """Tokenize a single word based on morphemepiece tokenization,
            
            Applies a set of rules to determine the order of the morphemes

            Args:
                word(`str`):
                    Word that should be tokenized.
                vocab_split(`Dict[str, List[str]]`):
                    Dictionary that consists of lists with prefixes, suffixes and words.
                dir(`int`, *optional*, defaults to `1`):
                    Reading direction: `1` = forwards,  `-1`= backwards.
                allow_compounds(`bool`, *optional*, defaults to `True`):
                    Whether or not allow compounds in the tokenization process.
                unk_token(`str`, *optional*, defaults to `"[UNK]"`):
                    The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
                    token instead.  
                max_chars(`int`, *optional*, defaults to `100`):
                    The maximum length of a word, that can be tokenized. Returns `"[UNK]"` if the word is too long.
            Returns: 
                `List[str]`: 
                    List of tokens of the word

        """
        if len(word) > max_chars:
            return [unk_token]
        frag_pat = "##"
        # load vocabulary
        prefixes = vocab_split['prefixes']
        words = vocab_split['words']
        suffixes = vocab_split['suffixes']
        is_bad = False
        start = 1
        sub_tokens = []
        word_len = len(word)
        end = word_len
        compound = False

        word_allowed = "XXX"

        if allow_compounds:
            word_allowed = "#"

        allowed_next_rules = {}
        allowed_next = []
        # set of rules for forward search
        if dir == 1:
            allowed_next_rules = {
                'p': ["p", "w", "s"],
                'w': ["s", word_allowed],
                's': "s"
            }
            allowed_next = ["p", "w"]
        else:
            allowed_next_rules = {
                'p': "p",
                'w': ["p", word_allowed],
                's': ["p", "w", "s"]
            }
            allowed_next = ["s", "w"]
        keep_going = True

        while keep_going:

            if dir == 1:
                end = word_len
            else:
                start = 1

            cur_substring = ""
            while start <= end:
                sub_str = word[start - 1:end]
                # look for prefixes, if allowed
                if "p" in allowed_next and end < word_len and sub_str in prefixes:
                    cur_substring = sub_str + frag_pat
                    allowed_next = allowed_next_rules['p']
                    break

                # look for suffixes, if allowed
                elif "s" in allowed_next and start > 1 and sub_str in suffixes:
                    cur_substring = frag_pat + sub_str
                    allowed_next = allowed_next_rules['s']
                    break

                # look for complete words, if allowed
                elif ("w" in allowed_next or "#" in allowed_next) and sub_str in words:
                    cur_substring = sub_str
                    if "#" in allowed_next:
                        if dir == 1:
                            sub_tokens.append(frag_pat)
                        else:
                            sub_tokens.insert(0, frag_pat)
                    allowed_next = allowed_next_rules['w']
                    break
                if dir == 1:
                    end = end - 1
                else:
                    start = start + 1
            if cur_substring == "":
                is_bad = True
                break

            if dir == 1:
                sub_tokens.append(cur_substring)
                start = end + 1
                keep_going = start <= word_len
            else:
                sub_tokens.insert(0, cur_substring)
                end = start - 1
                keep_going = end >= 1

        if is_bad:
            return [unk_token]

        return sub_tokens

    def tokenize_word_bidirectional(self, word: str, vocab_split, unk_token, max_chars, allow_compounds=True) -> List[str]:
        """Tokenize a single word bidirectional.
            Select the direction with the fewest tokens.

            Args:
                word(`str`):
                    Word that should be tokenized.
                vocab_split(`Dict[str, List[str]]`):
                    Dictionary that consists of lists with prefixes, suffixes and words.
                unk_token(`str`):
                    The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
                    token instead.  
                max_chars(`int`):
                    The maximum length of a word, that can be tokenized. 
                allow_compounds(`bool`, *optional*, defaults to `True`):
                    Whether or not allow compounds in the tokenization process.
            Returns: 
                `List[str]`: 
                    List of tokens of the word

        """

        forwards_list = self.tokenize_word(word, vocab_split=vocab_split, dir=1, allow_compounds=allow_compounds,
                                           unk_token=unk_token, max_chars=max_chars)
        backwards_list = self.tokenize_word(word, vocab_split=vocab_split, dir=-1, allow_compounds=allow_compounds,
                                            unk_token=unk_token, max_chars=max_chars)
        len_forward = len([token for token in forwards_list if token != "##"])
        len_backward = len([token for token in backwards_list if token != "##"])
        if len_backward < len_forward and len_backward > 1:
            return backwards_list
        else:
            return forwards_list

    def tokenize_word_lookup(self, word: str, vocab: Vocab, lookup: dict, unk_token, max_chars, allow_compounds=True) -> List[str]:
        """Tokenize a single using the lookup, if possible. 
            Otherwise use bidirectional tokenization.

            Args:
                word(`str`):
                    Word that should be tokenized.
                vocab(`Vocab`):
                    Vocab object, that consists of the vocabulary and the splitted vocabulary
                lookup('Dict[str, List[str]]'):
                    A dictionary that catches all specified special cases.
                unk_token(`str`):
                    The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
                    token instead.  
                max_chars(`int`):
                    The maximum length of a word, that can be tokenized. 
                allow_compounds(`bool`, *optional*, defaults to `True`):
                    Whether or not allow compounds in the tokenization process.
            Returns: 
                `List[str]`: 
                    List of tokens of the word

        """
        vocab_split = vocab.vocab_split
        # check if it is in raw vocabulary
        vocabulary = vocab.vocabulary

        if word == "":
            return 0

        if word in vocabulary:
            return [word]
        token_list: list
        if word in lookup.keys():
            breakdown: str = lookup[word]
            token_list = breakdown.split(" ")
        else:
            token_list = self.tokenize_word_bidirectional(word, vocab_split, unk_token, max_chars, allow_compounds)
        return token_list

    def __space_tokenizer(self, words: str):
        return re.findall(r"[\w']+|[.,!?;-]", words)

    def tokenize(self, text: str, vocab: Vocab, lookup, unk_token="[UNK]", max_chars=100) -> List[str]:
        """Default tokenization function. 

            Uses a basic tokenizer to split the text at whitespaces and punctuation. 
            Then performs the original morphemepiece algorithm to tokenize these words.

            Args:
                text(`str`):
                    Text that should be tokenized.
                vocab(`Vocab`):
                    Vocab object, that consists of the vocabulary and the splitted vocabulary
                lookup('Dict[str, List[str]]'):
                    A dictionary that catches all specified special cases.
                unk_token(`str`):
                    The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
                    token instead.  
                max_chars(`int`):
                    The maximum length of a word, that can be tokenized. 
                allow_compounds(`bool`, *optional*, defaults to `True`):
                    Whether or not allow compounds in the tokenization process.
            Returns: 
                `List[str]`: 
                    List of tokens of the word

        """
        is_cased = vocab.is_cased
        #if is_cased:
        #    text = text.lower()

        #word_list = self.__space_tokenizer(text)
        basic_tokenizer= BasicTokenizer(never_split=[self.unk_token, self.sep_token, self.pad_token, self.cls_token,self.mask_token])
        word_list=basic_tokenizer.tokenize(text)
        tokens = [self.tokenize_word_lookup(word, vocab, lookup, unk_token, max_chars) for word in word_list]
        # flatten the list
        if tokens == [] or isinstance(tokens[0], str):
            return tokens
        return [token for tokens_word in tokens for token in tokens_word]

    # methods for huggingface

    def get_added_vocab(self):
        """ Returns the vocabulary of the `Vocab` object from this object
            Returns: 
                `Dict[str, int]`: 
                    Dictionary of vocabularies with their correspondig ids.
        """
        vocab_dic = {}
        vocab_tmp = self.vocab.vocabulary
        for i in range(len(vocab_tmp)):
            vocab_dic[vocab_tmp[i]] = i + 1
        return vocab_dic

    def _convert_token_to_id(self, token: str):
        """ Returns the ID to a token"""
        return self.vocab.vocabulary.index(token) + 1

    def convert_tokens_to_ids(self, tokens):
        """Returns all IDs to the provided tokens.
            Args:
                tokens(`List[str]`):
                    All tokens from from which the ID should be returned.
            Returns:
                `List[int]`:
                    List with all corresponding IDs from the tokens.
        """
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def _convert_id_to_token(self, id):
        """ Returns the token to an ID"""
        return self.vocab.vocabulary[id - 1]

    def convert_ids_to_tokens(self, ids: list):
        """Returns all tokens to the provided IDs.
            Args:
                ids(`List[int]`):
                    All IDs from from which the token should be returned.
            Returns:
                `List[str]`:
                    List with all corresponding tokens from the IDs.
        """
        return [self._convert_id_to_token(id) for id in ids]

    def convert_tokens_to_string(self, tokens: list)-> str:
        """Reconcatenates all tokens to a continuous text.

            Problems with irregular tokenization catched by the lookup.
            Args:
                tokens(`List[str]`):
                    Tokens extracted by the tokenization process of this class.
            Returns:
                `str`:
                    Concatenated string of all tokens.
        """
        concatenat_compounds = tokens
        for token in concatenat_compounds:
            if token == '##':
                token_id = concatenat_compounds.index(token)
                first = token_id - 1
                second = token_id + 1
                concatenat_compounds.remove(token)
                concatenat_compounds[first:second] = ["".join(concatenat_compounds[first:second])]
        out_string = " ".join(concatenat_compounds).replace(" ##", "").replace("## ", "").strip()
        return out_string

    def decode(self, ids: list) -> str:
        """ Constructs encoded list of IDs to a continous text.
            Args:
                ids(`List[int]`):
                    IDs extracted by the encode functionality of this class.
            Returns: 
                `str`:
                    Concatenated string of all IDs.
        """
        return self.convert_tokens_to_string(self.convert_ids_to_tokens(ids))

    def batch_decode(self, sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor", "tf.Tensor"],
                     skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = True, **kwargs) -> List[
        str]:
        return [self.decode(sentence) for sentence in sequences]

    def encode(self, text: str):
        """ 
            Encodes a given text. 

            Args: 
                text(`str`): 
                    Text, that should be encoded. 
            Returns: 
                `List[int]`:
                    Corresponding IDs to the tokens of the input text. 
        """
        tokens = self.tokenize(text, self.vocab, self.lookup)
        return self.convert_tokens_to_ids(tokens)
