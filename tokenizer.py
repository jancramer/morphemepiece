import re
from typing import Any, Dict, List, Union, Optional
from vocab import Vocab
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers.utils import PaddingStrategy, TensorType
from transformers.tokenization_utils_base import TruncationStrategy


class MorphemepieceTokenizer(PreTrainedTokenizer):
    vocab_files_names: Dict[str, str]
    pretrained_vocab_files_map: Dict[str, Dict[str, str]]
    max_model_input_sizes: Dict[str, Optional[int]]
    pretrained_init_configuration: Dict[str, Dict[str, Any]]
    model_input_names: List[str]
    padding_side: str
    truncation_side: str

    def __init__(self,
                 vocab: Vocab,
                 lookup: Dict[str, List[str]],
                 do_lower_case=True,
                 do_basic_tokenize=True,
                 never_split=None,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 tokenize_chinese_chars=True,
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

        self.vocab = vocab
        self.lookup = lookup

    """
    def __call__(self,
                text: Union[str, List[str], List[List[str]]],
                text_pair: Optional[Union[str, List[str], List[List[str]]]] = None, 
                add_special_tokens: bool = True, 
                padding: Union[bool, str, PaddingStrategy] = False, 
                truncation: Union[bool, str, TruncationStrategy] = False, 
                max_length: Optional[int] = None, stride: int = 0, 
                is_split_into_words: bool = False, 
                pad_to_multiple_of: Optional[int] = None, 
                return_tensors: Optional[Union[str, TensorType]] = None, 
                return_token_type_ids: Optional[bool] = None,
                return_attention_mask: Optional[bool] = None,
                return_overflowing_tokens: bool = False, 
                return_special_tokens_mask: bool = False, 
                return_offsets_mapping: bool = False, 
                return_length: bool = False, 
                verbose: bool = True, **kwargs) -> BatchEncoding:
        input_ids=self.encode(text)
        token_type_ids=[]
        attention_mask=[]
        overflowing_tokens=[]
        data={'input_ids': input_ids,'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        return BatchEncoding(data=data)
    """

    def tokenize_word(self, word: str, vocab_split, dir=1, allow_compounds=True, unk_token="[UNK]", max_chars=100):
        if len(word) > max_chars:
            return unk_token
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
            return unk_token

        return sub_tokens

    def tokenize_word_bidirectional(self, word: str, vocab_split, unk_token, max_chars, allow_compounds=True):

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

    def tokenize_word_lookup(self, word: str, vocab: Vocab, lookup: dict, unk_token, max_chars):
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
            token_list = self.tokenize_word_bidirectional(word, vocab_split, unk_token, max_chars)

        return token_list

    def __space_tokenizer(self, words: str):
        return re.findall(r"[\w']+|[.,!?;-]", words)

    def tokenize(self, text: str, vocab: Vocab, lookup, unk_token="[UNK]", max_chars=100):
        is_cased = vocab.is_cased

        if is_cased:
            text = text.lower()

        word_list = self.__space_tokenizer(text)

        tokens = [self.tokenize_word_lookup(word, vocab, lookup, unk_token, max_chars) for word in word_list]
        # flatten the list
        return [token for tokens_word in tokens for token in tokens_word]

    # preparation for huggingface

    def get_added_vocab(self):
        vocab_dic = {}
        vocab_tmp = self.vocab.vocabulary
        for i in range(len(vocab_tmp)):
            vocab_dic[vocab_tmp[i]] = i + 1
        return vocab_dic

    def _convert_token_to_id(self, token: str):
        return self.vocab.vocabulary.index(token) + 1

    def convert_tokens_to_ids(self, tokens: list):
        return [self._convert_token_to_id(token) for token in tokens]

    def _convert_id_to_token(self, id):
        return self.vocab.vocabulary[id - 1]

    def convert_ids_to_tokens(self, ids: list):
        return [self._convert_id_to_token(id) for id in ids]

    def convert_tokens_to_string(self, tokens: list):
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

    def decode(self, ids: list):
        return self.convert_tokens_to_string(self.convert_ids_to_tokens(ids))

    def batch_decode(self, sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor", "tf.Tensor"],
                     skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = True, **kwargs) -> List[
        str]:
        return [self.decode(sentence) for sentence in sequences]

    def encode(self, text: str):
        tokens = self.tokenize(text, self.vocab, self.lookup)
        return self.convert_tokens_to_ids(tokens)
