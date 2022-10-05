import re
from vocab import Vocab


class MorphemepieceTokenizer(object):

    def __init__(self,vocab):
        self.vocab=vocab
        

    def tokenize_word(self, word:str, vocab_split, dir=1, allow_compounds=True, unk_token="[UNK]", max_chars=100):
        if len(word)> max_chars:
            return unk_token
        frag_pat="##"
        #load vocabulary
        prefixes=vocab_split['prefixes']
        words=vocab_split['words']
        suffixes=vocab_split['suffixes']
        is_bad=False
        start=1
        sub_tokens=[]
        word_len=len(word)
        end=word_len
        #verbesserungsfähig
        compound = False

        word_allowed = "XXX"

        if allow_compounds:
            word_allowed="#"

        allowed_next_rules={}
        allowed_next=[]
        #set of rules for forward search
        if dir==1:
            allowed_next_rules={
                'p': ["p","w","s"],
                'w': ["s", word_allowed],
                's': "s"
            }
            allowed_next=["p","w"]
        else:
            allowed_next_rules={
                'p': "p",
                'w': ["p", word_allowed],
                's': ["p","w","s"]
            }
            allowed_next=["s","w"]
        keep_going=True

        while keep_going:
            
            if dir==1:
                end = word_len
            else: 
                start=1
            
            cur_substring= ""
            while start <= end:
                sub_str = word[start-1:end] 
                #look for prefixes, if allowed
                if "p" in allowed_next and end < word_len and sub_str in prefixes:
                    cur_substring = sub_str+frag_pat
                    allowed_next=allowed_next_rules['p']
                    break

                #look for suffixes, if allowed
                elif "s" in allowed_next and start > 1 and sub_str in suffixes:
                    cur_substring = frag_pat+sub_str
                    allowed_next=allowed_next_rules['s']
                    break
                
                #look for complete words, if allowed
                elif ("w" in allowed_next or "#" in allowed_next) and sub_str in words:
                    cur_substring= sub_str
                    if "#" in allowed_next:
                        if dir == 1:
                            sub_tokens.append(frag_pat)
                        else:
                            sub_tokens.insert(0,frag_pat)
                    allowed_next=allowed_next_rules['w']                   
                    break
                if dir== 1:
                    end=end -1
                else: 
                    start= start+1
            if cur_substring == "":
                is_bad=True
                break
            
            if dir==1:
                sub_tokens.append(cur_substring)
                start=end+1
                keep_going= start <= word_len
            else:
                sub_tokens.insert(0, cur_substring)
                end = start-1
                keep_going= end >=1
            
        if is_bad:
            return unk_token

        return sub_tokens

    def tokenize_word_bidirectional(self, word:str, vocab_split, unk_token, max_chars, allow_compounds=True):

        forwards_list=self.tokenize_word(word, vocab_split=vocab_split,dir=1, allow_compounds=allow_compounds,unk_token=unk_token,max_chars=max_chars)
        backwards_list= self.tokenize_word(word, vocab_split=vocab_split,dir=-1, allow_compounds=allow_compounds,unk_token=unk_token,max_chars=max_chars)

        len_forward=len([token for token in forwards_list if token!="##"])
        len_backward=len([token for token in backwards_list if token!="##"])
        if len_backward < len_forward and len_backward >1:
            return backwards_list
        else:
            return forwards_list

    

    def tokenize_word_lookup(self, word:str, vocab:Vocab, lookup:dict, unk_token, max_chars):
        vocab_split=vocab.vocab_split
        #check if it is in raw vocabulary
        vocabulary=vocab.vocabulary
        
        if word =="":
            return 0
            
        if word in vocabulary:
            return [word]  
        token_list:list
        if word in lookup.keys():
            breakdown:str=lookup[word]
            token_list= breakdown.split(" ")
        else:
            token_list=self.tokenize_word_bidirectional(word, vocab_split, unk_token, max_chars)
        
        return token_list


    def __space_tokenizer(self, words:str):
        return re.findall(r"[\w']+|[.,!?;]", words)

    def tokenize(self, text:str, vocab:Vocab ,lookup,unk_token="[UNK]", max_chars=100):
        is_cased=vocab.is_cased
         
        if is_cased:
            text=text.lower()
        
        word_list=self.__space_tokenizer(text)
        
        tokens=[self.tokenize_word_lookup(word, vocab, lookup, unk_token, max_chars) for word in word_list]
        #flatten the list
        return [token for tokens_word in tokens for token in tokens_word]