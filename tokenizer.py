from operator import sub

from numpy import append


class MorphemepieceTokenizer(object):

    def __init__(self):
        pass

    def tokenize_word(self, word:str, vocab_split, dir=1, allow_compounds=True, unk_token="[UNK]", max_chars=1000):
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
        if dir==1:
            allowed_next_rules={
                'p': ["p","w","s"],
                'w': ["s", word_allowed],
                's': "p"
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
                print(sub_str)
                #look for prefixes, if allowed
                if "p" in allowed_next and end < word_len and sub_str in prefixes:
                    cur_substring = sub_str+frag_pat
                    allowed_next=allowed_next_rules['p']
                    print(cur_substring)
                    break

                #look for suffixes, if allowed
                if "s" in allowed_next and start > 1 and sub_str in suffixes:
                    cur_substring = frag_pat+sub_str
                    allowed_next=allowed_next_rules['s']
                    print(cur_substring)
                    break
                
                #look for complete words, if allowed
                if ("w" in allowed_next or "#" in allowed_next) and sub_str in words:
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
                
            if cur_substring[0] == "":
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
