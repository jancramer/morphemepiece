
def get_morpheme_tokens(morphemes, ignore_grouping):
    # parses the string into the morpheme tokens
    # example morphemes string "<un<{<ob<(trude)}>ive>>ness>"
    
    if ignore_grouping:
        # remove "{ }"because we do not need the grouping 
        morphemes = morphemes.replace("{","").replace("}","")
    
    morpheme_tokens = []
    morpheme_types = []
    idx = 0
    while idx < len(morphemes):
        char = morphemes[idx]
        morpheme = ""
        
        if char == "{":
            morpheme_types.append("(")
            idx += 1
            char = morphemes[idx]
            while char != "}":
                if char != "<" and char != ">" and char != "(" and char != ")":
                    morpheme += char
                idx += 1
                char = morphemes[idx]
            morpheme_tokens.append(morpheme)
            idx += 1
        
        # the special chars introduce a new morpheme
        elif char == "<" or char == "(" or char == ">":
            morpheme_types.append(char)
            idx += 1
            char = morphemes[idx]
            # as long as we do not have the closing symbol we continue to add up the chars of our morpheme
            while char != "<" and char != ")" and char != ">":
                morpheme += char
                idx += 1
                char = morphemes[idx]
            idx += 1
            morpheme_tokens.append(morpheme)
        else:
            print("error with the following morpheme")
            print(morphemes)
            idx += 1
    return morpheme_tokens, morpheme_types