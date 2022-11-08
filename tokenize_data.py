import lzma
import os
import datetime
from tokenizer import MorphemepieceTokenizer
from multiprocessing import Pool
from transformers import BertTokenizer
def tokenize_text(subset):
    subset_str=str(subset)
    if subset < 10:
        subset_str="0"+str(subset)
    tokenizer=MorphemepieceTokenizer()
    cur_path=os.path.dirname(__file__)
    for data in range (1,1001):
        data_path=os.path.relpath('./openwebtext/urlsf_subset'+subset_str+'-'+str(data)+'_data.xz', cur_path)
        text=lzma.open(data_path, mode='rt', encoding='utf-8').read()
        tokenized_text=tokenizer.tokenize(text)
        with open("./tokenized_data/urlsf_subset"+subset_str+"-"+str(data)+"_data_tokenized.txt", 'w', encoding='utf-8') as fp:

            fp.write(" ".join(token for token in tokenized_text))
    return subset

num_process=8
start=datetime.datetime.now()
#tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")

tokenizer=MorphemepieceTokenizer()
print("Start: "+str(start))
dictionary='tokenized_data'
isExist = os.path.exists(dictionary) 
# Create a new directory because it does not exist
if not isExist: 	
  
   os.makedirs(dictionary)
   print("The new directory is created!")
if __name__ == '__main__':
    with Pool(num_process) as p:
        p.map(tokenize_text, range(21))

fininished=datetime.datetime.now()
print("Finished: "+str(fininished))
print("Total time: "+ str(fininished-start))