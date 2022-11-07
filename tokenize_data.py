import lzma
import os
import datetime
from tokenizer import MorphemepieceTokenizer
from multiprocessing import Pool
from transformers import BertTokenizer
def tokenize_text(data):
    print(data)
    tokenizer=MorphemepieceTokenizer()
    cur_path=os.path.dirname(__file__)
    data_path=os.path.relpath('..\\openwebtext\\OpenWebText\\subsets\\openwebtext\\urlsf_subset00-'+str(data)+'_data.xz', cur_path)
    load_data=datetime.datetime.now().time()
    text=lzma.open(data_path, mode='rt', encoding='utf-8').read()
    tokenized_text=tokenizer.tokenize(text)
    with open("C:\\Users\\janch\\OneDrive\\Desktop\\Uni\\BachelorThesis\\tokenized_files\\urlsf_subset00-"+str(data)+"_data_tokenized.txt", 'w', encoding='utf-8') as fp:
        fp.write(" ".join(token for token in tokenized_text))
    return data

num_process=8
start=datetime.datetime.now()
#tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer=MorphemepieceTokenizer()
print("Start: "+str(start))
if __name__ == '__main__':
    with Pool(num_process) as p:
        p.map(tokenize_text, range(1,1001))
#for data in range(7, 9):
    #tokenize_text(data)
fininished=datetime.datetime.now()
print("Finished: "+str(fininished))
print("Total time: "+ str(fininished-start))