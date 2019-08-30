import fastText
from fastText import train_supervised
from fastText import load_model
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)%s : %(message)s',level=logging.INFO)

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

valid_data='/content/drive/drive/爬虫/data/train-test.txt'
classifier=train_supervised('/content/drive/drive/爬虫/data/train-train.txt',loss='hs',minCount=5,thread=8,dim=300,wordNgrams=2,epoch=5)
classifier.save_model('/content/drive/drive/爬虫/ckpt-facebook/fasttext.model')
classifier=load_model('/content/drive/drive/爬虫/ckpt-facebook/fasttext.model')
print_results(*classifier.test(valid_data))