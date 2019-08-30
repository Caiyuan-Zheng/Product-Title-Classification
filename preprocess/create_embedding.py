import logging
import os
from gensim.models import word2vec
#word2vec.word2vec('content.txt','embedding-w3-s300.bin',hs=1,min_count=1,window=3,size=300,iter=10')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.LineSentence('data/corpus.txt') 

model = word2vec.Word2Vec(sentences, sg=0, size=300,  window=3,  min_count=10,  negative=3, sample=0.001, hs=1, workers=8)
model.save('data/embedding-w3-s300-char.bin')