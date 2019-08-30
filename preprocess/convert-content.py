from __future__ import print_function, unicode_literals

import pandas as pd
from collections import Counter
import re



def process(our_data):
    our_data=our_data.lower()
    return list(our_data)

def is_right(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    elif uchar >= u'\u0061' and uchar <= u'\u007a':
        return True
    else:
        return False

def is_eng(word):
    flag=True
    for uchar in word:
        if uchar < u'\u0061' or uchar > u'\u007a':
            flag=False
    return flag

def format_str(content):
    content_str = ''
    for i in content:
        if is_right(i):
            content_str = content_str +i
    return content_str

import jieba
import jieba.posseg as pseg
#jieba.load_userdict("vocab-correct.txt")
'''
fin=open('plus-vocabs.txt')
for line in fin.readlines():
    word=line.strip().split()[0]
    jieba.add_word(word,100)

fin.close()
'''
#jieba.set_dictionary('vocab-correct.txt')
#dict_path='dict.txt'
#jieba.load_userdict(dict_path)
def fenci(datas):
    #cut_words=nlp.tag(datas)
    #return cut_words[0]['word']
    cut_words=jieba.cut(datas,cut_all=False)
    return cut_words

#os.chdir('data')
fcontent=open('test-content.txt','w')
fin=open('test-ubuntu.tsv')
readlines=fin.readlines()[1:]
for i in range(len(readlines)):
    line=readlines[i]
    if i%10000==0:
        print (i)
    content=line.strip()
    result=[]
    for part in re.split(r'[:-]',content):
        for word in part.split():
                result.extend(fenci(format_str(process(word))))
    if len(result)==0:
        print (i)
    fcontent.write(' '.join(result)+'\n')
fcontent.close()