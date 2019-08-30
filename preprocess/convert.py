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
import jieba.posseg as pseg#用于词性标注

'''
加载并使用自定义词表
'''
#jieba.load_userdict("vocab-correct.txt")
#jieba.set_dictionary('vocab-correct.txt')
def fenci(datas):
    #cut_words=nlp.tag(datas)
    #return cut_words[0]['word']
    cut_words=jieba.cut(datas,cut_all=False)
    return cut_words

#os.chdir('data')
words_in_cates={}
fcontent=open('data/train-content-org.txt','w')
flabel=open('data/train-label-org.txt','w')
fin=open('data/train-ubuntu.tsv')
rank=0
num=0
fix_num=0
right_num=0
readlines=fin.readlines()[1:]
engliesh_num=0
chars=list('abcdefghijklmnopqrstuvwxyz我的于')
for i in range(len(readlines)):
    line=readlines[i]
    if i%10000==0:
        print (i)
    try:
        label=line.strip().split('\t')[1]
        content=line.strip().split('\t')[0]
    except Exception as e:
        continue
    flabel.write(label+'\n')
    result=[]
    for part in re.split(r'[:-]',content):
        for word in part.split():
                result.extend(fenci(format_str(process(word))))
    if len(result)==0:
        print (i)
    fcontent.write(' '.join(result)+'\n')
fcontent.close()
flabel.close()