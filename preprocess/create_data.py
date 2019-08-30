import pickle

import tensorflow as tf
import time
from gensim.models.word2vec import Word2Vec
import gensim
import os
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
import pickle




#encoding:utf-8
from collections import  Counter
import tensorflow.contrib.keras as kr
import numpy as np
import codecs
import re
import jieba


# coding: utf-8

import sys
from collections import Counter

import numpy as np
import keras as kr
import jieba
import os
import codecs

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file(content_dir,label_dir,mode):
    """读取文件数据"""
    contents, labels= [], []
    rank=0
    max_len=0
    readlines_label=open(label_dir,'r').readlines()
    readlines_content=open(content_dir,'r').readlines()
    print (len(readlines_label))
    if len(readlines_label)!=len(readlines_content):
        print (len(readlines_content))
        print (len(readlines_label))
        print ('Error! content and Label have diffirent dim')
        return None,None
    
    for i in range(len(readlines_label)):
        '''
        content_id=int(readlines_label[i].strip().split('\t')[0])
        label=readlines_label[i].strip().split('\t')[1]
        content=readlines_content[content_id].strip()
        '''
        content=readlines_content[i].strip()
        if content=='':
            content='<PAD>'
        label=readlines_label[i].strip()
        words=content.split()
        if mode=='word':
            #words=shuffle(words)
            contents.append(words)
        else:
            new_line=[]
            for word in words:
                new_line.extend(list(word))
            contents.append(new_line)
        labels.append(label)
    return contents,labels



def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir,vocab_size=10000):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    frequencys={}
    rank=0
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = []
        for line in fp.readlines():
            #print (line)
            word=line.strip().split()[0]
            frequency=int(line.strip().split()[1])
            words.append(word)
            frequencys[word]=frequency
            rank+=1
            if rank>=vocab_size:
                break
    words=['<PAD>']+words
    #word_to_id = dict(zip(words, range(len(words))))
    word_to_id={}
    for i in range(len(words)):
        word_to_id[words[i]]=i
    return words, word_to_id,frequencys



def read_category(file_name):
    ftarget=codecs.open(file_name,'r',encoding='utf-8')
    input=ftarget.readlines()
    length=len(input)
    categories=[]
    for line in input:
        result=line.strip()
        categories.append(result)
    categories = [native_content(x) for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(content_dir,label_dir, word_to_id, cat_to_id, max_length=25):
    """将文件转换为id表示"""
    contents,labels = read_file(content_dir,label_dir,'word')
    '''
    在分词阶段只去除了特殊的符号，在这里去除停用词
    '''
    chars=list('abcdefghijklmnopqrstuvwxyz我的于超前越你及等免二非五元中个把之')
    '''
    chars+=['新款','mm','送','新','可','不','与','和','无','有','之','自','好','适用','新品','货到付款',
    '件','新款','专用','跟','适用','夏季','男女','四季','包围','通用','送','新','mm','现代',
    '可','号','的','舒适','可爱','码','版','正版','大众','双人','装','夹','懒人','新品','中','内','百搭','春秋','正品','色','纯色','天然','级','一','月','式','原装','岁','豆豆','包邮','把',
    '个','一对','不','公主','头层','多','衣服','和','无','细','小号','卡罗',
    '人','全新','子','秋季','颜色','备注','自动','老','后','用','甜美','标配','虎','条','挡','片','本','逸',
    '空气','一字','皮肤','达','三','美','熊','服','含','鱼','请','清新','棕','插','直径','货到付款','随机',
    '黄','帮','下','玫','件','味','精品','留言','阳光','有','配','特价','干','日常','升级版',
    '日','实用','思域','上','之','睿','拼','夜光','兔','速递','纯','静音','马','自','世家','克',
    '年份','大红','cmcm','开业','派','只','对','超','同款','官方','前','越','品','及','你',
    '边','系系','尺寸','道','包女','配送','普通','支','无极','可选','点','等','免','朋友','致胜','二','仿','韩',
    '车衣车','森林','朵','选','精准','不带','建议','五','梦','拿','防撞','坐',
    '雅','午睡','只装','横款','揽胜','显瘦','真','做','多款','打','加']
    #chars+=['系列','的','套装','大','版','包邮','小','经典','正版','可','款','中国','与','黑色','新','红色','现货','白色','和','蓝色','带','中','不','专用','时尚','正品','送','全','年','进口','型','新款','装','cm','号','包','原装','黑','绿色','一','多功能','红','个','书籍','黄色','米','通用','人','用','组合','白','蓝','高','式','美国','mm','粉色','出版社','无','含','必备','彩色','色','新品','金','灰色','男女','加厚','多','家用','创意','宝宝','绿','三','适用','本','紫色','月','套餐','双','世界','等','迷你','休闲','全新','ml','随机','黄','上','套','日本','设计','好','现代','防水','颜色','粉','卡','户外','学生','专业','橙色','之','防','全套','配','升级版','简约','发货','女','选','美','韩国','可爱','有','粉红色','书','运动','爱','精装','女士','男','卡通','dvd','手工','盒','生活','北京','盒装','配件','金色','礼品','你','单','礼物','袋','及','适合','我','男士','元','品牌','德国','定制','精品','特价','官方','汽车','商务','基础','银色','小号','风','cd','安全','光盘','用品','卷','寸','超','智能','岁','便携','金属','实用','下','情侣','办公','片','婴儿','浅','学习','非','礼盒','升级','浅蓝色','棕色','两用','环保','夏季','复古','级','旅行','长','二','猫','全集','在','个性','花','一个','图书','透气','女款','京东','紫','头','全国','普通','不锈钢','双层','立体','器','件','电脑','健康','灰','苹果','韩版','工具','上海','著','包装','开','学','买','赠品','到','克','天','舒适','内','咖啡色','其他','台湾','册','图','玩具','成人','可选','高端','玫瑰','标配','拍','手','码','教材','夹','水晶','透明','加','精选','训练','豪华','手册','咖啡','达','水','丛书','玫','多色','收纳','高档','全册','四','高级',
    #]
    '''
    print (contents[:10])
    data_id, label_id = [], []
    all_num=0
    cover_num=0
    longer_num=0
    wrong_num=0
    for i in range(len(contents)):
        tmp=[]
        try:         
            all_num+=len(contents[i])
            for word in contents[i]:
                if word in word_to_id.keys()  and word not in chars: 
                    tmp.append(word_to_id[word])
                    cover_num+=1                   
            label=cat_to_id[labels[i]]
        except Exception as e:
            wrong_num+=1
            continue
        if len(tmp)>=max_length:
            longer_num+=1
        data_id.append(tmp)
        label_id.append(label)
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    print (longer_num)
    print (type(data_id))
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    print ('覆盖率: '+str(cover_num*1.0/all_num))
    print ('错误数量: '+str(wrong_num))
    assert wrong_num==0
    return x_pad, label_id
'''
判断word是否是英文单词
'''
def is_eng(word):
    flag=True
    for uchar in word:
        if uchar < u'\u0061' or uchar > u'\u007a':
            flag=False
    return flag


def restore(x_train,y_train,id_to_word,id_to_label):
    fcontent=open('data/train-content-char.txt','w')
    flabel=open('data/train-label-char.txt','w')
    for i in range(len(x_train)):
        content=[id_to_word[id] for id in x_train[i]]
        label=id_to_label[y_train[i]]
        chars=[]
        for word in content:
            if is_eng(word)==True:
                chars.append(word)
            else:
                chars.extend(list(word))
        fcontent.write(' '.join(chars)+'\n')
        flabel.write(label+'\n')

    fcontent.close()
    flabel.close()



vocab_dir_word='data/vocab-correct.txt'
vocab_dir_char='data/vocab-char.txt'
word_max_len=25
char_max_len=45
model='word'
max_token=150000#词汇表的大小




if __name__ == '__main__':
    word_to_id,label_to_id=None,None
    with open('data/label2idx_dict.pkl','rb') as f:
        label_to_id=pickle.load(f)
    id_to_label={}
    for label in label_to_id.keys():
        id_to_label[label_to_id[label]]=label
    vocab_dir=vocab_dir_word
    maxlen=word_max_len
    if model=='char':
        vocab_dir=vocab_dir_char
        maxlen=char_max_len

    words,word_to_id,frequencys=read_vocab(vocab_dir,max_token)
    print (len(words))
    #print (words[:100])
    '''
    dir1:存放分词后的商品标题
    dir2:存放标签
    每个标题和标签都是一行
    '''
    dir1='data/train-content-org.txt'
    dir2='data/train-label-org.txt'
    x_all_train, y_all_train = process_file(dir1,dir2, word_to_id,label_to_id, maxlen,words[:100])
    x_all_train,y_all_train=shuffle(x_all_train,y_all_train)

    print (len(x_all_train))
    print ('读取数据完毕')
    assert len(x_all_train)==len(y_all_train)
    with open('data/train-org.pkl','wb') as f:
        pickle.dump((x_all_train,y_all_train),f)
