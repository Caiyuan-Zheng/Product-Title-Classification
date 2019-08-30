from tensorflow.nn.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib import rnn
import time
import gensim
import os
import numpy as np
import pickle
import tensorflow as tf 


base_dir='/content/drive/drive/master/A01-master/data-with-eng'
content_train_dir=os.path.join(base_dir,'train-content.txt')
label_train_dir=os.path.join(base_dir,'train-label.txt')
content_test_dir = os.path.join(base_dir, 'test-content-all.txt')
label_test_dir=os.path.join(base_dir,'test-label-fix.txt')
vocab_dir = 'data/vocab-correct.txt'
#vocab_dir='/media/zcy/DATA/大三上/服务外包比赛/比赛数据/train/new-vocab60000.txt'
cate_dir=os.path.join(base_dir,'categories-1258.txt')

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

def read_vocab(vocab_dir,vocab_size=10000):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    frequencys=dict()
    rank=0
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = []
        for line in fp.readlines():
            rank+=1
            #print (line)
            word=line.strip().split()[0]
            frequency=int(line.strip().split()[1])
            words.append(word)
            frequencys[word]=frequency
            if rank>=vocab_size:
                break
    words=['<PAD>']+words
    word_to_id = dict(zip(words, range(len(words))))
    frequencys['<PAD>']=10
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


def make_train_feed_dict(model, batch):
    """make train feed dict for training"""
    feed_dict = {model.x: batch[0],
                 model.label: batch[1],
                 model.keep_prob: .75,
                 model.embedding_keep_prob:0.9}
    return feed_dict


def make_test_feed_dict(model, batch):
    feed_dict = {model.x: batch[0],
                model.label: batch[1],
                 model.keep_prob: 1.0,
                 model.embedding_keep_prob:1.0}
    return feed_dict


def run_train_step(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    to_return = {
    'optimizer': model.train_op,
    'accuracy':model.accuracy,
    'loss': model.loss,
    'global_step':model.global_step,
    }
    return sess.run(to_return, feed_dict)


def run_eval_step(model, sess, batch):
    feed_dict = make_test_feed_dict(model, batch)
    acc,loss = sess.run([model.accuracy,model.loss], feed_dict)
    return acc,loss


def fill_feed_dict(data_X, data_Y, batch_size):
    """Generator to yield batches"""
    # Shuffle data first.
    shuffled_X, shuffled_Y = shuffle(data_X, data_Y)
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = shuffled_X[batch_size * idx: batch_size * (idx + 1)]
        y_batch = shuffled_Y[batch_size * idx: batch_size * (idx + 1)]
        yield x_batch, y_batch

def run_eval(classifier, sess, x_dev,y_dev):
    print ('Start Validation')
    total_right=0.0
    total_loss=0.0
    total_len=0.0
    for x_batch, y_batch in fill_feed_dict(x_dev, y_dev,256):
        curr_acc,curr_loss = run_eval_step(classifier, sess, (x_batch, y_batch))
        total_right+=len(x_batch)*curr_acc
        total_loss+=len(x_batch)*curr_loss
        total_len+=len(x_batch)

    print (total_right)
    print (len(x_dev))
    print (total_len)
    return total_right/total_len,total_loss/total_len

def scale_l2(x, norm_length):
    # shape(x) = (batch, num_timesteps, d)
    # Divide x by max(abs(x)) for a numerically stable L2 norm.
    # 2norm(x) = a * 2norm(x/a)
    # Scale over the full sequence, dims (1, 2)
    alpha = tf.reduce_max(tf.abs(x), (1, 2), keepdims=True) + 1e-12
    l2_norm = alpha * tf.sqrt(
        tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keepdims=True) + 1e-6)
    x_unit = x / l2_norm
    return norm_length * x_unit


def normalize(emb, weights):
    # weights = vocab_freqs / tf.reduce_sum(vocab_freqs) ?? 这个实现没问题吗
    print("Weights: ", weights)
    mean = tf.reduce_sum(weights * emb, 0, keep_dims=True)
    var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keep_dims=True)
    stddev = tf.sqrt(1e-6 + var)
    return (emb - mean) / stddev

def create_embedding(words):
    embedding_dir='data/embedding-w5-s300-word.bin'
    embedding_model=Word2Vec.load(embedding_dir)
    #embedding_model=Word2Vec.load('/content/drive/drive/master/A01-master/data/embedding-w3-s300.bin')
    #embedding_model=gensim.models.KeyedVectors.load_word2vec_format('data/vectors.txt') #GloVe Model
    #embedding_model=Glove.load('data/cooccurrence.bin')
    embedding_word2vec=[]
    
    num=0
    for i in range(0,len(words)):
        if words[i]  in embedding_model:
            embedding_word2vec.append(embedding_model[words[i]].tolist())
            num+=1
        else:
            embedding_word2vec.append(list(np.random.uniform(-1,1,size=300)))
    '''
    embedding_dir=os.path.join(base_dir,'vectors-jieba-300d-w5.txt')
    embedding_model=gensim.models.KeyedVectors.load_word2vec_format(embedding_dir) #GloVe Model
    embedding_glove=[]
    for i in range(len(words)):
        if words[i]  in embedding_model:
            embedding_glove.append(embedding_model[words[i]].tolist())
            num+=1
        else:
            embedding_glove.append(list(np.random.uniform(-1,1,size=300)))
    print ('word embedding num: '+str(num))
    embedding=np.concatenate([embedding_word2vec,embedding_glove],axis=1)
    '''
    return embedding_word2vec



class AdversarialCharClassifier(object):
    def __init__(self, config):
        self.max_len=config.word_max_len
        self.hidden_size=config.word_hidden_size
        self.vocab_size=config.word_vocab_size
        self.embedding_size=config.word_embedding_size
        self.n_class=config.num_classes
        self.learning_rate=config.learning_rate
        self.epsilon=config.epsilon
        self.init_embedding=config.word_embedding

        # placeholder
        self.x = tf.placeholder(tf.int32, [None, self.max_len],name='input_x')
        self.label = tf.placeholder(tf.int32, [None],name='input_y')
        self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        self.embedding_keep_prob=tf.placeholder(tf.float32,name='embedding_keep_prob')

    def _add_perturbation(self, embedded, loss):
        """Adds gradient to embedding and recomputes classification loss."""
        grad, = tf.gradients(
            loss,
            embedded,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        grad = tf.stop_gradient(grad)
        perturb = scale_l2(grad, self.epsilon)
        return embedded + perturb

    def _get_freq(self, vocab_freq, word2idx):
        """get a frequency dict format as {word_idx: word_freq}"""
        words = vocab_freq.keys()
        freq = [0] * self.vocab_size
        for word in words:
            word_idx = word2idx.get(word)
            word_freq = vocab_freq[word]
            freq[word_idx] = word_freq
        return freq

    def build_graph(self, vocab_freq, word2idx):
        vocab_freqs = tf.constant(self._get_freq(vocab_freq, word2idx),
                                  dtype=tf.float32, shape=(self.vocab_size, 1))
        weights = vocab_freqs / tf.reduce_sum(vocab_freqs)
        embeddings_var =tf.get_variable(name='embedding',initializer=self.init_embedding,trainable=True)
        #embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
        #                   trainable=True)
        embedding_norm = normalize(embeddings_var, weights)
        batch_embedded = tf.nn.embedding_lookup(embedding_norm, self.x)
        #batch_embedded = tf.nn.dropout(batch_embedded,self.embedding_keep_prob)

        W = tf.Variable(tf.random_normal([self.hidden_size], stddev=0.1))
        W_fc = tf.Variable(tf.truncated_normal([self.hidden_size, self.n_class], stddev=0.1))
        b_fc = tf.Variable(tf.constant(0., shape=[self.n_class]))

        def cal_loss_logit(embedded, keep_prob, reuse=True, scope="loss"):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
                rnn_outputs, _ = bi_rnn(LSTMCell(self.hidden_size),
                                        LSTMCell(self.hidden_size),
                                        inputs=embedded, dtype=tf.float32)

                # Attention
                H = tf.add(rnn_outputs[0], rnn_outputs[1])  # fw + bw
                M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)
                # alpha (bs * sl, 1)
                alpha = tf.nn.softmax(tf.matmul(tf.reshape(M, [-1, self.hidden_size]),
                                                tf.reshape(W, [-1, 1])))
                r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, self.max_len,
                                                                             1]))  # supposed to be (batch_size * HIDDEN_SIZE, 1)
                r = tf.squeeze(r)
                h_star = tf.tanh(r)
                drop = tf.nn.dropout(h_star, keep_prob)

                # Fully connected layer（dense layer)
                y_hat = tf.nn.xw_plus_b(drop, W_fc, b_fc,name='logits')

            return y_hat, tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=self.label))

        logits, self.cls_loss = cal_loss_logit(batch_embedded, self.keep_prob, reuse=False)
        embedding_perturbated = self._add_perturbation(batch_embedded, self.cls_loss)
        adv_logits, self.adv_loss = cal_loss_logit(embedding_perturbated, self.keep_prob, reuse=True)
        self.loss = self.cls_loss+self.adv_loss



        # optimization
        loss_to_minimize = self.loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
        self.prediction = tf.argmax(tf.nn.softmax(logits), 1)
        self.prediction=tf.cast(self.prediction,tf.int32)
        self.correct_pred = tf.equal(self.label, self.prediction)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        print("graph built successfully!")


class AdversarialModel():
    def __init__(self,config,save_path,frequencys,word2idx):
        self.classifier = AdversarialCharClassifier(config)
        self.classifier.build_graph(frequencys, word2idx)
        self.session=tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    #输入一个矩阵
    def predict(self,input_x):
        feed_dict={
            self.classifier.x:input_x,
            self.classifier.keep_prob:1.0,
            self.classifier.embedding_keep_prob:1.0
        }
        y_pred_cls = self.session.run(self.model.prediction, feed_dict=feed_dict)
        return y_pred_cls


class Args():
    def __init__(self):
        print ('配置参数')



if __name__ == '__main__':
    flags=Args()
    flags.word_max_len=25
    flags.char_max_len=45
    flags.data_file='dataset-50w-word.pkl'
    flags.word_hidden_size=256
    flags.word_embedding_size=300
    flags.last_rank=0.0
    flags.best_acc=0.0
    flags.learning_rate=1e-3
    flags.train_epoch=40
    flags.batch_size=512
    flags.epsilon=5
    flags.save_path='ckpt-50w/adversial-word'
    flags.best_acc=0.000
    flags.num_classes=1258

    tf.reset_default_graph()
    maxlen,max_token,label_to_id,word_to_id=None,None,None,None
    #with open('data/myinfo.pkl','rb') as f:
    #    maxlen,max_token,label_to_id,word_to_id=pickle.load(f)



    word_to_id,label_to_id=None,None
    with open('data/label2idx_dict.pkl','rb') as f:
        label_to_id=pickle.load(f)

    id_to_label={}
    for key in label_to_id:
        id_to_label[label_to_id[key]]=key


    words,word_to_id,frequencys=read_vocab(vocab_dir,150000)
    if os.path.exists('data/embedding.pkl'):
        with open('data/embedding.pkl','rb') as f:
            flags.word_embedding=pickle.load(f)
    else:
        flags.word_embedding=create_embedding(words)
        with open('data/embedding.pkl','wb') as f:
            pickle.dump(flags.word_embedding,f)

    #flags.word_embedding=flags.word_embedding.astype(np.float32)
    #flags.word_embedding=create_embedding(words)

    x_test,y_test=None,None
    x_train,y_train=None,None
    x_tmp,y_tmp=None,None
    flags.word_max_len=25
    flags.word_vocab_size=len(words)
    import pickle
    from sklearn.utils import shuffle
    with open('data/train-fix-50w.pkl','rb') as f:
        x_tmp,y_tmp=pickle.load(f)
        x_test,y_test=x_tmp[:512],y_tmp[:512]
        x_train,y_train=x_tmp[512:],y_tmp[512:]
    x_train,y_train,x_test,y_test=np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)
    x_train,y_train=shuffle(x_train,y_train)
    #print (y_tmp[:10])
    print (len(words))
    classifier = AdversarialCharClassifier(flags)
    classifier.build_graph(frequencys,word_to_id)
    saver=tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    start = time.time()
    for e in range(flags.train_epoch):
        t0 = time.time()
        print("Epoch %d start !" % (e + 1))
        iters=0
        #x_train,y_train=shuffle(x_train,y_train)
        #avg_loss=0
        for x_batch, y_batch in fill_feed_dict(x_train, y_train, flags.batch_size):
            #if e==0 and (iters+1)<=10000:
            #    iters+=1
            #    continue
            return_dict = run_train_step(classifier, sess, (x_batch, y_batch))
            #avg_loss+=return_dict['loss']
            if iters%200==0:
                print (return_dict)
            iters+=1


        t1 = time.time()
        dev_acc,dev_loss = run_eval(classifier, sess,x_test,y_test)
        #saver.save(sess=sess,save_path=flags.save_path)
        print ('Have saved model')
        print("validation accuracy: %.3f loss: %.3f " % (dev_acc,dev_loss))
        if dev_acc > flags.best_acc:
            flags.best_acc=dev_acc
            if flags.best_acc>=0.900:
                saver.save(sess=sess,save_path=flags.save_path+str(flags.best_acc))


        print("Train Epoch time:  %.3f s" % (t1 - t0))