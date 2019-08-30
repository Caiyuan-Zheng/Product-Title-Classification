import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector
from gensim.models.word2vec import Word2Vec
import gensim
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
import pickle

vocab_dir = 'data/vocab-correct.txt'


import numpy as np


def partition_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]

def run_eval_step(model, sess, batch):
    feed_dict = make_test_feed_dict(model, batch)
    acc,loss,prediction,logits = sess.run([model.accuracy,model.loss,model.prediction,model.logits], feed_dict)
    return acc,loss,prediction,logits
def wash(classifier, sess, x_dev,y_dev,id_to_label):
    print ('Start Validation')
    total_right=0.0
    total_loss=0.0
    total_len=0.0
    total_add=0.0
    x_plus=[]
    y_plus=[]
    cnt=0
    start=0
    end=0
    correct_num={}
    total_num={}
    add_num=0
    for i in range(1258):
        correct_num[i]=0
        total_num[i]=0
    cate_num={}
    cate_error={}
    books=[]
    for i in range(1258):
        if id_to_label[i].split('--')[0]=='图书杂志':
            books.append(i)
    for i in range(1258):
        cate_num[i]=0
        cate_error[i]=0
    while end<len(x_dev):
        start=end
        end=start+1024
        if end>len(x_dev):
            end=len(x_dev)
        x_batch,y_batch=x_dev[start:end],y_dev[start:end]
        curr_acc,curr_loss,prediction,logits = run_eval_step(classifier, sess, (x_batch, y_batch))
        reliability=np.max(logits,axis=1)
        for i in range(len(x_batch)):
            cate_num[y_batch[i]]+=1
            if prediction[i]!=y_batch[i] and y_batch[i] not in books:
                cate_error[y_batch[i]]+=1
            else:
                x_plus.append(x_batch[i])
                y_plus.append(y_batch[i])
            #y_plus.append(prediction[i])
        total_right+=len(x_batch)*curr_acc
        total_loss+=len(x_batch)*curr_loss
        total_len+=len(x_batch)
        cnt+=1
        if cnt%200==0:
            print ('rank '+str(cnt))
    print (total_right)
    print (len(x_dev))
    print (total_len)
    print (total_add)
    for i in range(1258):
        rate=0
        if cate_num[i]!=0:
            rate=cate_error[i]/cate_num[i]
        print (id_to_label[i]+'----'+str(cate_num[i])+'----'+str(rate))



    '''
    fout=open('data/wash.txt','w')
    for i in range(len(x_plus)):
        content=x_plus[i]
        label=y_plus[i]
        fout.write(str(label)+'\t'+' '.join('%s' %id for id in content))
        fout.write('\n')
    fout.close()
    '''
    with open('data/train-fix-50w.pkl','wb') as f:
        pickle.dump((x_plus,y_plus),f)
    return total_right/total_len,total_loss/total_len

def predict(classifier,sess,x_test,y_label,id_to_label,id_to_word,batch_size=1024):
    start=0
    end=0
    y_test=[]
    rank=0
    print (len(x_test))
    while end<len(x_test):
        start=end
        end=start+batch_size
        if end>len(x_test):
            end=len(x_test)
        x_batch=x_test[start:end]
        feed_dict={
            classifier.x:x_batch,
            classifier.keep_prob:1.0,
            classifier.embedding_keep_prob:1.0
        }
        logits = sess.run(classifier.logits, feed_dict=feed_dict)
        for logit in logits:
            y_test.append(np.argsort(-logit)[:3])
        rank+=1
        if rank%200==0:
            print (rank)
    print (len(x_test))
    print (len(y_test))
    #y_text=[id_to_label[id] for id in y_test]
    fout=open('data/train-val-predict.txt','w')
    right_num=0
    for i in range(len(y_test)):
        predict=list(y_test[i])
        predict=[id_to_label[int(idx)] for idx in predict]
        label=id_to_label[y_label[i]]
        for cate in predict:
            if i==0:
                print (cate)
                print (label)
            if cate==label:
                right_num+=1
                break
        if i==0:
            print (predict)
            print (type(predict))
        fout.write(label+'----'+predict[0]+'----'+predict[1]+'----'+predict[2]+'\n')
    fcontent=open('train-val-content.txt','w')
    for line in x_test:
        content=[id_to_word[idx] for idx in line if idx!=0]
        fcontent.write(' '.join(content)+'\n')
    #fout.write('\n'.join(y_text))
    print (right_num)
    fout.close()
    fcontent.close()

def read_vocab(vocab_dir,vocab_size=10000):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    frequencys={}
    rank=0
    with open(vocab_dir) as fp:
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
    word_to_id={}
    for i in range(len(words)):
        word_to_id[words[i]]=i
    #word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id,frequencys

def make_train_feed_dict(model, batch):
    """make train feed dict for training"""
    feed_dict = {model.x: batch[0],
                 model.label: batch[1],
                 model.keep_prob: .5,
                 model.embedding_keep_prob:0.8}
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
    'loss': model.loss,
    'accuracy':model.accuracy,
    'global_step_op':model.global_step,
    }
    return sess.run(to_return, feed_dict)
'''
def run_eval_step(model, sess, batch):
    feed_dict = make_test_feed_dict(model, batch)
    acc,loss = sess.run([model.accuracy,model.loss], feed_dict)
    return acc,loss
'''
def get_attn_weight(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    return sess.run(model.alpha, feed_dict)


def create_embedding(words):
    embedding_dir='data/embedding-w3-s300-char.bin'
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
    #embedding_glove=None
    #with open('data/embedding-glove-150d.pkl','rb') as f:
    #    embedding_glove=pickle.load(f)
    '''
    embedding_dir=os.path.join('data','vector-150d-w5-tmp.txt')
    embedding_model=gensim.models.KeyedVectors.load_word2vec_format(embedding_dir) #GloVe Model
    embedding_glove=[]
    num=0
    for i in range(len(words)):
        if words[i]  in embedding_model:
            embedding_glove.append(embedding_model[words[i]].tolist())
            num+=1
        else:
            embedding_glove.append(list(np.random.uniform(-1,1,size=150)))
    '''
    #embedding=np.concatenate([embedding_word2vec,embedding_glove],axis=1)
    print ('word embedding num: '+str(num))
    return embedding_word2vec
def fill_feed_dict(data_X, data_Y, batch_size):
    """Generator to yield batches"""
    # Shuffle data first.
    shuffled_X, shuffled_Y = shuffle(data_X, data_Y)
    # print("before shuffle: ", data_Y[:10])
    # print(data_X.shape[0])
    # perm = np.random.permutation(data_X.shape[0])
    # data_X = data_X[perm]
    # shuffled_Y = data_Y[perm]
    # print("after shuffle: ", shuffled_Y[:10])
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
        curr_acc,curr_loss,_,_ = run_eval_step(classifier, sess, (x_batch, y_batch))
        total_right+=len(x_batch)*curr_acc
        total_loss+=len(x_batch)*curr_loss
        total_len+=len(x_batch)

    print (total_right)
    print (len(x_dev))
    print (total_len)
    return total_right/total_len,total_loss/total_len


class TextFAST(object):
    """A FASTTEXT for text classification."""

    def __init__(self, config):

        # Placeholders for input, output, dropout_prob and training_tag
        self.init_embedding=config.word_embedding
        self.vocab_size=config.word_vocab_size
        self.embedding_size=config.word_embedding_size
        self.sequence_length=config.word_max_len
        self.num_classes=config.num_classes
        self.learning_rate=config.learning_rate
        self.x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.label = tf.placeholder(tf.int32, [None], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.embedding_keep_prob=tf.placeholder(tf.float32, name="embedding_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        def _linear(input_, output_size, scope="SimpleLinear"):
            """
            Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
            Args:
                input_: a tensor or a list of 2D, batch x n, Tensors.
                output_size: int, second dimension of W[i].
                scope: VariableScope for the created subgraph; defaults to "SimpleLinear".
            Returns:
                A 2D Tensor with shape [batch x output_size] equal to
                sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
            Raises:
                ValueError: if some of the arguments has unspecified or wrong shape.
            """

            shape = input_.get_shape().as_list()
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: {0}".format(str(shape)))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: {0}".format(str(shape)))
            input_size = shape[1]

            # Now the computation.
            with tf.variable_scope(scope):
                W = tf.get_variable("W", [input_size, output_size], dtype=input_.dtype)
                b = tf.get_variable("b", [output_size], dtype=input_.dtype)

            return tf.nn.xw_plus_b(input_, W, b)

        def _highway_layer(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu):
            """
            Highway Network (cf. http://arxiv.org/abs/1505.00387).
            t = sigmoid(Wy + b)
            z = t * g(Wy + b) + (1 - t) * y
            where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
            """

            for idx in range(num_layers):
                g = f(_linear(input_, size, scope=("highway_lin_{0}".format(idx))))
                t = tf.sigmoid(_linear(input_, size, scope=("highway_gate_{0}".format(idx))) + bias)
                output = t * g + (1. - t) * input_
                input_ = output

            return output

        # Embedding Layer
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            # Use random generated the word vector by default
            # Can also be obtained through our own word vectors trained by our corpus
            self.embedding = tf.get_variable(name='embedding',initializer=self.init_embedding,trainable=True)
            self.embedded_sentence = tf.nn.embedding_lookup(self.embedding, self.x)
            self.embedded_sentence = tf.nn.dropout(self.embedded_sentence,self.embedding_keep_prob)

        # Average Vectors
        self.embedded_sentence_average = tf.reduce_mean(self.embedded_sentence, axis=1)  # [batch_size, embedding_size]

        # Highway Layer
        with tf.name_scope("highway"):
            self.highway = _highway_layer(self.embedded_sentence_average,
                                          self.embedded_sentence_average.get_shape()[1], num_layers=1, bias=0)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.highway, self.keep_prob)

        # Final scores
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[self.embedding_size, self.num_classes],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[self.num_classes], dtype=tf.float32), name="b")
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.prediction=tf.argmax(tf.nn.softmax(self.logits),1,name='prediction')
            self.prediction=tf.cast(self.prediction,tf.int32)
            correct_pred=tf.equal(self.prediction,self.label)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Calculate mean cross-entropy loss, L2 loss
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            losses = tf.reduce_mean(losses)
            #l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
            #                     name="l2_losses") * l2_reg_lambda
            self.loss = losses
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads, vars = zip(*optimizer.compute_gradients(self.loss))
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=2.0)
            self.train_op = optimizer.apply_gradients(zip(grads, vars), global_step=self.global_step, name="train_op")


class FasttextModel():
    def __init__(self,config,save_path):
        self.classifier=TextFAST(sequence_length=config.word_max_len,
                                num_classes=config.num_classes,
                                vocab_size=config.word_vocab_size,
                                embedding_size=config.word_embedding_size)
        self.session=tf.Session()
        with session.as_default:
            self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    #输入一个矩阵
    def predict(self,input_x):
        feed_dict={
            self.classifier.input_x:input_x,
            self.classifier.dropout_keep_prob:1.0,
            self.classifier.embedding_keep_prob:1.0
        }
        y_pred_cls = self.session.run(self.model.prediction, feed_dict=feed_dict)
        return y_pred_cls

class Args():
    def __init__(self):
        print ('配置参数')

flags=Args()
flags.learning_rate=0.001
flags.word_max_len=25
flags.word_embedding_size=300
flags.lstm_hidden_size=256
flags.attention_unit_size=350
flags.attention_hops_size=35
flags.attention_penalization=True
flags.fc_hidden_size=1024
flags.dropout_keep_prob=0.5
flags.learning_rate=1e-3
flags.l2_reg_lambda=0.0
flags.num_classes=1199
flags.train_epoch=20
flags.batch_size=512
flags.data_file='dataset-50w-word.pkl'
flags.best_acc=0.0
flags.save_path=''
FLAGS=flags


def train_fasttext():
    tf.reset_default_graph()
    word_to_id,label_to_id=None,None
    label_to_id_one,label_to_id_two,label_to_id_three={},{},{}
    id_to_label_one,id_to_label_two,id_to_label_three={},{},{}
    labels1,labels2,labels3=[],[],[]
    import pickle
    with open('data/label2idx_dict.pkl','rb') as f:
        label_to_id=pickle.load(f)
    id_to_label={}
    for label in label_to_id.keys():
        id_to_label[label_to_id[label]]=label

    for label in label_to_id.keys():
        cuts=label.split('--')
        if cuts[0] not in labels1:
            labels1.append(cuts[0])
        if cuts[1] not in labels2:
            labels2.append(cuts[1])
        if cuts[2] not in labels3:
            labels3.append(cuts[2])

    for i in range(len(labels1)):
        label_to_id_one[labels1[i]]=i
        id_to_label_one[i]=labels1[i]

    for i in range(len(labels2)):
        label_to_id_two[labels2[i]]=i
        id_to_label_two[i]=labels2[i]

    for i in range(len(labels3)):
        label_to_id_three[labels3[i]]=i
        id_to_label_three[i]=labels3[i]

    print (len(id_to_label_three.keys()))
    words,word_to_id,frequencys=read_vocab(vocab_dir,150000)
    id_to_word={}
    for word in word_to_id:
        id_to_word[word_to_id[word]]=word
    if os.path.exists('data/embedding.pkl'):
        with open('data/embedding.pkl','rb') as f:
            flags.word_embedding=pickle.load(f)
    else:
        flags.word_embedding=create_embedding(words)
        with open('data/embedding.pkl','wb') as f:
            pickle.dump(flags.word_embedding,f)


    x_test,y_test1,y_test2,y_test3=None,None,None,None
    x_train,y_train1,y_train2,y_train3=None,None,None,None
    x_tmp,y_tmp1,y_tmp2,y_tmp3=None,None,None,None
    flags.word_vocab_size=len(words)

    import pickle
    from sklearn.utils import shuffle
    with open('data/train-org.pkl','rb') as f:
        x_tmp,y_tmp=pickle.load(f)
        x_test,y_test=x_tmp[:100000],y_tmp[:100000]
        x_train,y_train=x_tmp[100000:],y_tmp[100000:]

    print (len(x_train))
    x_train,y_train,x_test,y_test=np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)
    x_train,y_train=shuffle(x_train,y_train)
    print (len(words))
    classifier = TextFAST(flags)
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
            return_dict = run_train_step(classifier, sess, (x_batch, y_batch))
            #avg_loss+=return_dict['loss']
            if iters%200==0:
                print (return_dict)
            iters+=1


        t1 = time.time()
        dev_acc,dev_loss = run_eval(classifier, sess,x_test,y_test)
        #saver.save(sess=sess,save_path=flags.save_path)
        print ('Have saved model')
        print("validation accuracy: %.6f loss: %.6f " % (dev_acc,dev_loss))
        if dev_acc > flags.best_acc:
            flags.best_acc=dev_acc
            saver.save(sess=sess,save_path=flags.save_path+str(flags.best_acc))


        print("Train Epoch time:  %.3f s" % (t1 - t0))

    print("Training finished, time consumed : ", time.time() - start, " s")



if __name__ == '__main__':
    train_fasttext()
