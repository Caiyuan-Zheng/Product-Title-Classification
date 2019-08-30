import tensorflow as tf
import time
from gensim.models.word2vec import Word2Vec
import gensim
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
import pickle


vocab_dir='data/vocab-correct.txt'


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
    'optimizer': model.optimizer,
    'loss': model.loss,
    'accuracy':model.accuracy,
    'global_step_op':model.global_step_op,
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

'''
生成词向量矩阵,包括word2vec和glove

'''
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


class ABLSTM(object):
    def __init__(self, config):
        self.max_len=config.word_max_len
        self.hidden_size=config.word_hidden_size
        self.vocab_size=config.word_vocab_size
        self.embedding_size=config.word_embedding_size
        self.n_class=config.num_classes
        self.learning_rate=config.learning_rate
        self.max_grad_norm=5.0

        # placeholder
        self.x = tf.placeholder(tf.int32, [None, self.max_len],name='input_x')
        self.label = tf.placeholder(tf.int32, [None],name='input_y')
        self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        self.embedding_keep_prob=tf.placeholder(tf.float32,name='embedding_keep_prob')
        self.init_embedding=config.word_embedding


    def build_graph(self):
        print("building graph")
        # Word embedding
        embeddings_var = tf.get_variable(name='embedding',initializer=self.init_embedding,trainable=True)
        print ('finish init embedding')
        #embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
        #                           trainable=True)
        batch_embedded = tf.nn.embedding_lookup(embeddings_var, self.x)
        batch_embedded = tf.nn.dropout(batch_embedded,self.embedding_keep_prob)

        rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(tf.nn.rnn_cell.LSTMCell(self.hidden_size,forget_bias=0.9,name='cell1'),
                                tf.nn.rnn_cell.LSTMCell(self.hidden_size,forget_bias=0.9,name='cell2'),
                                inputs=batch_embedded, dtype=tf.float32)

        fw_outputs, bw_outputs = rnn_outputs

        W = tf.Variable(tf.random_normal([self.hidden_size], stddev=0.1))
        H = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
        M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)

        self.alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, self.hidden_size]),
                                                        tf.reshape(W, [-1, 1])),
                                              (-1, self.max_len)))  # batch_size x seq_len
        r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                      tf.reshape(self.alpha, [-1, self.max_len, 1]))
        r = tf.squeeze(r)
        h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE

        h_drop = tf.nn.dropout(h_star, self.keep_prob)

        FC_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.n_class], stddev=0.1))
        FC_b = tf.Variable(tf.constant(0., shape=[self.n_class]))
        y_hat = tf.nn.xw_plus_b(h_drop, FC_W, FC_b,name='logits')
        self.softmax=tf.nn.softmax(y_hat,name='softmax')
        self.logits=y_hat


        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=self.label))


        self.prediction = tf.argmax(tf.nn.softmax(y_hat), 1)
        self.prediction=tf.cast(self.prediction,tf.int32)
        self.top_k=tf.nn.top_k(y_hat,3,name='top_k')
        #self.top_k=tf.cast(self.top_k,tf.int32)
        self.correct_pred = tf.equal(self.label, self.prediction)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        # optimization
        variable_names = [v.name for v in tf.all_variables()]
        print (variable_names)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.global_step_op=self.global_step.assign(self.global_step+1)
        print("graph built successfully!")


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
    flags.train_epoch=20
    flags.batch_size=512
    flags.epsilon=5
    flags.save_path=''
    flags.best_acc=0.000
    flags.num_classes=1258
    flags.word_embedding=None

    tf.reset_default_graph()
    word_to_id,label_to_id=None,None
    label_to_id_one,label_to_id_two,label_to_id_three={},{},{}
    id_to_label_one,id_to_label_two,id_to_label_three={},{},{}
    labels1,labels2,labels3=[],[],[]

    with open('data/label2idx_dict.pkl','rb') as f:
        label_to_id=pickle.load(f)
    id_to_label={}
    for label in label_to_id.keys():
        id_to_label[label_to_id[label]]=label

    for label in label_to_id.keys():
        cuts=label.split('--')
        cate1=cuts[0]
        cate2=cuts[0]+'--'+cuts[1]
        cate3=label
        if cate1 not in labels1:
            labels1.append(cate1)
        if cate2 not in labels2:
            labels2.append(cate2)
        if cate3 not in labels3:
            labels3.append(cate3)

    for i in range(len(labels1)):
        label_to_id_one[labels1[i]]=i
        id_to_label_one[i]=labels1[i]

    for i in range(len(labels2)):
        label_to_id_two[labels2[i]]=i
        id_to_label_two[i]=labels2[i]

    for i in range(len(labels3)):
        label_to_id_three[labels3[i]]=i
        id_to_label_three[i]=labels3[i]

    print (len(id_to_label_one.keys()))
    print (len(id_to_label_two.keys()))
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

    #flags.word_embedding=flags.word_embedding.astype(np.float32)
    #flags.word_embedding=create_embedding(words)

    x_test,y_test1,y_test2,y_test3=None,None,None,None
    x_train,y_train1,y_train2,y_train3=None,None,None,None
    x_tmp,y_tmp1,y_tmp2,y_tmp3=None,None,None,None
    flags.word_vocab_size=len(words)

    import pickle
    from sklearn.utils import shuffle
    with open('data/train-multi.pkl','rb') as f:
        x_tmp,y_tmp1,y_tmp2,y_tmp3=pickle.load(f)
        x_test,y_test1,y_test2,y_test3=x_tmp[:100000],y_tmp1[:100000],y_tmp2[:100000],y_tmp3[:100000]
        x_train,y_train1,y_train2,y_train3=x_tmp[100000:],y_tmp1[100000:],y_tmp2[100000:],y_tmp3[100000:]
    print (len(x_train))
    x_train,y_train1,y_train2,y_train3,x_test,y_test1,y_test2,y_test3=np.array(x_train),np.array(y_train1),np.array(y_train2),np.array(y_train3),np.array(x_test),np.array(y_test1),np.array(y_test2),np.array(y_test3)
    print (x_train[:10])
    x_train,y_train1,y_train2,y_train3=shuffle(x_train,y_train1,y_train2,y_train3)
    #x_train,y_train=x_train[:400000],y_train[:400000]
    print (y_tmp3[:10])
    print (len(words))
    classifier = ABLSTM(flags)
    classifier.build_graph()
    saver=tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    start = time.time()
    for e in range(flags.train_epoch):
        t0 = time.time()
        print("Epoch %d start !" % (e + 1))
        iters=0
        for x_batch, y_batch in fill_feed_dict(x_train, y_train3, flags.batch_size):
            return_dict = run_train_step(classifier, sess, (x_batch, y_batch))
            if iters%200==0:
                print (return_dict)
            iters+=1


        t1 = time.time()
        dev_acc,dev_loss = run_eval(classifier, sess,x_test,y_test3)
        #saver.save(sess=sess,save_path=flags.save_path)
        print ('Have saved model')
        print("validation accuracy: %.6f loss: %.6f " % (dev_acc,dev_loss))
        if dev_acc > flags.best_acc:
            flags.best_acc=dev_acc
            saver.save(sess=sess,save_path=flags.save_path+str(flags.best_acc))


        print("Train Epoch time:  %.3f s" % (t1 - t0))

    print("Training finished, time consumed : ", time.time() - start, " s")