import time
from gensim.models.word2vec import Word2Vec
import gensim
import os
import numpy as np
import pickle
import tensorflow as tf


vocab_dir='data/vocab-correct.txt'

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


def run_eval_step(model, sess, batch):
    feed_dict = make_test_feed_dict(model, batch)
    acc,loss = sess.run([model.accuracy,model.loss], feed_dict)
    return acc,loss


def get_attn_weight(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    return sess.run(model.alpha, feed_dict)

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
        curr_acc,curr_loss = run_eval_step(classifier, sess, (x_batch, y_batch))
        total_right+=len(x_batch)*curr_acc
        total_loss+=len(x_batch)*curr_loss
        total_len+=len(x_batch)

    print (total_right)
    print (len(x_dev))
    print (total_len)
    return total_right/total_len,total_loss/total_len

def read_vocab(vocab_dir,vocab_size=10000):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    frequencys=dict()
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
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id,frequencys


def linear(input_, output_size, scope=None):
    """
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


class CNNClassfier(object):
    def __init__(self, config):
        # configuration
        self.max_len = config.word_max_len
        # topic nums + 1
        self.num_classes = config.num_classes
        self.vocab_size = config.word_vocab_size
        self.embedding_size = config.word_embedding_size
        self.filter_sizes = config.filter_sizes
        self.num_filters = config.num_filters
        self.learning_rate = config.learning_rate
        self.init_embedding=config.word_embedding
        self.hidden_size=config.word_hidden_size

        # placeholder
        self.x = tf.placeholder(tf.int32, [None, self.max_len], name="input_x")
        self.label = tf.placeholder(tf.int32, [None], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.embedding_keep_prob=tf.placeholder(tf.float32,name='embedding_keep_prob')

    def build_graph(self):
        print("building graph")
        l2_loss = tf.constant(0.0)
        with tf.variable_scope("discriminator"):
            # Embedding:
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.get_variable(name='embedding',initializer=self.init_embedding,trainable=True)
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.x)  # batch_size * seq * embedding_size
                self.embedded_chars = tf.nn.dropout(self.embedded_chars,self.embedding_keep_prob)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1) # expand dims for conv operation
            pooled_outputs = list()
            # Create a convolution + max-pool layer for each filter size
            for filter_size, filter_num in zip(self.filter_sizes, self.num_filters):
                with tf.name_scope("cov2d-maxpool%s" % filter_size):
                    #filter_shape = [filter_size, self.embedding_size, 1, filter_num]
                    #W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    #b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")
                    #conv = tf.nn.conv2d(self.embedded_chars_expanded,W,strides=[1, 1, 1, 1],padding="VALID",name="conv")
                    # print(conv.name, ": ", conv.shape) batch * (seq - filter_shape) + 1 * 1(output channel) *
                    # filter_num
                    conv=tf.layers.conv2d(self.embedded_chars_expanded,filter_num,(filter_size,self.embedding_size),activation=tf.nn.relu)
                    #h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    #pooled = tf.nn.max_pooling2d(conv,ksize=[1, self.max_len - filter_size + 1, 1, 1],strides=[1, 1, 1, 1],padding='VALID',name="pool")  # 全部池化到 1x1
                    pool=tf.layers.max_pooling2d(conv,(self.max_len-filter_size+1,1),(2,2))
                    print (pool.get_shape())
                    #pool=tf.layers.max_pooling2d(conv,4,4,padding='valid')
                    # print(conv.name, ": ", conv.shape , "----", pooled.name, " : " ,pooled.shape)
                    pooled_outputs.append(pool)
            total_filters_num = sum(self.num_filters)
            '''
            for filter_size, filter_num in zip(self.filter_sizes, self.num_filters):
                with tf.name_scope("cov2d-maxpool%s" % filter_size):
                    filter_shape = [filter_size, self.embedding_size, 1, filter_num]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1,1,1,1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name = "relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.max_len - filter_size + 1, 1, 1],
                        strides=[1,1,1,1],
                        padding="VALID",
                        name="pool")
                    pooled_outputs.append(pooled)


            total_filters_num = sum(self.num_filters)
            '''
            '''
            pooled_outputs = list()
            # Create a convolution + max-pool layer for each filter size
            for filter_size, filter_num in zip(self.filter_sizes, self.num_filters):
                with tf.name_scope("cov2d-maxpool%s" % filter_size):
                    conv = tf.layers.conv1d(self.embedded_chars,filter_num, filter_size)
                    # global max pooling layer
                    pooled = tf.reduce_max(conv, reduction_indices=[4])
                    pooled_outputs.append(pooled)
            '''
            self.h_pool = tf.concat(pooled_outputs, -1)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_filters_num])  # batch * total_num

            # highway network
            '''
            with tf.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)
            '''

            # add droppout
            with tf.name_scope("fc"):
                #dense=tf.layers.dense(inputs=self.h_pool,units=self.hidden_size,activation=tf.nn.relu)
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)



            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([total_filters_num, self.num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.ypred_for_auc = tf.nn.softmax(self.scores)
                self.prediction = tf.cast(tf.argmax(self.ypred_for_auc, 1), dtype=tf.int32)

            with tf.name_scope("loss"):
                losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.label))
                self.loss = losses
            with tf.name_scope("accuracy"):
                self.accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(self.prediction, self.label), tf.float32))

        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        d_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # aggregation_method =2 能够帮助减少内存占用
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params)
        #self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        print("graph built successfully!")


class Args():
    def __init__(self):
        print ('配置参数')

if __name__ == '__main__':
    flags=Args()
    flags.word_max_len=25
    flags.char_max_len=45
    flags.data_file='dataset-50w-word.pkl'
    flags.word_hidden_size=1024
    flags.word_embedding_size=300
    flags.last_rank=0.0
    flags.best_acc=0.0
    flags.learning_rate=1e-3
    flags.train_epoch=20
    flags.batch_size=512
    flags.epsilon=5
    flags.save_path='ckpt-50w/cnn'
    flags.best_acc=0.0
    flags.filter_sizes=[2,3,4,5]
    flags.num_classes=1258
    flags.num_filters=[256,256,256,256]

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
    flags.word_vocab_size=len(words)

    import pickle
    from sklearn.utils import shuffle
    with open('data/train-org.pkl','rb') as f:
        x_tmp,y_tmp=pickle.load(f)
        x_train,y_train=x_tmp[100000:],y_tmp[100000:]
        x_test,y_test=x_tmp[:100000],y_tmp[:100000]
    x_train,y_train,x_test,y_test=np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)
    '''
    with open('tmp.pkl','wb') as f:
        pickle.dump((x_train,y_train,x_test,y_test),f)
    '''
    print (x_train[:10])

    classifier = CNNClassfier(flags)
    classifier.build_graph()

    saver=tf.train.Saver()

    # auto GPU growth, avoid occupy all GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    sess.run(tf.global_variables_initializer())
    #saver.restore(sess=sess,save_path=flags.save_path)
    start = time.time()
    for e in range(flags.train_epoch):
        t0 = time.time()
        print("Epoch %d start !" % (e + 1))
        iters=0
        for x_batch, y_batch in fill_feed_dict(x_train, y_train, flags.batch_size):
            #if e==0 and (iters+1)<=10000:
            #    iters+=1
            #    continue
            return_dict = run_train_step(classifier, sess, (x_batch, y_batch))
            if iters%200==0:
                print (return_dict)
            iters+=1
        dev_acc,dev_loss = run_eval(classifier, sess,x_test,y_test)
        print("validation accuracy: %.3f " % dev_acc)
        if dev_acc > flags.best_acc:
            flags.best_acc=dev_acc
            #saver.save(sess=sess,save_path=flags.save_path)
            print ('Have saved model')


        t1 = time.time()


        print("Train Epoch time:  %.3f s" % (t1 - t0))

    print("Training finished, time consumed : ", time.time() - start, " s")
