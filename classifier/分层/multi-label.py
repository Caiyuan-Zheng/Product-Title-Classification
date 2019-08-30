'''
包括两种层次模型
'''
import tensorflow as tf
import pickle

from gensim.models.word2vec import Word2Vec
import gensim
import os
import numpy as np
import pickle
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle

vocab_dir_word = os.path.join('data', 'vocab-correct.txt')
vocab_dir_char = os.path.join('data','vocab-char.txt')

class Linear(object):
    def __init__(self):

        # placeholder
        self.predict1 = tf.placeholder(tf.float32, [None, 1258])
        self.predict2 = tf.placeholder(tf.float32, [None, 1258])
        self.predict3 = tf.placeholder(tf.float32, [None, 1258])
        self.predict4 = tf.placeholder(tf.float32, [None, 1258])
        self.label = tf.placeholder(tf.int32, [None])
        self.weight1=tf.placeholder(tf.float32,name='weight1')
        self.weight2=tf.placeholder(tf.float32,name='weight2')
        self.weight3=tf.placeholder(tf.float32,name='weight3')
        self.weight4=tf.placeholder(tf.float32,name='weight4')
        self.learning_rate=1e-3


    def build_graph(self):
        y_hat=self.predict1*self.weight1+self.predict2*self.weight2+self.predict3*self.weight3+self.predict4*self.weight4




        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=self.label))


        self.prediction = tf.argmax(tf.nn.softmax(y_hat), 1)
        self.prediction=tf.cast(self.prediction,tf.int32)
        correct_pred = tf.equal(self.label, self.prediction)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # optimization
        self.optimizer = None
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        #self.global_step_op=self.global_step.assign(self.global_step+1)
        print("graph built successfully!")

class AdversarialModel():
    def __init__(self,config,save_path,frequencys,word2idx):
        self.graph=tf.Graph()
        #self.classifier = AdversarialClassifier(config)
        #self.classifier.build_graph(frequencys, word2idx)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.session=tf.Session(graph=self.graph)
        with self.session.as_default():
            with self.graph.as_default():
                tf.global_variables_initializer().run()
                saver = tf.train.import_meta_graph(save_path+'.meta')
                saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型
                self.input_x=self.graph.get_operation_by_name('input_x').outputs[0]
                self.keep_prob=self.graph.get_operation_by_name('keep_prob').outputs[0]
                self.embedding_keep_prob=self.graph.get_operation_by_name('embedding_keep_prob').outputs[0]
                self.prediction=self.graph.get_operation_by_name('loss/logits').outputs[0]

    #输入一个矩阵
    def predict(self,input_x):
        feed_dict={
            self.input_x:input_x,
            self.keep_prob:1.0,
            self.embedding_keep_prob:1.0
        }
        y_pred_cls = self.session.run(self.prediction, feed_dict=feed_dict)
        return y_pred_cls


class AdversarialCharModel():
    def __init__(self,config,save_path,frequencys,word2idx):
        self.graph=tf.Graph()
        #self.classifier = AdversarialClassifier(config)
        #self.classifier.build_graph(frequencys, word2idx)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.session=tf.Session(graph=self.graph)
        with self.session.as_default():
            with self.graph.as_default():
                tf.global_variables_initializer().run()
                saver = tf.train.import_meta_graph(save_path+'.meta')
                saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型
                self.input_x=self.graph.get_operation_by_name('input_x').outputs[0]
                self.keep_prob=self.graph.get_operation_by_name('keep_prob').outputs[0]
                self.embedding_keep_prob=self.graph.get_operation_by_name('embedding_keep_prob').outputs[0]
                self.prediction=self.graph.get_operation_by_name('loss/logits').outputs[0]

    #输入一个矩阵
    def predict(self,input_x):
        feed_dict={
            self.input_x:input_x,
            self.keep_prob:1.0,
            self.embedding_keep_prob:1.0
        }
        y_pred_cls = self.session.run(self.prediction, feed_dict=feed_dict)
        return y_pred_cls

class AbilstmModel():
    def __init__(self,config,save_path):
        self.graph=tf.Graph()
        #self.classifier = ABLSTM(config)
        #self.classifier.build_graph()
        self.session=tf.Session(graph=self.graph)
        with self.session.as_default():
            with self.graph.as_default():
                tf.global_variables_initializer().run()
                saver = tf.train.import_meta_graph(save_path+'.meta')
                saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型
                layers = [op.name for op in self.graph.get_operations()]
                print (layers)
                self.input_x=self.graph.get_operation_by_name('input_x').outputs[0]
                self.keep_prob=self.graph.get_operation_by_name('keep_prob').outputs[0]
                self.embedding_keep_prob=self.graph.get_operation_by_name('embedding_keep_prob').outputs[0]
                self.prediction=self.graph.get_operation_by_name('logits').outputs[0]

    #输入一个矩阵
    def predict(self,input_x):
        feed_dict={
            self.input_x:input_x,
            self.keep_prob:1.0,
            self.embedding_keep_prob:1.0
        }
        softmax = self.session.run(self.prediction, feed_dict=feed_dict)
        #print (prediction)
        return softmax

class AbilstmCharModel():
    def __init__(self,config,save_path):
        self.graph=tf.Graph()
        #self.classifier = ABLSTM(config)
        #self.classifier.build_graph()
        self.session=tf.Session(graph=self.graph)
        with self.session.as_default():
            with self.graph.as_default():
                tf.global_variables_initializer().run()
                saver = tf.train.import_meta_graph(save_path+'.meta')
                saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型
                layers = [op.name for op in self.graph.get_operations()]
                print (layers)
                self.input_x=self.graph.get_operation_by_name('input_x').outputs[0]
                self.keep_prob=self.graph.get_operation_by_name('keep_prob').outputs[0]
                self.embedding_keep_prob=self.graph.get_operation_by_name('embedding_keep_prob').outputs[0]
                self.prediction=self.graph.get_operation_by_name('logits').outputs[0]

    #输入一个矩阵
    def predict(self,input_x):
        feed_dict={
            self.input_x:input_x,
            self.keep_prob:1.0,
            self.embedding_keep_prob:1.0
        }
        prediction,value = self.session.run(self.prediction, feed_dict=feed_dict)
        return prediction,value


class FasttextModel():
    def __init__(self,config,save_path):
        self.graph=tf.Graph()
        #self.classifier=TextFAST(sequence_length=config.word_max_len,
        #                        num_classes=config.num_classes,
        #                        vocab_size=config.word_vocab_size,
        #                        embedding_size=config.word_embedding_size)
        self.session=tf.Session(graph=self.graph)
        with self.session.as_default():
            with self.graph.as_default():
                tf.global_variables_initializer().run()
                saver = tf.train.import_meta_graph(save_path+'.meta')
                saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型
                self.input_x=self.graph.get_operation_by_name('input_x').outputs[0]
                self.keep_prob=self.graph.get_operation_by_name('keep_prob').outputs[0]
                self.embedding_keep_prob=self.graph.get_operation_by_name('embedding_keep_prob').outputs[0]
                self.prediction=self.graph.get_operation_by_name('output/logits').outputs[0]

    #输入一个矩阵
    def predict(self,input_x):
        feed_dict={
            self.input_x:input_x,
            self.keep_prob:1.0,
            self.embedding_keep_prob:1.0
        }
        y_pred_cls = self.session.run(self.prediction, feed_dict=feed_dict)
        return y_pred_cls

class CNNModel():
    def __init__(self,config,save_path):
        self.graph=tf.Graph()
        #self.classifier=TextFAST(sequence_length=config.word_max_len,
        #                        num_classes=config.num_classes,
        #                        vocab_size=config.word_vocab_size,
        #                        embedding_size=config.word_embedding_size)
        self.session=tf.Session(graph=self.graph)
        with self.session.as_default():
            with self.graph.as_default():
                tf.global_variables_initializer().run()
                saver = tf.train.import_meta_graph(save_path+'.meta')
                saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型
                self.input_x=self.graph.get_operation_by_name('input_x').outputs[0]
                self.keep_prob=self.graph.get_operation_by_name('keep_prob').outputs[0]
                self.prediction=self.graph.get_operation_by_name('discriminator/output/scores').outputs[0]

    #输入一个矩阵
    def predict(self,input_x):
        feed_dict={
            self.input_x:input_x,
            self.keep_prob:1.0,
        }
        y_pred_cls = self.session.run(self.prediction, feed_dict=feed_dict)
        return y_pred_cls


def fill_feed_dict(data_X_Char,data_X_Word, data_Y, batch_size):
    """Generator to yield batches"""
    # Shuffle data first.
    shuffled_X_Char,shuffled_X_Word, shuffled_Y = shuffle(data_X_Char,data_X_Word, data_Y)
    # print("before shuffle: ", data_Y[:10])
    # print(data_X.shape[0])
    # perm = np.random.permutation(data_X.shape[0])
    # data_X = data_X[perm]
    # shuffled_Y = data_Y[perm]
    # print("after shuffle: ", shuffled_Y[:10])
    for idx in range(data_X_Char.shape[0] // batch_size):
        x_batch_char = shuffled_X_Char[batch_size * idx: batch_size * (idx + 1)]
        x_batch_word = shuffled_X_Word[batch_size * idx: batch_size* (idx+1)]
        y_batch = shuffled_Y[batch_size * idx: batch_size * (idx + 1)]
        yield x_batch_char,x_batch_word, y_batch
def make_train_feed_dict(model, batch):
    feed_dict = {model.predict1:batch[0],
                model.predict2:batch[1],
                 model.label: batch[3]}
    return feed_dict


def make_test_feed_dict(model, batch,weights):
    weight1=weights[0]
    weight2=weights[1]
    weight3=weights[2]
    weight4=weights[3]
    feed_dict = {model.predict1:batch[0],
                model.predict2:batch[1],
                model.predict3:batch[2],
                model.predict4:batch[3],
                 model.label: batch[4],
                 model.weight1:weight1,
                 model.weight2:weight2,
                 model.weight3:weight3,
                 model.weight4:weight4}
    return feed_dict


def run_train_step(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    to_return = {
        'loss': model.loss,
        'accuracy':model.accuracy
    }
    return sess.run(to_return, feed_dict)


def run_eval_step(model, sess, batch,weights):
    feed_dict = make_test_feed_dict(model, batch,weights)
    acc,loss = sess.run([model.accuracy,model.loss], feed_dict)
    return acc,loss


def run_eval(classifier,adversial,adversial_char,abilstm,abilstm_char, sess, x_dev_char,x_dev_word,y_dev,weights):
    print ('Start Validation')
    total_right=0.0
    total_loss=0.0
    total_len=0.0
    #num_batch=len()
    for x_batch_char,x_batch_word, y_batch in fill_feed_dict(x_dev_char,x_dev_word, y_dev, 256):
        predict1=adversial.predict(x_batch_word)
        predict2=adversial_char.predict(x_batch_char)
        predict3=abilstm.predict(x_batch_word)
        predict4=abilstm_char.predict(x_batch_char)
        curr_acc,curr_loss = run_eval_step(classifier, sess, (predict1,predict2,predict3,predict4, y_batch),weights)
        total_right+=len(x_batch_char)*curr_acc
        total_loss+=len(x_batch_char)*curr_loss
        total_len+=len(x_batch_char)

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


class Args():
    def __init__(self):
        print ('配置参数')

flags=Args()
flags.word_max_len=25
flags.char_max_len=45
flags.char_embedding_size=300
flags.word_embedding_size=300
flags.word_hidden_size=384
flags.char_hidden_size=256
flags.batch_size=256
flags.word_data_file='dataset-50w-word.pkl'#存放训练集缓存
flags.char_data_file='dataset-50w-char.pkl'
flags.filter_sizes=[2,3,4,5,6,7]
flags.num_filter=128
flags.num_classes=1258
flags.word_vocab_size=100001
flags.char_vocab_size=6000
flags.keep_prob=0.5
flags.embedding_keep_prob=0.8
flags.train_epoch=20
flags.epsilon=5
flags.learning_rate=1e-3
flags.save_path='ckpt-50w/en'
flags.best_acc=0.0

x_train_word,x_train_char,y_train,x_dev_word,x_dev_char,y_dev=None,None,None,None,None,None




def ensemble():
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

    one_to_two={}
    for i in range(len(labels1)):
        label1=labels1[i]
        one_to_two[i]=[]
        for j in range(len(labels2)):
            label2=labels2[j]
            if label2.find(label1)!=-1:
                one_to_two[i].append(j)

    two_to_three={}
    for i in range(len(labels2)):
        label2=labels2[i]
        two_to_three[i]=[]
        for j in range(len(labels3)):
            label3=labels3[j]
            if label3.find(label2)!=-1:
                two_to_three[i].append(j)

    flags.save_path_two='ckpt-multi/fasttext-two0.909725560897436'
    flags.save_path_three='ckpt-multi/fasttext-three0.8701822916666667'
    flags.save_path_one='ckpt-multi/fasttext-one0.947636217948718'

    #cnn=CNNModel(flags,flags.cnn_savepath)
    #adversial=AdversarialModel(flags,flags.ad_savepath,frequencys_word,word2idx_word)
    #fasttext=FasttextModel(flags,flags.fasttext_savepath)
    #abilstm_two=AbilstmModel(flags,flags.save_path_two)
    #abilstm_three=AbilstmModel(flags,flags.save_path_three)
    #abilstm_one=AbilstmModel(flags,flags.save_path_one)
    fasttext_one=FasttextModel(flags,flags.save_path_one)
    fasttext_two=FasttextModel(flags,flags.save_path_two)
    fasttext_three=FasttextModel(flags,flags.save_path_three)
    x_test,y_test=None,None
    with open('data/train-multi.pkl','rb') as f:
        x_tmp,y_tmp1,y_tmp2,y_tmp3=pickle.load(f)
        x_test,y_test1,y_test2,y_test3=x_tmp[:100000],y_tmp1[:100000],y_tmp2[:100000],y_tmp3[:100000]
        x_train,y_train1,y_train2,y_train3=x_tmp[100000:],y_tmp1[100000:],y_tmp2[100000:],y_tmp3[100000:]
        x_test,y_test=x_test,y_test3


    # auto GPU growth, avoid occupy all GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    #zuhe=[[0.5,0.2,0.5,0.2],[0.4,0.2,0.4,0.2],[0.6,0.03,0.4,0.07],[0.4,0.1,0.6,0.1]]
    inputs=tf.placeholder(tf.float32,[None])
    logits_input=tf.placeholder(tf.float32,[None,None])
    softmax=tf.nn.softmax(logits_input)
    length=tf.placeholder(tf.int32)
    #gailv=tf.nn.sigmoid(inputs)
    outputs=tf.nn.top_k(inputs,length)
    zuhe=[[0.6,0.05,0.4,0.03],[0.6,0.1,0.4,0.05]]
    with sess.as_default():
        model=Linear()
        model.build_graph()
        sess.run(tf.global_variables_initializer())
        start=0
        end=0
        right_num=0
        rank=0
        houxuan=5
        logits1,logits2,logits3=[],[],[]
        start=
        while end<len(x_test):
            start=end
            end=start+1024
            if end>len(x_test):
                end=len(x_test)
            x_batch,y_batch=x_test[start:end],y_test[start:end]
            #values1,predict1=abilstm_one.predict(x_batch)
            #values2,predict2=abilstm_two.predict(x_batch)
            #values3,predict3=abilstm_three.predict(x_batch)

            logit1=fasttext_one.predict(x_batch)
            logit1=sess.run(softmax,{logits_input:logit1})
            logits1.extend(logit1)
            logit2=fasttext_two.predict(x_batch)
            logit2=sess.run(softmax,{logits_input:logit2})
            logits2.extend(logit2)
            logit3=fasttext_three.predict(x_batch)
            logit3=sess.run(softmax,{logits_input:logit3})
            logits3.extend(logit3)

        #valuess,indicess=sess.run(outputs,feed_dict={inputs:logits1})
        for i in range(len(x_test)):
            max_value,max_p=-1,-1
            logit1=logits1[i]
            values1,indices1=sess.run(outputs,{inputs:logit1,length:houxuan})
            for x in range(3):
                value1,indice1=values1[x],indices1[x]
                values2,indices2=[],[]
                for j in one_to_two[indice1]:
                    values2.append(logits2[i][j])
                    indices2.append(j)
                length2=min(len(values2),houxuan)
                _,tmp_indices=sess.run(outputs,{inputs:values2,length:length2})
                values2,indices2=np.array(values2)[tmp_indices],np.array(indices2)[tmp_indices]

                for y in range(length2):
                    value2,indice2=values2[y],indices2[y]
                    values3,indices3=[],[]
                    for k in two_to_three[indice2]:
                        values3.append(logits3[i][k])
                        indices3.append(k)
                    length3=min(len(values3),houxuan)
                    _,tmp_indices=sess.run(outputs,{inputs:values3,length:length3})
                    values3,indices3=np.array(values3)[tmp_indices],np.array(indices3)[tmp_indices]

                    for z in range(length3):
                        value3,indice3=values3[z],indices3[z]
                        current_value=1*value1+2*value2+4*value3
                        if current_value>max_value:
                            max_value=current_value
                            max_p=indice3
            label=y_test[i]
            if max_p==label:
                right_num+=1
            if i%1000==0:
                rate=right_num/(i+1)
                print ('当前数量: '+str(i)+' 正确数量 '+str(right_num)+' 正确率 '+str(rate))
        print ('right_num: '+str(right_num))
        '''
        for i in range(len(x_test)):
            _,indice1=sess.run(outputs,{inputs:logits1[i],length:1})
            indice1=indice1[0]
            _,indice2=sess.run(outputs,{inputs:logits2[i],length:1})
            indice2=indice2[0]
            _,indices3=sess.run(outputs,{inputs:logits3[i],length:3})
            flag=True
            predict=indices3[0]
            if labels3[predict].find(labels1[indice1])!=-1:
                flag=False
            if flag==False:
                for indice3 in indices3:
                    if indice3 in two_to_three[indice2]:
                        predict=indice3
                        flag=True
                        break
            if flag==False:
                for indice3 in indices3:
                    if labels3[indice3].find(labels1[indice1])!=-1:
                        predict=indice3
                        flag=True
                        break
            if predict==y_test[i]:
                right_num+=1
            if i%1000==0:
                rate=right_num/(i+1)
                print ('当前数量: '+str(i)+' 正确数量 '+str(right_num)+' 正确率 '+str(rate))
        print ('right_num: '+str(right_num))
        '''


if __name__ == '__main__':
    ensemble()