#!/usr/bin/env python
#Reference: https://www.amazon.co.jp/product-reviews/4839962510/
#           https://arxiv.org/pdf/1503.08895.pdf

import pandas as pd
import numpy as np
import ml_metrics as metrics
import nltk
import nltk.data
import sklearn.metrics as mtrx

# Makoto.Sano@Mack-the-Psych.com 08/22/2020
# import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

np.random.seed(0)

# Makoto.Sano@Mack-the-Psych.com 08/22/2020
import tf_log_regress_classify as lreg
import tensorflow as tf

# Modified by mack.sano@gmail.com 6/20/2020
# tf.compat.v1.set_random_seed(1234)
tf.set_random_seed(1234)

# Makoto.Sano@Mack-the-Psych.com 08/22/2020
# import tf_log_regress_classify as lreg

# Modified by mack.sano@gmail.com 3/14/2020
import os
import shutil
LOG_DIR = os.path.join(os.path.dirname(__file__), 'log')

class tmv_tf_memn_classify(lreg.tmv_tf_log_regress_classify):
    def __init__(self, data_dir=r'./'):
        self.data_dir = data_dir
        
    def load_data(self, csv_file_kspa, dependent_var, langs = None, task_word = 'Definition',
                  answer_ex_clm = 'Definition'):
        # Modified by mack.sano@gmail.com 3/22/2020
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        os.mkdir(LOG_DIR)

        self.dependent_var = dependent_var
        self.df_response_answer_ex = pd.read_csv(self.data_dir + csv_file_kspa, encoding= 'latin1')
        self.df_response_answer_ex = self.df_response_answer_ex.set_index(r'Student_Question_Index')

        if langs != None:
            lang_clm = task_word + r'-Language'
            self.df_response_answer_ex = \
                self.df_response_answer_ex[self.df_response_answer_ex[lang_clm].isin(langs)]
            
        ans_clm = task_word + r'-Answer'

        ans_tokens_all = self.get_tokens(ans_clm)
        ans_ex_tokens_all = self.get_tokens(answer_ex_clm)

        self.vocab = set()
        for x in ans_tokens_all + ans_ex_tokens_all:
            self.vocab |= set(x)
        self.vocab = sorted(self.vocab)
        self.vocab_size = len(self.vocab) + 1  # for padding +1

        self.df_ac_modeling_values = pd.DataFrame({'Anser_Tokens': ans_tokens_all,
                                                   'Anser_example_Tokens': ans_ex_tokens_all},
                                                  index = self.df_response_answer_ex.index)

        self.df_ac_modeling_values[self.dependent_var] = \
                 self.df_response_answer_ex[self.dependent_var]

        self.word_indices = dict((c, i + 1) for i, c in enumerate(self.vocab))

        self.ans_ex_maxlen = \
            max(map(len, (x for x in ans_ex_tokens_all)))
        self.ans_maxlen = \
            max(map(len, (x for x in ans_tokens_all)))
            
        # Modified by mack.sano@gmail.com 3/20/2020
        words = ["{word}\n".format(word=x) for x in self.vocab]
        with open( LOG_DIR + "/words.tsv", 'w', encoding="utf-8") as f:
            f. writelines(words)

    def get_tokens(self, content_column):
        df_ac_buf = self.df_response_answer_ex.copy()
        list_cntnt = list(df_ac_buf[content_column])
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

        tokens_all = []
        for x in list_cntnt:
            tokens = []
            sentences = sent_detector.tokenize(x.strip())
            for y in sentences:
                # modified by Makoto.Sano@Mack-the-Psych.com 9/8/2020
                # print(y)
                tokens += nltk.word_tokenize(y)

            # modified by Makoto.Sano@Mack-the-Psych.com 9/8/2020
            # print(tokens)
            tokens_all = tokens_all + [tokens]

        return tokens_all
        
    def vectorize_tokens(self, data, maxlen):
        X = []
        for answer in data:
            #Modified by Sano.Makoto@otsuka.jp 4/28/2019
            #x = [self.word_indices[w] for w in answer]
            x = []
            for w in answer:
                if w in self.word_indices:
                    x = x + [self.word_indices[w]]
            X.append(x)
        return (self.padding(X, maxlen = maxlen))

    def padding(self, words, maxlen):
        for i, word in enumerate(words):
            words[i] = [0] * (maxlen - len(word)) + word
        return np.array(words)

    def inference(self, x, q, n_batch, number_class,
                  vocab_size=None,
                  embedding_dim=None,
                  ans_maxlen=None):        
        # Modified by mack.sano@gmail.com 3/20/2020
        def weight_variable(shape, stddev=0.08, name='Variable'):
            initial = tf.random.truncated_normal(shape, stddev=stddev)
            return tf.Variable(initial, name=name)
        def bias_variable(shape, name='Variable'):
            initial = tf.zeros(shape, dtype=tf.float32)
            return tf.Variable(initial, name=name)

        # Modified by mack.sano@gmail.com 3/20/2020
        A = weight_variable([vocab_size, embedding_dim], name='A')
        B = weight_variable([vocab_size, embedding_dim], name='B')
        C = weight_variable([vocab_size, ans_maxlen], name='C')
                        
        m = tf.nn.embedding_lookup(A, x)
        u = tf.nn.embedding_lookup(B, q)
        c = tf.nn.embedding_lookup(C, x)
        p = tf.nn.softmax(tf.einsum('ijk,ilk->ijl', m, u))
        o = tf.add(p, c)
        o = tf.transpose(o, perm=[0, 2, 1])
        ou = tf.concat([o, u], axis=-1)

        cell = tf.contrib.rnn.BasicLSTMCell(embedding_dim//2, forget_bias=1.0)
        initial_state = cell.zero_state(n_batch, tf.float32)
        state = initial_state
        outputs = []

        # modified by Makoto.Sano@Mack-the-Psych.com 9/8/2020
        # with tf.variable_scope('LSTM'):
        with tf.variable_scope('LSTM', reuse=tf.AUTO_REUSE):
            for t in range(ans_maxlen):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(ou[:, t, :], state)
                outputs.append(cell_output)
        output = outputs[-1]
        
        # Modified by mack.sano@gmail.com 3/20/2020
        W = weight_variable([embedding_dim//2, number_class], stddev=0.01, name='W')
        b = bias_variable([number_class], name='b')

        a = tf.nn.softmax(tf.matmul(output, W) + b)

        return a

    def loss(self, y, t):
        cross_entropy = \
            tf.reduce_mean(-tf.reduce_sum(
                           t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),
                           reduction_indices=[1]))
        return cross_entropy

    def training(self, loss):
        optimizer = \
            tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
        train_step = optimizer.minimize(loss)
        return train_step

    def accuracy(self, y, t):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy
    
    
    def perform_modeling(self, df_ac_modeling_data, key_word = r'', csv_dump = False,
                         number_class = 3, epochs = 200, batch_size = 100, embedding_dim = 64): # Modified by mack.sano@gmail.com 3/21/2020
        #Modified by Sano.Makoto@otsuka.jp 4/21/2019
        #df_ac_modeling_data = self.df_ac_modeling_values.iloc[self.concatenated_value_order, :]
        df_ac_modeling_target = df_ac_modeling_data.loc[:,[self.dependent_var]]
        
        y_train = df_ac_modeling_target.transpose().values[0]
        y_matrix_train = y_train.reshape(len(y_train),1)
        ohe = OneHotEncoder(categorical_features=[0])
        y_ohe_train = ohe.fit_transform(y_matrix_train).toarray()
        ans_tokens_vector = self.vectorize_tokens(list(df_ac_modeling_data['Anser_Tokens']),
                                                  self.ans_maxlen)
        ans_ex_tokens_vector = self.vectorize_tokens(list(df_ac_modeling_data['Anser_example_Tokens']),
                                                  self.ans_ex_maxlen)

        print('Building model...')
        tf.reset_default_graph()
        x = tf.placeholder(tf.int32, shape=[None, self.ans_ex_maxlen])
        q = tf.placeholder(tf.int32, shape=[None, self.ans_maxlen])
        a = tf.placeholder(tf.float32, shape=[None, number_class])
        n_batch = tf.placeholder(tf.int32, shape=[])

        self.y = self.inference(x, q, n_batch, number_class,
                      vocab_size=self.vocab_size,
                      embedding_dim=embedding_dim, # Modified by mack.sano@gmail.com 3/21/2020
                      ans_maxlen=self.ans_maxlen)
        loss = self.loss(self.y, a)
        
        # Modified by mack.sano@gmail.com 3/14/2020
        tf.summary.scalar('cross_entropy', loss)  # for TensorBoard
        
        train_step = self.training(loss)
        acc = self.accuracy(self.y, a)
        history = {
            'val_loss': [],
            'val_acc': []
        }
        
        print('Training model...')
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        
        # Modified by mack.sano@gmail.com 3/14/2020; 6/20/2020
        #file_writer = tf.compat.v1.summary.FileWriter(LOG_DIR, self.sess.graph)
        file_writer = tf.summary.FileWriter(LOG_DIR, self.sess.graph)
        summaries = tf.summary.merge_all()  # merge all variables
        saver = tf.train.Saver(max_to_keep=3)

        self.sess.run(init)

        n_batches = len(ans_ex_tokens_vector) // batch_size

        for epoch in range(epochs):
            ans_ex_tokens_vector_, ans_tokens_vector_, y_ohe_train_ = \
                shuffle(ans_ex_tokens_vector, ans_tokens_vector, y_ohe_train)

            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size

                self.sess.run(train_step, feed_dict={
                    x: ans_ex_tokens_vector_[start:end],
                    q: ans_tokens_vector_[start:end],
                    a: y_ohe_train_[start:end],
                    n_batch: batch_size
                })
                
                # Modified by mack.sano@gmail.com 3/14/2020
                summary, loss2 = self.sess.run([summaries, loss], feed_dict={
                    x: ans_ex_tokens_vector_[start:end],
                    q: ans_tokens_vector_[start:end],
                    a: y_ohe_train_[start:end],
                    n_batch: batch_size
                })                                    
                
            val_loss = loss.eval(session=self.sess, feed_dict={
                x: ans_ex_tokens_vector,
                q: ans_tokens_vector,
                a: y_ohe_train,
                n_batch: len(ans_ex_tokens_vector)
            })
            val_acc = acc.eval(session=self.sess, feed_dict={
                x: ans_ex_tokens_vector,
                q: ans_tokens_vector,
                a: y_ohe_train,
                n_batch: len(ans_ex_tokens_vector)
            })

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            print('epoch:', epoch,
                  ' validation loss:', val_loss,
                  ' validation accuracy:', val_acc)
            
            # Modified by mack.sano@gmail.com 3/14/2020
            file_writer.add_summary(summary, epoch)  # TensorBoard
            saver.save(self.sess, LOG_DIR+'/my_model.ckpt', epoch)

        # Modified by mack.sano@gmail.com 3/20/2020
        file_writer.flush()
        
        self.x = x
        self.q = q
        self.a = a
        self.n_batch = n_batch

        self.perform_prediction(df_ac_modeling_data, number_class)
        if csv_dump == True:
            self.df_ac_classified.to_csv(self.data_dir + key_word + r'-Classified-Model' + r'.csv', encoding= 'latin1')

    def perform_prediction(self, df_ac_prediction_data, number_class):
        self.df_ac_predict_target = df_ac_prediction_data.loc[:,[self.dependent_var]]

        y_train = self.df_ac_predict_target.transpose().values[0]
        y_matrix_train = y_train.reshape(len(y_train),1)
        ohe = OneHotEncoder(categorical_features=[0])
        y_ohe_train = ohe.fit_transform(y_matrix_train).toarray()
        ans_tokens_vector = self.vectorize_tokens(list(df_ac_prediction_data['Anser_Tokens']),
                                                  self.ans_maxlen)
        ans_ex_tokens_vector = self.vectorize_tokens(list(df_ac_prediction_data['Anser_example_Tokens']),
                                                  self.ans_ex_maxlen)

        predictions = self.sess.run(self.y, feed_dict={
            self.x: ans_ex_tokens_vector,
            self.q: ans_tokens_vector,
            self.a: y_ohe_train,
            self.n_batch: len(ans_ex_tokens_vector)
        })
        
        self.predict_res = np.zeros(len(predictions), dtype=np.int)
        i = 0
        for i in range(len(predictions)):
            self.predict_res[i] =  np.argmax(predictions[i])
            i += 1

        self.df_ac_classified = pd.DataFrame(np.array(self.predict_res,
                        dtype=np.int64), df_ac_prediction_data.index,
                        [r'Score_Class'])
        self.df_ac_classified[self.dependent_var] = self.df_ac_predict_target[self.dependent_var]
    
    def modeling_prediction_evaluation_all(self, key_word = r'', csv_dump = False, number_class = 3,
                                           epochs = 200, batch_size = 100, embedding_dim = 64): # Modified by mack.sano@gmail.com 3/21/2020)
        self.df_ac_predict_target_all = pd.DataFrame()
        self.predict_res_all = np.array([], np.int64)
        self.df_ac_classified_all = pd.DataFrame()
                
        for x in range(len(self.random_order_set)):            
            print(r'----------------')
            print(r'RANDOM SET: ', x)
            self.iloc_concat_for_cross_validation(x)
            #Modified by Sano.Makoto@otsuka.jp 4/21/2019
            self.perform_modeling(self.df_ac_modeling_values.iloc[self.concatenated_value_order, :],
                                  key_word, csv_dump, number_class, epochs, batch_size, embedding_dim)
            self.perform_prediction(self.df_ac_modeling_values.iloc[self.random_order_set[x], :], number_class)
            self.evaluate_prediction(key_word)
            if len(self.df_ac_predict_target_all) == 0:
                self.df_ac_predict_target_all = self.df_ac_predict_target.copy()
            else:
                self.df_ac_predict_target_all = self.df_ac_predict_target_all.append(self.df_ac_predict_target)
            self.predict_res_all = np.append(self.predict_res_all, self.predict_res)
            if len(self.df_ac_classified_all) == 0:
                self.df_ac_classified_all = self.df_ac_classified.copy()
            else:
                self.df_ac_classified_all = self.df_ac_classified_all.append(self.df_ac_classified)
        
        print(r'----------------')
        print(r'ALL DATA:')
        self.evaluate_prediction(key_word, csv_dump = True,
                df_ac_predict_target = self.df_ac_predict_target_all, predict_res = self.predict_res_all)

if __name__ == "__main__":
    number_data_set = 4
    csv_dump = True
    #epochs = 200
    epochs = 20
    batch_size = 100

    dependent_var = r'Definition-Score'
    task_word = r'Definition'
    number_class = 3
    
    memnd = tmv_tf_memn_classify(r'../data/')
    memnd.load_data(r'Serialized-Def-ELVA.PILOT.PRE-TEST.csv', dependent_var, [0, 1], task_word)
    memnd.iloc_split_for_cross_validation(number_data_set = number_data_set)

    memnd.modeling_prediction_evaluation_all(r'TF_MEMN-Def-PRE-All', csv_dump, number_class, epochs,
                                             batch_size)
    memnd.df_ac_classified_all.to_csv(r'../data/' + 'TF_MEMN-Classified-Prediction-Def-PRE-All.csv',
                                      encoding= 'latin1')

    '''
    dependent_var = r'Sentence-Score'
    task_word = r'Sentence'
    number_class = 4

    memns = tmv_tf_memn_classify(r'../data/')
    memns.load_data(r'Refmt_Serialized-Sen-ELVA.PILOT.PRE-TEST.csv', dependent_var, [0, 1], task_word, r'Example')
    memns.iloc_split_for_cross_validation(number_data_set = number_data_set)

    memns.modeling_prediction_evaluation_all(r'TF_MEMN-Sen-PRE-All', csv_dump, number_class, epochs,
                                             batch_size)
    memns.df_ac_classified_all.to_csv(r'../data/' + 'TF_MEMN-Classified-Prediction-Sen-PRE-All.csv',
                                      encoding= 'latin1')
    '''
