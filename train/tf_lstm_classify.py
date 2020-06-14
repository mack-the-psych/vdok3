#!/usr/bin/env python
#Reference: https://www.amazon.co.jp/product-reviews/4839962510/
#           https://arxiv.org/pdf/1503.08895.pdf

import pandas as pd
import numpy as np
import ml_metrics as metrics
import nltk
import nltk.data
import sklearn.metrics as mtrx
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

np.random.seed(0)
tf.set_random_seed(1234)

import tf_memn_classify as memn

class tmv_tf_lstm_classify(memn.tmv_tf_memn_classify):
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
        
        xq = tf.concat([x, q], axis=-1)
        D = weight_variable([vocab_size, embedding_dim], name='D')
        d = tf.nn.embedding_lookup(D, xq)
        
        cell = tf.contrib.rnn.BasicLSTMCell(embedding_dim//2, forget_bias=1.0)
        initial_state = cell.zero_state(n_batch, tf.float32)
        state = initial_state
        outputs = []
        with tf.variable_scope('LSTM'):
            for t in range(self.ans_ex_maxlen + self.ans_maxlen):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(d[:, t, :], state)
                outputs.append(cell_output)
        output = outputs[-1]

        # Modified by mack.sano@gmail.com 3/20/2020
        W = weight_variable([embedding_dim//2, number_class], stddev=0.01, name='W')
        b = bias_variable([number_class], name='b')

        a = tf.nn.softmax(tf.matmul(output, W) + b)

        return a

if __name__ == "__main__":
    number_data_set = 4
    csv_dump = True
    #epochs = 200
    epochs = 20
    batch_size = 100

    dependent_var = r'Definition-Score'
    task_word = r'Definition'
    number_class = 3
    
    lstmd = tmv_tf_lstm_classify(r'../data/')
    lstmd.load_data(r'Serialized-Def-ELVA.PILOT.PRE-TEST.csv', dependent_var, [0, 1], task_word)
    lstmd.iloc_split_for_cross_validation(number_data_set = number_data_set)

    lstmd.modeling_prediction_evaluation_all(r'TF_LSTM-Def-PRE-All', csv_dump, number_class, epochs,
                                             batch_size)
    lstmd.df_ac_classified_all.to_csv(r'../data/' + 'TF_LSTM-Classified-Prediction-Def-PRE-All.csv',
                                      encoding= 'latin1')

    '''
    dependent_var = r'Sentence-Score'
    task_word = r'Sentence'
    number_class = 4

    lstms = tmv_tf_lstm_classify(r'../data/')
    lstms.load_data(r'Refmt_Serialized-Sen-ELVA.PILOT.PRE-TEST.csv', dependent_var, [0, 1], task_word, r'Example')
    lstms.iloc_split_for_cross_validation(number_data_set = number_data_set)

    lstms.modeling_prediction_evaluation_all(r'TF_LSTM-Sen-PRE-All', csv_dump, number_class, epochs,
                                             batch_size)
    lstms.df_ac_classified_all.to_csv(r'../data/' + 'TF_LSTM-Classified-Prediction-Sen-PRE-All.csv',
                                      encoding= 'latin1')
    '''
