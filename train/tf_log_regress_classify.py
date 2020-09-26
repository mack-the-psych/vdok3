#!/usr/bin/env python
#Reference: https://ameblo.jp/cognitive-solution/entry-12296011977.html

import pandas as pd
import numpy as np
import ml_metrics as metrics
import sklearn.metrics as mtrx
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

import tf_pca_classify as tpca

class tmv_tf_log_regress_classify(tpca.tmv_tf_pca_classify):
    def perform_modeling(self, df_ac_modeling_data, key_word = r'', csv_dump = False,
                         number_class = 3, epochs = 200):
        df_ac_modeling_target = df_ac_modeling_data.loc[:,[self.dependent_var]]
        df_ac_modeling_data_ind = df_ac_modeling_data.drop([self.dependent_var], axis=1)
        self.ac_modeling_data_mean = df_ac_modeling_data_ind.mean(0)
        self.ac_modeling_data_std = df_ac_modeling_data_ind.std(0)
        df_ac_data_normalize = (df_ac_modeling_data_ind - self.ac_modeling_data_mean).div(self.ac_modeling_data_std, axis=1)

        y_train = df_ac_modeling_target.transpose().values[0]
        X_train = df_ac_data_normalize.values
        self.number_component = X_train.shape[1]

        x = tf.placeholder(tf.float32, [None, self.number_component])
        w = tf.Variable(tf.zeros([self.number_component, number_class]))
        w0 = tf.Variable(tf.zeros([number_class]))
        f = tf.matmul(x, w) + w0
        p = tf.nn.softmax(f)

        t = tf.placeholder(tf.float32, [None, number_class])
        loss = -tf.reduce_sum(t*tf.log(p))
        train_step = tf.train.AdamOptimizer().minimize(loss)

        correct_prediction = tf.equal(tf.argmax(p,1), tf.argmax(t,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())

        y_matrix_train = y_train.reshape(len(y_train),1)
        ohe = OneHotEncoder(categorical_features=[0])
        y_ohe_train = ohe.fit_transform(y_matrix_train).toarray()

        i = 0
        for _ in range(epochs):
            i += 1
            sess.run(train_step, feed_dict={x:X_train, t:y_ohe_train})
            if i % 10 == 0:
                loss_val, acc_val = sess.run(
                    [loss, accuracy], feed_dict={x:X_train, t:y_ohe_train})
                print('Step: %d, Loss: %f, Accuracy(train): %f' % (i, loss_val, acc_val))

        self.w0_val, self.w_val = sess.run([w0, w])

        self.perform_prediction(df_ac_modeling_data)
        if csv_dump == True:
            self.df_ac_classified.to_csv(self.data_dir + r'TF_Log_Reg-Classified-Model-' + key_word + r'.csv', encoding= 'latin1')

    def perform_prediction(self, df_ac_prediction_data, use_model_mean_std = False):
        self.df_ac_predict_target = df_ac_prediction_data.loc[:,[self.dependent_var]]
        df_ac_prediction_data = df_ac_prediction_data.drop([self.dependent_var], axis=1)

        if use_model_mean_std == False:
            df_ac_data_normalize = (df_ac_prediction_data - df_ac_prediction_data.mean(0)).div(df_ac_prediction_data.std(0), axis=1)
        else:
            df_ac_data_normalize = (df_ac_prediction_data - self.ac_modeling_data_mean).div(self.ac_modeling_data_std, axis=1)

        X_test = df_ac_data_normalize.values

        self.predict_res = self.predict(X_test)
        self.df_ac_classified = pd.DataFrame(np.array(self.predict_res,
                        dtype=np.int64), df_ac_prediction_data.index,
                        [r'Score_Class'])
        self.df_ac_classified[self.dependent_var] = self.df_ac_predict_target[self.dependent_var]

    # Modified by Makoto.Sano@Mack-the-Psych.com on 09/26/2020
    def modeling_prediction_evaluation_all(self, key_word = r'', csv_dump = False, number_class = 3,
                                           epochs = 200, use_model_mean_std = False):
        self.df_ac_predict_target_all = pd.DataFrame()
        self.predict_res_all = np.array([], np.int64)
        self.df_ac_classified_all = pd.DataFrame()
                
        for x in range(len(self.random_order_set)):
            print(r'----------------')
            print(r'RANDOM SET: ', x)
            self.iloc_concat_for_cross_validation(x)
            self.perform_modeling(self.df_ac_modeling_values.iloc[self.concatenated_value_order, :],
                                  key_word, csv_dump, number_class, epochs)
            self.perform_prediction(self.df_ac_modeling_values.iloc[self.random_order_set[x], :],
                                    use_model_mean_std)
            self.evaluate_prediction(key_word)
            if len(self.df_ac_predict_target_all) == 0:
                self.df_ac_predict_target_all = self.df_ac_predict_target.copy()
            else:
                self.df_ac_predict_target_all = self.df_ac_predict_target_all.append(self.df_ac_predict_target)
            self.predict_res_all = np.append(self.predict_res_all, self.predict_res)
            if len(self.df_ac_classified_all) == 0:
                self.df_ac_classified_all = self.df_ac_classified.copy()
                self.df_indices_all = pd.DataFrame(self.se_indices)
            else:
                self.df_ac_classified_all = self.df_ac_classified_all.append(self.df_ac_classified)
                self.df_indices_all = pd.concat([self.df_indices_all, self.se_indices], axis=1)

        self.df_indices_all = self.df_indices_all.T
        print(r'----------------')
        print(r'ALL DATA (Macro Average):')
        print(self.df_indices_all.describe())
        if csv_dump == True:
            self.df_indices_all.describe().to_csv(self.data_dir + r'Classified-Prediction-Indices-Macro-' + key_word + r'.csv', encoding= 'latin1')
        print(r'----------------')
        print(r'ALL DATA (Micro Average):')
        self.evaluate_prediction(key_word, csv_dump = True,
                df_ac_predict_target = self.df_ac_predict_target_all, predict_res = self.predict_res_all)

if __name__ == "__main__":
    number_data_set = 4
    csv_dump = True
    epochs = 800
    
    dependent_var = r'Definition-Score'
    number_class = 3
    use_model_mean_std = True
    drop_vars = None
    
    lregd = tmv_tf_log_regress_classify(r'Independent_Variable_w_Label-Def.csv', r'../data/')
    lregd.load_data(r'Key-Stem-Passage-Aggregate_plim-Def-PRE.csv', True, drop_vars, dependent_var)
    lregd.iloc_split_for_cross_validation(number_data_set = number_data_set)

    lregd.modeling_prediction_evaluation_all(r'TF_Log_Reg-Def-PRE-All', csv_dump, number_class, epochs,
                                             use_model_mean_std)
    lregd.df_ac_classified_all.to_csv(r'../data/' + 'TF_Log_Reg-Classified-Prediction-Def-PRE-All.csv',
                                      encoding= 'latin1')

    '''
    dependent_var = r'Sentence-Score'
    number_class = 4
    use_model_mean_std = True
    drop_vars = [r'Count_Match_w_Synset_Question', r'Count_Match_w_Hyper_Question',
                  r'Count_Match_w_Hypo_Question', r'Question_Count_Match_w_Hyper_Answer']

    lregs = tmv_tf_log_regress_classify(r'Independent_Variable_w_Label-Sen.csv', r'../data/')
    lregs.load_data(r'Key-Stem-Passage-Aggregate_plim-Sen-PRE.csv', True, drop_vars, dependent_var)
    lregs.iloc_split_for_cross_validation(number_data_set = number_data_set)

    lregs.modeling_prediction_evaluation_all(r'TF_Log_Reg-Sen-PRE-All', csv_dump, number_class, epochs,
                                             use_model_mean_std)
    lregs.df_ac_classified_all.to_csv(r'../data/' + 'TF_Log_Reg-Classified-Prediction-Sen-PRE-All.csv',
                                      encoding= 'latin1')
    '''
