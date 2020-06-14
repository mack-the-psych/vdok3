#!/usr/bin/env python
#Reference: https://ameblo.jp/cognitive-solution/entry-12296011977.html

import pandas as pd
import numpy as np
import ml_metrics as metrics
import sklearn.metrics as mtrx
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

import svm_classify as svmc

class tmv_tf_pca_classify(svmc.tmv_svm_classify):
    def perform_modeling(self, df_ac_modeling_data, key_word = r'', csv_dump = False,
                         number_class = 3, epochs = 200, number_component = 2):
        df_ac_modeling_target = df_ac_modeling_data.loc[:,[self.dependent_var]]
        df_ac_modeling_data_ind = df_ac_modeling_data.drop([self.dependent_var], axis=1)
        self.ac_modeling_data_mean = df_ac_modeling_data_ind.mean(0)
        self.ac_modeling_data_std = df_ac_modeling_data_ind.std(0)
        df_ac_data_normalize = (df_ac_modeling_data_ind - self.ac_modeling_data_mean).div(self.ac_modeling_data_std, axis=1)

        y_train = df_ac_modeling_target.transpose().values[0]
        X_train = df_ac_data_normalize.values
        self.number_component = number_component

        self.pca = PCA(n_components=self.number_component)
        #X_train_pca = self.pca.fit_transform(X_train_std)
        X_train_pca = self.pca.fit_transform(X_train)

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
            sess.run(train_step, feed_dict={x:X_train_pca, t:y_ohe_train})
            if i % 10 == 0:
                loss_val, acc_val = sess.run(
                    [loss, accuracy], feed_dict={x:X_train_pca, t:y_ohe_train})
                print('Step: %d, Loss: %f, Accuracy(train): %f' % (i, loss_val, acc_val))

        self.w0_val, self.w_val = sess.run([w0, w])

        self.perform_prediction(df_ac_modeling_data)
        if csv_dump == True:
            self.df_ac_classified.to_csv(self.data_dir + r'TF_PCA-Classified-Model-' + key_word + r'.csv', encoding= 'latin1')

    def net_input(self, X):
        return np.dot(X,self.w_val)+self.w0_val

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, X):
        p = self.softmax(self.net_input(X))
        print(p)
        predicted = np.zeros(len(X), dtype=np.int)
        i = 0
        for i in range(len(X)):
            predicted[i] =  np.argmax(p[i])
            i += 1
        return predicted

    def perform_prediction(self, df_ac_prediction_data, use_model_mean_std = False):
        self.df_ac_predict_target = df_ac_prediction_data.loc[:,[self.dependent_var]]
        df_ac_prediction_data = df_ac_prediction_data.drop([self.dependent_var], axis=1)

        if use_model_mean_std == False:
            df_ac_data_normalize = (df_ac_prediction_data - df_ac_prediction_data.mean(0)).div(df_ac_prediction_data.std(0), axis=1)
        else:
            df_ac_data_normalize = (df_ac_prediction_data - self.ac_modeling_data_mean).div(self.ac_modeling_data_std, axis=1)

        X_test = df_ac_data_normalize.values
        X_test_pca = self.pca.transform(X_test)

        self.predict_res = self.predict(X_test_pca)
        self.df_ac_classified = pd.DataFrame(np.array(self.predict_res,
                        dtype=np.int64), df_ac_prediction_data.index,
                        [r'Score_Class'])
        self.df_ac_classified[self.dependent_var] = self.df_ac_predict_target[self.dependent_var]

    def modeling_prediction_evaluation_all(self, key_word = r'', csv_dump = False, number_class = 3,
                                           epochs = 200, use_model_mean_std = False, number_component = 2):
        self.df_ac_predict_target_all = pd.DataFrame()
        self.predict_res_all = np.array([], np.int64)
        self.df_ac_classified_all = pd.DataFrame()
                
        for x in range(len(self.random_order_set)):
            print(r'----------------')
            print(r'RANDOM SET: ', x)
            self.iloc_concat_for_cross_validation(x)
            self.perform_modeling(self.df_ac_modeling_values.iloc[self.concatenated_value_order, :],
                                  key_word, csv_dump, number_class, epochs, number_component)
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
            else:
                self.df_ac_classified_all = self.df_ac_classified_all.append(self.df_ac_classified)
        
        print(r'----------------')
        print(r'ALL DATA:')
        self.evaluate_prediction(key_word, csv_dump = True,
                df_ac_predict_target = self.df_ac_predict_target_all, predict_res = self.predict_res_all)

if __name__ == "__main__":
    number_data_set = 4
    csv_dump = True
    epochs = 300
    
    dependent_var = r'Definition-Score'
    number_class = 3
    use_model_mean_std = False
    number_component = 2
    drop_vars = None
    
    tpcad = tmv_tf_pca_classify(r'Independent_Variable_w_Label-Def.csv', r'../data/')
    tpcad.load_data(r'Key-Stem-Passage-Aggregate_plim-Def-PRE.csv', True, drop_vars, dependent_var)
    tpcad.iloc_split_for_cross_validation(number_data_set = number_data_set)

    tpcad.modeling_prediction_evaluation_all(r'TF_PCA-Def-PRE-All', csv_dump, number_class, epochs,
                                             use_model_mean_std, number_component)
    tpcad.df_ac_classified_all.to_csv(r'../data/' + 'TF_PCA-Classified-Prediction-Def-PRE-All.csv',
                                      encoding= 'latin1')

    '''
    dependent_var = r'Sentence-Score'
    number_class = 4
    use_model_mean_std = True
    number_component = 23
    drop_vars = [r'Count_Match_w_Synset_Question', r'Count_Match_w_Hyper_Question',
                  r'Count_Match_w_Hypo_Question', r'Question_Count_Match_w_Hyper_Answer']

    tpcas = tmv_tf_pca_classify(r'Independent_Variable_w_Label-Sen.csv', r'../data/')
    tpcas.load_data(r'Key-Stem-Passage-Aggregate_plim-Sen-PRE.csv', True, drop_vars, dependent_var)
    tpcas.iloc_split_for_cross_validation(number_data_set = number_data_set)

    tpcas.modeling_prediction_evaluation_all(r'TF_PCA-Sen-PRE-All', csv_dump, number_class, epochs,
                                             use_model_mean_std, number_component)
    tpcas.df_ac_classified_all.to_csv(r'../data/' + 'TF_PCA-Classified-Prediction-Sen-PRE-All.csv',
                                      encoding= 'latin1')
    '''
