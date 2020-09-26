#!/usr/bin/env python

import pandas as pd
import numpy as np
import ml_metrics as metrics
import sklearn.metrics as mtrx
from sklearn import svm, model_selection
import pca as apca

class tmv_svm_classify(apca.tmv_pca):
    def perform_modeling(self, df_ac_modeling_data, key_word = r'', csv_dump = False):
        df_ac_modeling_target = df_ac_modeling_data.loc[:,[self.dependent_var]]
        df_ac_modeling_data = df_ac_modeling_data.drop([self.dependent_var], axis=1)
        self.ac_modeling_data_mean = df_ac_modeling_data.mean(0)
        self.ac_modeling_data_std = df_ac_modeling_data.std(0)
        df_ac_data_normalize = (df_ac_modeling_data - self.ac_modeling_data_mean).div(self.ac_modeling_data_std, axis=1)
                
        mean_max = 0.0
        self.c_mean_max = 0.0
        for C in np.logspace(-4, 10, 15):
            clf = svm.LinearSVC(C=C)
            scores = model_selection.cross_val_score(clf, df_ac_data_normalize.values,
                                        df_ac_modeling_target.transpose().values[0], cv=5)
            mean_scores = np.mean(scores)
            print("C:",C)
            print("score(mean):", mean_scores)
            if mean_scores >= mean_max:
                mean_max = mean_scores
                self.c_mean_max = C

        self.clf = svm.LinearSVC(C = self.c_mean_max)
        print(self.clf.fit(df_ac_data_normalize.values, df_ac_modeling_target.transpose().values[0]))
                
        predict_res = self.clf.predict(df_ac_data_normalize.values)
        df_ac_classified = pd.DataFrame(np.array(predict_res,
                        dtype=np.int64), df_ac_modeling_data.index,
                        [r'Score_Class'])

        df_ac_classified[self.dependent_var] = df_ac_modeling_target[self.dependent_var]
        if csv_dump == True:
            df_ac_classified.to_csv(self.data_dir + r'SVM-Classified-Model-' + key_word + r'.csv', encoding= 'latin1')

    def perform_prediction(self, df_ac_prediction_data, use_model_mean_std = False):
        self.df_ac_predict_target = df_ac_prediction_data.loc[:,[self.dependent_var]]
        df_ac_prediction_data = df_ac_prediction_data.drop([self.dependent_var], axis=1)

        if use_model_mean_std == False:
            df_ac_data_normalize = (df_ac_prediction_data - df_ac_prediction_data.mean(0)).div(df_ac_prediction_data.std(0), axis=1)
        else:
            df_ac_data_normalize = (df_ac_prediction_data - self.ac_modeling_data_mean).div(self.ac_modeling_data_std, axis=1)

        self.predict_res = self.clf.predict(df_ac_data_normalize.values)
        self.df_ac_classified = pd.DataFrame(np.array(self.predict_res,
                        dtype=np.int64), df_ac_prediction_data.index,
                        [r'Score_Class'])
        self.df_ac_classified[self.dependent_var] = self.df_ac_predict_target[self.dependent_var]

    def evaluate_prediction(self, key_word = r'', decimal_places = 4, csv_dump = False,
                            df_ac_predict_target = pd.DataFrame(), predict_res = np.array([])):
        if len(df_ac_predict_target) == 0:
            df_ac_predict_target = self.df_ac_predict_target
        if len(predict_res) == 0:
            predict_res = self.predict_res

        recall = round(mtrx.recall_score(df_ac_predict_target.transpose().values[0], 
                                              predict_res, average='weighted'), decimal_places)
        precision = round(mtrx.precision_score(df_ac_predict_target.transpose().values[0], 
                                                    predict_res, average='weighted'), decimal_places)
        f1 = round(mtrx.f1_score(df_ac_predict_target.transpose().values[0], predict_res,
                                      average='weighted'), decimal_places)
        kappa = round(metrics.kappa(predict_res, df_ac_predict_target.transpose().values[0]),
                           decimal_places)
        qwk = round(metrics.quadratic_weighted_kappa(predict_res,
                                    df_ac_predict_target.transpose().values[0]), decimal_places)

        self.se_indices = pd.Series([recall, precision, f1, kappa, qwk],
                            index=['Recall', 'Precision', 'F1', 'Kappa', 'Quadratic Weighted Kappa'])
        print(self.se_indices)
        self.conf_mtx = pd.DataFrame(mtrx.confusion_matrix(df_ac_predict_target.transpose().values[0], predict_res))
        print('Confusion Matrix:')
        print(self.conf_mtx)
        
        if csv_dump == True:
            self.se_indices.to_csv(self.data_dir + r'Classified-Prediction-Indices-' + key_word + r'.csv', encoding= 'latin1')
            self.conf_mtx.to_csv(self.data_dir + r'Classified-Prediction-Confusion-Matrix-' + key_word + r'.csv', encoding= 'latin1')

    # Modified by Makoto.Sano@Mack-the-Psych.com on 09/26/2020
    def modeling_prediction_evaluation_all(self, key_word = r'', csv_dump = False):
        self.df_ac_predict_target_all = pd.DataFrame()
        self.predict_res_all = np.array([], np.int64)
        self.df_ac_classified_all = pd.DataFrame()
                
        for x in range(len(self.random_order_set)):
            print(r'----------------')
            print(r'RANDOM SET: ', x)
            self.iloc_concat_for_cross_validation(x)
            self.perform_modeling(self.df_ac_modeling_values.iloc[self.concatenated_value_order, :],
                                  key_word)
            self.perform_prediction(self.df_ac_modeling_values.iloc[self.random_order_set[x], :])
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
    dependent_var = r'Definition-Score'
    drop_vars = None
    number_data_set = 4
    #omit_set_index = 0

    svmpd = tmv_svm_classify(r'Independent_Variable_w_Label-Def.csv', r'../data/')
    svmpd.load_data(r'Key-Stem-Passage-Aggregate_plim-Def-PRE.csv', True, drop_vars, dependent_var)
    svmpd.iloc_split_for_cross_validation(number_data_set = number_data_set)

    svmpd.modeling_prediction_evaluation_all(r'SVM-Def-PRE-All')
    svmpd.df_ac_classified_all.to_csv(r'../data/' + 'SVM-Classified-Prediction-Def-PRE-All.csv',
                                      encoding= 'latin1')

    '''
    dependent_var = r'Sentence-Score'
    drop_vars = [r'Count_Match_w_Synset_Question', r'Count_Match_w_Hyper_Question',
                  r'Count_Match_w_Hypo_Question', r'Question_Count_Match_w_Hyper_Answer']
    number_data_set = 4
    #omit_set_index = 2

    svmps = tmv_svm_classify(r'Independent_Variable_w_Label-Sen.csv', r'../data/')
    svmps.load_data(r'Key-Stem-Passage-Aggregate_plim-Sen-PRE.csv', True, drop_vars, dependent_var)
    svmps.iloc_split_for_cross_validation(number_data_set = number_data_set)

    svmps.modeling_prediction_evaluation_all(r'SVM-Sen-PRE-All')
    svmps.df_ac_classified_all.to_csv(r'../data/' + 'SVM-Classified-Prediction-Sen-PRE-All.csv',
                                      encoding= 'latin1')
    '''
