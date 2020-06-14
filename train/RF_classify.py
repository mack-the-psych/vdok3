#!/usr/bin/env python
#Reference: https://ohke.hateblo.jp/entry/2017/08/04/230000

import pandas as pd
import numpy as np
import svm_classify as svmc

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

class tmv_RF_classify(svmc.tmv_svm_classify):
    def perform_modeling(self, df_ac_modeling_data, key_word = r'', csv_dump = False):
        df_ac_modeling_target = df_ac_modeling_data.loc[:,[self.dependent_var]]
        df_ac_modeling_data = df_ac_modeling_data.drop([self.dependent_var], axis=1)
                
        forest_grid_param = {
            'n_estimators': [100],
            'max_features': [1, 'auto', None],
            'max_depth': [1, 5, 10, None],
            'min_samples_leaf': [1, 2, 4,]
        }

        f1_scoring = make_scorer(f1_score,  pos_label=1, average='micro')

        forest_grid_search = GridSearchCV(RandomForestClassifier(random_state=0, n_jobs=-1), forest_grid_param, scoring=f1_scoring, cv=4)
        forest_grid_search.fit(df_ac_modeling_data.values, df_ac_modeling_target.transpose().values[0])

        print('Best parameters: {}'.format(forest_grid_search.best_params_))
        print('Best score: {:.3f}'.format(forest_grid_search.best_score_))
                        
        best_params = forest_grid_search.best_params_
        self.forest = RandomForestClassifier(random_state=0, n_jobs=-1, 
                                        max_depth=best_params['max_depth'], 
                                        max_features=best_params['max_features'], 
                                        min_samples_leaf=best_params['min_samples_leaf'],
                                        n_estimators=best_params['n_estimators'])
        self.forest.fit(df_ac_modeling_data.values, df_ac_modeling_target.transpose().values[0])

        predict_res = self.forest.predict(df_ac_modeling_data.values)
        df_ac_classified = pd.DataFrame(np.array(predict_res,
                        dtype=np.int64), df_ac_modeling_data.index,
                        [r'Score_Class'])

        df_ac_classified[self.dependent_var] = df_ac_modeling_target[self.dependent_var]
        if csv_dump == True:
            df_ac_classified.to_csv(self.data_dir + r'RF-Classified-Model-' + key_word + r'.csv', encoding= 'latin1')

    def perform_prediction(self, df_ac_prediction_data):
        self.df_ac_predict_target = df_ac_prediction_data.loc[:,[self.dependent_var]]
        df_ac_prediction_data = df_ac_prediction_data.drop([self.dependent_var], axis=1)

        self.predict_res = self.forest.predict(df_ac_prediction_data.values)
        self.df_ac_classified = pd.DataFrame(np.array(self.predict_res,
                        dtype=np.int64), df_ac_prediction_data.index,
                        [r'Score_Class'])
        self.df_ac_classified[self.dependent_var] = self.df_ac_predict_target[self.dependent_var]

if __name__ == "__main__":
    dependent_var = r'Definition-Score'
    drop_vars = None
    number_data_set = 4

    rndfd = tmv_RF_classify(r'Independent_Variable_w_Label-Def.csv', r'../data/')
    rndfd.load_data(r'Key-Stem-Passage-Aggregate_plim-Def-PRE.csv', True, drop_vars, dependent_var)
    rndfd.iloc_split_for_cross_validation(number_data_set = number_data_set)

    rndfd.modeling_prediction_evaluation_all(r'RF-Def-PRE-All')
    rndfd.df_ac_classified_all.to_csv(r'../data/' + 'RF_Classified-Prediction-Def-PRE-All.csv',
                                      encoding= 'latin1')

    '''
    dependent_var = r'Sentence-Score'
    #drop_vars = [r'Count_Match_w_Synset_Question', r'Count_Match_w_Hyper_Question',
    #              r'Count_Match_w_Hypo_Question', r'Question_Count_Match_w_Hyper_Answer']
    number_data_set = 4

    rndfs = tmv_RF_classify(r'Independent_Variable_w_Label-Sen.csv', r'../data/')
    rndfs.load_data(r'Key-Stem-Passage-Aggregate_plim-Sen-PRE.csv', True, drop_vars, dependent_var)
    rndfs.iloc_split_for_cross_validation(number_data_set = number_data_set)

    rndfs.modeling_prediction_evaluation_all(r'RF-Sen-PRE-All')
    rndfs.df_ac_classified_all.to_csv(r'../data/' + 'RF-Classified-Prediction-Sen-PRE-All.csv',
                                      encoding= 'latin1')
    '''
