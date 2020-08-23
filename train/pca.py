#!/usr/bin/env python

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import ac_pca as apca

'''
Put a path file like "plimac-custom.pth" into any of your sys.path directories
(e.g. C:/ProgramData/Anaconda3/Lib/site-packages).

# plimac-custom.pth ###############################

# .pth file for the PLIMAC extension

C:/Users/macks/Documents/Research/ContentTextAnalysis/plimac/3.00/Lib
C:/Users/macks/Documents/Research/ContentTextAnalysis/plimac/3.00/Tools

###################################################
'''

class tmv_pca:
    def __init__(self, csv_file_in, data_dir=r'./'):
        self.data_dir = data_dir
        self.df_ind_vars = pd.read_csv(self.data_dir + csv_file_in, encoding= 'latin1')

    def load_data(self, csv_file_kspa, labeled = True, drop_ind_vars = None, dependent_var = None):
        self.dependent_var = dependent_var
        self.df_ac_aggregate_item_level = pd.read_csv(self.data_dir + csv_file_kspa, encoding= 'latin1')
        self.df_ac_aggregate_item_level = self.df_ac_aggregate_item_level.set_index('AC_Doc_ID')
        self.df_ac_modeling_values = self.df_ac_aggregate_item_level.loc[:,
                                                        list(self.df_ind_vars['Variables'])]
        if labeled == True:
            for x in self.df_ac_modeling_values.columns:
                if x in self.df_ind_vars['Variables'].values:
                    self.df_ac_modeling_values = self.df_ac_modeling_values.rename(
                        columns={x : self.df_ind_vars[self.df_ind_vars['Variables'].isin([x])]['Label'].values[0]})

        if drop_ind_vars != None:
            self.df_ac_modeling_values = self.df_ac_modeling_values.drop(drop_ind_vars, axis=1)
                    
        if self.dependent_var != None:
            self.df_ac_modeling_values[self.dependent_var] = \
                    self.df_ac_aggregate_item_level[self.dependent_var]

    def iloc_split_for_cross_validation(self, seed_num = 0, number_data_set = 4):
        np.random.seed(seed_num)
        data_len = len(self.df_ac_modeling_values)
        self.random_order_indices = np.random.permutation(data_len)

        self.random_order_set = []
        unit_num = int(data_len / number_data_set)
        for i in range(number_data_set):
            if i + 1 != number_data_set:
                self.random_order_set = self.random_order_set + \
                                        [self.random_order_indices[i * unit_num : (i + 1) * unit_num]]
            else:
                self.random_order_set = self.random_order_set + \
                                        [self.random_order_indices[i * unit_num : data_len]]

    def iloc_concat_for_cross_validation(self, number_omit_set = None):
        data_set_indices = list(range(len(self.random_order_set)))

        if number_omit_set != None:
            del data_set_indices[number_omit_set]

        self.concatenated_value_order = []
        for x in data_set_indices:
            self.concatenated_value_order = self.concatenated_value_order + list(self.random_order_set[x])
        print(r'Concat Value Len: ' + str(len(self.concatenated_value_order)))
                    
    def perform_modeling(self, df_ac_modeling_data, key_word = r''):
        self.df_ac_pca = apca.ac_pca(df_ac_modeling_data)

        print(self.df_ac_pca.loc[u'EIGEN_VALUES'])
        '''
        ev = self.df_ac_pca.loc[u'EIGEN_VALUES']
        plt.plot(self.df_ac_pca.columns, ev, label = r'Eigenvalue', color = 'k', marker='o')
        plt.xlim([0, len(ev)])
        plt.xlabel(r'Component Number')
        plt.ylabel(r'Eigenvalue')
        plt.savefig(self.data_dir + r'PCA-scree-plot-'+ key_word + r'.tif', dpi=300)
        plt.show()
        '''

if __name__ == "__main__":
    #drop_vars = [r'POS_sum', r'PMI_Bigram_Mean']
    #dependent_var = r'Definition-Score'
    drop_vars = None
    dependent_var = None
    number_data_set = 10
    omit_set_index = 5
    
    pcapd = tmv_pca(r'Independent_Variable_w_Label-Def.csv', r'../data/')
    pcapd.load_data(r'Key-Stem-Passage-Aggregate_plim-Def-PRE.csv', True, drop_vars, dependent_var)
    pcapd.perform_modeling(pcapd.df_ac_modeling_values, r'Def-PRE')
    pcapd.df_ac_pca.to_csv(r'../data/' + 'PCA-Def-PRE.csv', encoding= 'latin1')

    '''
    pcaps = tmv_pca(r'Independent_Variable_w_Label-Sen.csv', r'../data/')
    pcaps.load_data(r'Key-Stem-Passage-Aggregate_plim-Sen-PRE.csv', True, drop_vars, dependent_var)
    pcaps.perform_modeling(pcaps.df_ac_modeling_values, r'Sen-PRE')
    pcaps.df_ac_pca.to_csv(r'../data/' + 'PCA-Sen-PRE.csv', encoding= 'latin1')    
    '''
