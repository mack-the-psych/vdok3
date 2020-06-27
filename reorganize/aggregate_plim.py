#!/usr/bin/env python

import pandas as pd
import numpy as np
import ac_aggregate_plim as agpl
import ac_aggregate_item_level_plim as agpi

'''
Put a path file like "plimac-custom.pth" into any of your sys.path directories
(e.g. C:/ProgramData/Anaconda3/Lib/site-packages).

# plimac-custom.pth ###############################

# .pth file for the PLIMAC extension

C:/Users/macks/Documents/Research/ContentTextAnalysis/plimac/3.00/Lib
C:/Users/macks/Documents/Research/ContentTextAnalysis/plimac/3.00/Tools

###################################################
'''

class rop_aggregate_plim:
    def __init__(self, csv_file_in_q, question_id_clm, stem_option_name_clm = r'Pre_Col_Name',
                 csv_file_in_p = None, data_dir=r'./', stop_words_pos = None,
                 passage_name_clm_q = None, passage_sec_clm_q = None,
                 passage_name_clm_p = None, passage_sec_clm_p = None):
        self.data_dir = data_dir
        self.stop_words_pos = stop_words_pos
        self.question_id_clm = question_id_clm
        self.stem_option_name_clm = stem_option_name_clm        
        self.passage_name_clm_p = passage_name_clm_p
        self.passage_sec_clm_p = passage_sec_clm_p
        self.passage_name_clm_q = passage_name_clm_q
        self.passage_sec_clm_q = passage_sec_clm_q
        
        self.df_ac_in_q = pd.read_csv(self.data_dir + csv_file_in_q, encoding= 'latin1')
        self.num_clm_in_q = len(self.df_ac_in_q.columns)
        if csv_file_in_p != None:
            self.df_ac_in_p = pd.read_csv(self.data_dir + csv_file_in_p, encoding= 'latin1')
            self.num_clm_in_p = len(self.df_ac_in_p.columns)
        else:
            self.df_ac_in_p = None
            self.num_clm_in_p = 0 #Dummy

    def load_fex_files(self, csv_file_ac_lemma_q, csv_file_ac_q, csv_file_ac_p = None):
        self.df_ac_lemma_q = pd.read_csv(self.data_dir + csv_file_ac_lemma_q, encoding= 'latin1')
        self.df_ac_lemma_q = self.df_ac_lemma_q.set_index('AC_Doc_ID')
        
        self.df_ac_pos_q = pd.read_csv(self.data_dir + csv_file_ac_q, encoding= 'latin1')
        self.df_ac_pos_q = self.df_ac_pos_q.set_index('AC_Doc_ID')

        if csv_file_ac_p != None:
            self.df_ac_pos_p = pd.read_csv(self.data_dir + csv_file_ac_p, encoding= 'latin1')
            self.df_ac_pos_p = self.df_ac_pos_p.set_index('AC_Doc_ID')
        else: self.df_ac_pos_p = None

    def load_overlapping_stat_files(self, csv_file_ac_lemma, csv_file_ac_synset, csv_file_ac_hyper,
                                    csv_file_ac_hypo):
        self.df_ac_ovl_lemma = pd.read_csv(self.data_dir + csv_file_ac_lemma, encoding= 'latin1')
        self.df_ac_ovl_lemma = self.df_ac_ovl_lemma.set_index('AC_Doc_ID')
        self.df_ac_ovl_lemma = self.df_ac_ovl_lemma.drop([self.question_id_clm, self.stem_option_name_clm], axis=1)
        self.df_ac_ovl_synset = pd.read_csv(self.data_dir + csv_file_ac_synset, encoding= 'latin1')
        self.df_ac_ovl_synset = self.df_ac_ovl_synset.set_index('AC_Doc_ID')
        self.df_ac_ovl_synset = self.df_ac_ovl_synset.drop([self.question_id_clm, self.stem_option_name_clm], axis=1)
        self.df_ac_ovl_hyper = pd.read_csv(self.data_dir + csv_file_ac_hyper, encoding= 'latin1')
        self.df_ac_ovl_hyper = self.df_ac_ovl_hyper.set_index('AC_Doc_ID')
        self.df_ac_ovl_hyper = self.df_ac_ovl_hyper.drop([self.question_id_clm, self.stem_option_name_clm], axis=1)
        self.df_ac_ovl_hypo = pd.read_csv(self.data_dir + csv_file_ac_hypo, encoding= 'latin1')
        self.df_ac_ovl_hypo = self.df_ac_ovl_hypo.set_index('AC_Doc_ID')
        self.df_ac_ovl_hypo = self.df_ac_ovl_hypo.drop([self.question_id_clm, self.stem_option_name_clm], axis=1)

    def load_distribution_stat_files(self, csv_file_ac_oanc, csv_file_ac_bigram, csv_file_ac_trigram):
        self.df_ac_oanc = pd.read_csv(self.data_dir + csv_file_ac_oanc, encoding= 'latin1')
        self.df_ac_oanc = self.df_ac_oanc.set_index('AC_Doc_ID')
        self.df_ac_oanc = self.df_ac_oanc.drop([self.question_id_clm, self.stem_option_name_clm], axis=1)

        df_ac_pmi_bigram = pd.read_csv(self.data_dir + csv_file_ac_bigram, encoding= 'latin1')
        df_ac_pmi_bigram = df_ac_pmi_bigram.set_index('AC_Doc_ID')
        df_ac_pmi_bigram = df_ac_pmi_bigram.iloc[:, self.num_clm_in_q:]
        df_ac_pmi_bigram['Cntnt_Bigram'] = df_ac_pmi_bigram['Cntnt_Bigram'].fillna('')
        df_ac_pmi_bigram['PMI_Bigram_SD'] = df_ac_pmi_bigram['PMI_Bigram_SD'].fillna(0.0)
        self.df_ac_pmi_bigram = df_ac_pmi_bigram.fillna(-10.0)

        df_ac_pmi_trigram = pd.read_csv(self.data_dir + csv_file_ac_trigram, encoding= 'latin1')
        df_ac_pmi_trigram = df_ac_pmi_trigram.set_index('AC_Doc_ID')
        df_ac_pmi_trigram = df_ac_pmi_trigram.iloc[:, self.num_clm_in_q:]
        df_ac_pmi_trigram['Cntnt_Trigram'] = df_ac_pmi_trigram['Cntnt_Trigram'].fillna('')
        df_ac_pmi_trigram['PMI_Trigram_SD'] = df_ac_pmi_trigram['PMI_Trigram_SD'].fillna(0.0)
        self.df_ac_pmi_trigram = df_ac_pmi_trigram.fillna(-10.0)

    def aggregate_plim(self, specific_count_lemmas, key_word = r'Definition', decimal_places = 4):
        self.question_clm_name = key_word + r'-Question'
        answer_clm_name = key_word + r'-Answer'

        self.df_ac_aggregate = agpl.ac_aggregate_plim(self.df_ac_pos_q, self.num_clm_in_q + 1, 
                                self.df_ac_ovl_lemma, self.df_ac_ovl_synset, 
                                None, self.df_ac_oanc, self.stem_option_name_clm, self.question_clm_name,
                                list(self.df_ac_in_q.columns), self.stop_words_pos, self.df_ac_lemma_q,
                                specific_count_lemmas, self.df_ac_pos_p, self.passage_name_clm_q,
                                self.passage_sec_clm_q, self.passage_name_clm_p, self.passage_sec_clm_p,
                                self.num_clm_in_p + 1, decimal_places,
                                self.df_ac_ovl_hyper, self.df_ac_ovl_hypo,
                                df_ac_bigram_pmi_distribution = self.df_ac_pmi_bigram, 
                                df_ac_trigram_pmi_distribution = self.df_ac_pmi_trigram)

        self.key_dummy = r'Key_Dummy'
        t = self.df_ac_aggregate.shape
        row_lgth = t[0]
        df_key_dummy = pd.DataFrame(np.empty((row_lgth, 1),
                            dtype=object), self.df_ac_aggregate.index,
                            [self.key_dummy])
        df_key_dummy = df_key_dummy.fillna(answer_clm_name)
        self.df_ac_aggregate[self.key_dummy] = df_key_dummy[self.key_dummy]

    def aggregate_item_level_plim(self, key_word = r'Definition', cntnt_clm = 'Content', 
                                  decimal_places = 4):
        self.df_ac_aggregate_item_level = agpi.ac_aggregate_item_level_plim(self.df_ac_aggregate,
                                self.key_dummy, self.stem_option_name_clm, self.question_clm_name, 
                                None, decimal_places, cntnt_clm)
        self.df_ac_aggregate_item_level_corr = self.df_ac_aggregate_item_level.corr()
        self.df_ac_aggregate_item_level_describe = self.df_ac_aggregate_item_level.describe()
        
if __name__ == "__main__":
    stop_words_pos = None
    specific_count_lemmas = [r'dk', r'nr']

    agrpd = rop_aggregate_plim(r'EN-QA-Serialized-Def-ELVA.PILOT.PRE-TEST.csv',
                               r'Student_Question_Index', r'Pre_Col_Name',
                               r'Questin_ID_Definition.csv', r'../data/', stop_words_pos,
                               r'Question_ID', r'Question_ID_Sec',
                               r'Question_ID', r'Question_ID_Sec')
    agrpd.load_fex_files(r'EN-Lemma-Question-Def-PRE.csv', r'EN-POS-Question-Def-PRE.csv',
                         r'EN-POS-Passage-Def-PRE.csv')
    agrpd.load_overlapping_stat_files(r'Overlapping-Lemma-Def-PRE.csv', r'Overlapping-Synset-Def-PRE.csv',
                         r'Overlapping-Hypernyms-Def-PRE.csv', r'Overlapping-Hyponyms-Def-PRE.csv')
    agrpd.load_distribution_stat_files(r'Lemma-OANC-Frequency-Def-PRE.csv', 
                         r'PMI-Distribution-Bigram-Def-PRE.csv', r'PMI-Distribution-Trigram-Def-PRE.csv')
    agrpd.aggregate_plim(specific_count_lemmas, r'Definition')
    agrpd.aggregate_item_level_plim(r'Definition')
    
    agrpd.df_ac_aggregate.to_csv(r'../data/' + 'Aggregate_plim-Def-PRE.csv', encoding= 'latin1')
    agrpd.df_ac_aggregate_item_level.to_csv(r'../data/' + 'Key-Stem-Passage-Aggregate_plim-Def-PRE.csv',
                                            encoding= 'latin1')
    agrpd.df_ac_aggregate_item_level_corr.to_csv(r'../data/' + 'Corr_Key-Stem-Passage-Aggregate_plim-Def-PRE.csv', encoding= 'latin1')
    agrpd.df_ac_aggregate_item_level_describe.to_csv(r'../data/' + 'Describe-Key-Stem-Passage-Aggregate_plim-Def-PRE.csv', encoding= 'latin1')

    '''
    agrps = rop_aggregate_plim(r'EN-QA-Serialized-Sen-ELVA.PILOT.PRE-TEST.csv',
                               r'Student_Question_Index', r'Pre_Col_Name',
                               r'Questin_ID_Sentence_Example.csv', r'../data/', stop_words_pos,
                               r'Question_ID', r'Question_ID_Sec',
                               r'Question_ID', r'Question_ID_Sec')
    agrps.load_fex_files(r'EN-lemma-Question-Sen-PRE.csv', r'EN-POS-Question-Sen-PRE.csv',
                         r'EN-POS-Passage-Sen-PRE.csv')
    agrps.load_overlapping_stat_files(r'Overlapping-Lemma-Sen-PRE.csv', r'Overlapping-Synset-Sen-PRE.csv',
                         r'Overlapping-Hypernyms-Sen-PRE.csv', r'Overlapping-Hyponyms-Sen-PRE.csv')
    agrps.load_distribution_stat_files(r'Lemma-OANC-Frequency-Sen-PRE.csv', 
                         r'PMI-Distribution-Bigram-Sen-PRE.csv', r'PMI-Distribution-Trigram-Sen-PRE.csv')
    agrps.aggregate_plim(specific_count_lemmas, r'Sentence')
    agrps.aggregate_item_level_plim(r'Sentence')
    
    agrps.df_ac_aggregate.to_csv(r'../data/' + 'Aggregate_plim-Sen-PRE.csv', encoding= 'latin1')
    agrps.df_ac_aggregate_item_level.to_csv(r'../data/' + 'Key-Stem-Passage-Aggregate_plim-Sen-PRE.csv',
                                            encoding= 'latin1')
    agrps.df_ac_aggregate_item_level_corr.to_csv(r'../data/' + 'Corr_Key-Stem-Passage-Aggregate_plim-Sen-PRE.csv', encoding= 'latin1')
    agrps.df_ac_aggregate_item_level_describe.to_csv(r'../data/' + 'Describe-Key-Stem-Passage-Aggregate_plim-Sen-PRE.csv', encoding= 'latin1')
    '''
