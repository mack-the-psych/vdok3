#!/usr/bin/env python

import pandas as pd
import ac_overlapping_lemma as olle
import ac_overlapping_synset_lemma as olsy

'''
Put a path file like "plimac-custom.pth" into any of your sys.path directories
(e.g. C:/ProgramData/Anaconda3/Lib/site-packages).

# plimac-custom.pth ###############################

# .pth file for the PLIMAC extension

C:/Users/macks/Documents/Research/ContentTextAnalysis/plimac/3.00/Lib
C:/Users/macks/Documents/Research/ContentTextAnalysis/plimac/3.00/Tools

###################################################
'''

class odi_overlapping:
    def __init__(self, csv_file_in_q, csv_file_in_p = None, data_dir=r'./', stop_words = None):
        self.data_dir = data_dir
        self.df_ac_in_q = pd.read_csv(self.data_dir + csv_file_in_q, encoding= 'latin1')
        self.num_clm_in_q = len(self.df_ac_in_q.columns)
        if csv_file_in_p != None:
            self.df_ac_in_p = pd.read_csv(self.data_dir + csv_file_in_p, encoding= 'latin1')
            self.num_clm_in_p = len(self.df_ac_in_p.columns)
        else:
            self.df_ac_in_p = None
            self.num_clm_in_p = 0 #Dummy
        self.stop_words = stop_words

    def count_overlapping(self, csv_file_ac_q, question_id_clm, stem_option_name_clm = r'Pre_Col_Name', 
                         passage_name_clm_q = None, passage_sec_clm_q = None,
                         csv_file_ac_p = None, passage_name_clm_p = None,
                         passage_sec_clm_p = None):
        self.question_id_clm = question_id_clm
        self.stem_option_name_clm = stem_option_name_clm
        self.passage_name_clm_p = passage_name_clm_p
        self.passage_sec_clm_p = passage_sec_clm_p
        self.passage_name_clm_q = passage_name_clm_q
        self.passage_sec_clm_q = passage_sec_clm_q
        
        self.df_ac_lemma_q = pd.read_csv(self.data_dir + csv_file_ac_q, encoding= 'latin1')
        self.df_ac_lemma_q = self.df_ac_lemma_q.set_index('AC_Doc_ID')

        if csv_file_ac_p != None:
            self.df_ac_lemma_p = pd.read_csv(self.data_dir + csv_file_ac_p, encoding= 'latin1')
            self.df_ac_lemma_p = self.df_ac_lemma_p.set_index('AC_Doc_ID')
        else: self.df_ac_lemma_p = None

        self.df_ac_overlapping_lemma = olle.ac_overlapping_lemma(self.df_ac_lemma_q, self.question_id_clm, 
                                self.stem_option_name_clm, self.num_clm_in_q + 1, self.stop_words,
                                self.passage_name_clm_q, self.passage_sec_clm_q, self.df_ac_lemma_p,
                                self.passage_name_clm_p, self.passage_sec_clm_p, self.num_clm_in_p + 1)

    def count_overlapping_synset(self, csv_file_ac_q, stop_words = None):
        df_ac_synset_q = pd.read_csv(self.data_dir + csv_file_ac_q, encoding= 'latin1')
        df_ac_synset_q = df_ac_synset_q.set_index('AC_Doc_ID')

        if stop_words != None:
            stop_words_syn = stop_words
        else:
            stop_words_syn = self.stop_words

        self.df_ac_overlapping_syn_lemma = olsy.ac_overlapping_synset_lemma(self.df_ac_lemma_q,
                                self.question_id_clm, self.stem_option_name_clm, self.num_clm_in_q + 1, 
                                df_ac_synset_q, self.num_clm_in_q + 1, stop_words_syn,
                                self.passage_name_clm_q, self.passage_sec_clm_q, self.df_ac_lemma_p,
                                self.passage_name_clm_p, self.passage_sec_clm_p, self.num_clm_in_p + 1)

    def count_overlapping_hypernyms(self, csv_file_ac_q, stop_words = None):
        df_ac_hypernyms_q = pd.read_csv(self.data_dir + csv_file_ac_q, encoding= 'latin1')
        df_ac_hypernyms_q = df_ac_hypernyms_q.set_index('AC_Doc_ID')

        if stop_words != None:
            stop_words_hy = stop_words
        else:
            stop_words_hy = self.stop_words
            
        self.df_ac_overlapping_hyper_lemma = olsy.ac_overlapping_synset_lemma(self.df_ac_lemma_q,
                                self.question_id_clm, self.stem_option_name_clm, self.num_clm_in_q + 1, 
                                df_ac_hypernyms_q, self.num_clm_in_q + 1, stop_words_hy,
                                self.passage_name_clm_q, self.passage_sec_clm_q, self.df_ac_lemma_p,
                                self.passage_name_clm_p, self.passage_sec_clm_p, self.num_clm_in_p + 1)
                                        
        column_list = []
        for x in self.df_ac_overlapping_hyper_lemma.columns:
            column_list = column_list + [x.replace('_s_', '_hype_')]
        self.df_ac_overlapping_hyper_lemma.columns = column_list

    def count_overlapping_hyponyms(self, csv_file_ac_q, stop_words = None):
        df_ac_hyponyms_q = pd.read_csv(self.data_dir + csv_file_ac_q, encoding= 'latin1')        
        df_ac_hyponyms_q = df_ac_hyponyms_q.set_index('AC_Doc_ID')

        if stop_words != None:
            stop_words_hy = stop_words
        else:
            stop_words_hy = self.stop_words
            
        self.df_ac_overlapping_hypo_lemma = olsy.ac_overlapping_synset_lemma(self.df_ac_lemma_q,
                                self.question_id_clm, self.stem_option_name_clm, self.num_clm_in_q + 1, 
                                df_ac_hyponyms_q, self.num_clm_in_q + 1, stop_words_hy,
                                self.passage_name_clm_q, self.passage_sec_clm_q, self.df_ac_lemma_p,
                                self.passage_name_clm_p, self.passage_sec_clm_p, self.num_clm_in_p + 1)
                                        
        column_list = []
        for x in self.df_ac_overlapping_hypo_lemma.columns:
            column_list = column_list + [x.replace('_s_', '_hypo_')]
        self.df_ac_overlapping_hypo_lemma.columns = column_list
        
if __name__ == "__main__":
    stop_words_d = [r'a', r'be', r'to', r'and', r'or']
    stop_words_hy = [r'be']

    ovlpd = odi_overlapping(r'EN-QA-Serialized-Def-ELVA.PILOT.PRE-TEST.csv',
                            r'Questin_ID_Definition.csv', r'../data/', stop_words_d)
    ovlpd.count_overlapping(r'EN-Lemma-Question-Def-PRE.csv', r'Student_Question_Index',
                         r'Pre_Col_Name', r'Question_ID', r'Question_ID_Sec',
                         r'EN-Lemma-Passage-Def-PRE.csv', r'Question_ID', r'Question_ID_Sec')
    ovlpd.df_ac_overlapping_lemma.to_csv(r'../data/' + 'Overlapping-Lemma-Def-PRE.csv',
                                         encoding= 'latin1')
    
    ovlpd.count_overlapping_synset(r'EN-Synset-Question-Def-PRE.csv')
    ovlpd.df_ac_overlapping_syn_lemma.to_csv(r'../data/' + 'Overlapping-Synset-Def-PRE.csv',
                                             encoding= 'latin1')    
    ovlpd.count_overlapping_hypernyms(r'EN-Hypernyms-Question-Def-PRE.csv', stop_words_hy)
    ovlpd.df_ac_overlapping_hyper_lemma.to_csv(r'../data/' + 'Overlapping-Hypernyms-Def-PRE.csv',
                                               encoding= 'latin1')
    ovlpd.count_overlapping_hyponyms(r'EN-Hyponyms-Question-Def-PRE.csv', stop_words_hy)
    ovlpd.df_ac_overlapping_hypo_lemma.to_csv(r'../data/' + 'Overlapping-Hyponyms-Def-PRE.csv',
                                              encoding= 'latin1')

    '''
    stop_words_s = [r'a', r'be', r'to', r'and']
    ovlps = odi_overlapping(r'EN-QA-Serialized-Sen-ELVA.PILOT.PRE-TEST.csv',
                            r'Questin_ID_Definition.csv', r'../data/', stop_words_s)
    ovlps.count_overlapping(r'EN-Lemma-Question-Sen-PRE.csv', r'Student_Question_Index',
                         r'Pre_Col_Name', r'Question_ID', r'Question_ID_Sec',
                         r'EN-Lemma-Passage-Sen-PRE.csv', r'Question_ID', r'Question_ID_Sec')
    
    ovlps.df_ac_overlapping_lemma.to_csv(r'../data/' + 'Overlapping-Lemma-Sen-PRE.csv',
                                         encoding= 'latin1')
    ovlps.count_overlapping_synset(r'EN-Synset-Question-Sen-PRE.csv')
    ovlps.df_ac_overlapping_syn_lemma.to_csv(r'../data/' + 'Overlapping-Synset-Sen-PRE.csv',
                                             encoding= 'latin1')
    
    ovlps.count_overlapping_hypernyms(r'EN-Hypernyms-Question-Sen-PRE.csv', stop_words_hy)
    ovlps.df_ac_overlapping_hyper_lemma.to_csv(r'../data/' + 'Overlapping-Hypernyms-Sen-PRE.csv',
                                               encoding= 'latin1')

    ovlps.count_overlapping_hyponyms(r'EN-Hyponyms-Question-Sen-PRE.csv', stop_words_hy)
    ovlps.df_ac_overlapping_hypo_lemma.to_csv(r'../data/' + 'Overlapping-Hyponyms-Sen-PRE.csv',
                                              encoding= 'latin1')

    '''
