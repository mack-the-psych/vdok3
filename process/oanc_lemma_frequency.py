#!/usr/bin/env python

import pandas as pd
import shelve
import ac_oanc_lemma_frequency as olfq

'''
Put a path file like "plimac-custom.pth" into any of your sys.path directories
(e.g. C:/ProgramData/Anaconda3/Lib/site-packages).

# plimac-custom.pth ###############################

# .pth file for the PLIMAC extension

C:/Users/macks/Documents/Research/ContentTextAnalysis/plimac/3.00/Lib
C:/Users/macks/Documents/Research/ContentTextAnalysis/plimac/3.00/Tools

###################################################
'''

class odi_oanc_lemma_frequency:
    def __init__(self, csv_file_in_q, oanc_shelve, csv_file_in_p = None, data_dir=r'./', stop_words = None):
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
        self.oanc_lemma = shelve.open(oanc_shelve, flag='r')

    def oanc_lemma_frequency(self, csv_file_ac_q, question_id_clm, stem_option_name_clm = r'Pre_Col_Name', 
                             unknown_word_len_min = 1, passage_name_clm_q = None, passage_sec_clm_q = None,
                             csv_file_ac_p = None, passage_name_clm_p = None, passage_sec_clm_p = None,
                             decimal_places = 4):
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

        self.df_ac_oanc_lemma_freq_q = olfq.ac_oanc_lemma_frequency(self.df_ac_lemma_q, self.question_id_clm, 
                                self.stem_option_name_clm, self.num_clm_in_q + 1, self.oanc_lemma,
                                self.stop_words, unknown_word_len_min,
                                self.passage_name_clm_q, self.passage_sec_clm_q, self.df_ac_lemma_p,
                                self.passage_name_clm_p, self.passage_sec_clm_p, self.num_clm_in_p + 1,
                                decimal_places)
        
if __name__ == "__main__":
    stop_words_d = [r'a', r'be', r'to', r'and', r'or']

    oanc_shelve = r'../../plimac3/Resource/OANC/ANC-all-lemma-04262014.db'
    
    oalpd = odi_oanc_lemma_frequency(r'EN-QA-Serialized-Def-ELVA.PILOT.PRE-TEST.csv',
                                     oanc_shelve, None, r'../data/', stop_words_d)    
    oalpd.oanc_lemma_frequency(r'EN-Lemma-Question-Def-PRE.csv', r'Student_Question_Index', r'Pre_Col_Name')
    oalpd.df_ac_oanc_lemma_freq_q.to_csv(r'../data/' + 'Lemma-OANC-Frequency-Def-PRE.csv',
                                         encoding= 'latin1')

    '''
    stop_words_s = [r'a', r'be', r'to', r'and', r'or']

    oalps = odi_oanc_lemma_frequency(r'EN-QA-Serialized-Sen-ELVA.PILOT.PRE-TEST.csv',
                                     oanc_shelve, None, r'../data/', stop_words_s)
    oalps.oanc_lemma_frequency(r'EN-Lemma-Question-Sen-PRE.csv', r'Student_Question_Index', r'Pre_Col_Name')
    oalps.df_ac_oanc_lemma_freq_q.to_csv(r'../data/' + 'Lemma-OANC-Frequency-Sen-PRE.csv',
                                         encoding= 'latin1')
    '''
