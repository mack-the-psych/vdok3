#!/usr/bin/env python

import pandas as pd
import ac_term_proportion as tmpr
import ac_bi_trigram_pmi as gpmi
import ac_bi_trigram_pmi_distribution as gpmd

'''
Put a path file like "plimac-custom.pth" into any of your sys.path directories
(e.g. C:/ProgramData/Anaconda3/Lib/site-packages).

# plimac-custom.pth ###############################

# .pth file for the PLIMAC extension

C:/Users/macks/Documents/Research/ContentTextAnalysis/plimac/3.00/Lib
C:/Users/macks/Documents/Research/ContentTextAnalysis/plimac/3.00/Tools

###################################################
'''

class odi_pmi:
    def __init__(self, csv_file_in_q, data_dir=r'./'):
        self.data_dir = data_dir
        self.df_ac_in_q = pd.read_csv(self.data_dir + csv_file_in_q, encoding= 'latin1')
        self.num_clm_in_q = len(self.df_ac_in_q.columns)

    def bi_trigram_pmi(self, csv_file_ac_lemma_q, csv_file_ac_gram_q, ref_scores,
                       stem_option_name_clm = r'Pre_Col_Name', key_word = r'Definition',
                       gram = r'bigram', pmi_frq_min = 2, decimal_places = 4):
        self.stem_option_name_clm = stem_option_name_clm
        self.ref_scores = ref_scores
        self.pmi_frq_min = pmi_frq_min
        self.decimal_places = decimal_places
        self.answer_clm_name = key_word + r'-Answer'
        self.score_clm_name = key_word + r'-Score'
        self.term_proportion(csv_file_ac_lemma_q)
        self.bi_trigram_ref(csv_file_ac_gram_q)
        df_ac_sum_t_gram_q_buf = gpmi.ac_bi_trigram_pmi(self.df_ac_gram_q_buf, self.num_clm_in_q + 1,
                                        self.df_ac_sum_t_lemma_q_buf, self.lemma_sum_total, gram,
                                        self.decimal_places)

        if gram == r'bigram':
            self.df_ac_pmi_sum_t_bigram_q = df_ac_sum_t_gram_q_buf[df_ac_sum_t_gram_q_buf['Bigram_sum'] >= 1]
        else:
            self.df_ac_pmi_sum_t_trigram_q = df_ac_sum_t_gram_q_buf[df_ac_sum_t_gram_q_buf['Trigram_sum'] >= 1]
        self.pmi_distribution(gram)

    def term_proportion(self, csv_file_ac_q):
        self.df_ac_lemma_q = pd.read_csv(self.data_dir + csv_file_ac_q, encoding= 'latin1')
        self.df_ac_lemma_q = self.df_ac_lemma_q.set_index('AC_Doc_ID')
        df_ac_lemma_q_buf = self.df_ac_lemma_q[self.df_ac_lemma_q[self.stem_option_name_clm].isin([self.answer_clm_name])]
        df_ac_lemma_q_buf = df_ac_lemma_q_buf[df_ac_lemma_q_buf[self.score_clm_name].isin(self.ref_scores)]
        self.df_ac_sum_t_lemma_q_buf, self.lemma_sum_total = tmpr.ac_term_proportion(df_ac_lemma_q_buf,
                                                                self.num_clm_in_q + 1)

    def bi_trigram_ref(self, csv_file_ac_gram_q):
        self.df_ac_gram_q = pd.read_csv(self.data_dir + csv_file_ac_gram_q, encoding= 'latin1')
        self.df_ac_gram_q = self.df_ac_gram_q.set_index('AC_Doc_ID')
        df_ac_gram_q_buf = self.df_ac_gram_q[self.df_ac_gram_q[self.stem_option_name_clm].isin([self.answer_clm_name])]
        self.df_ac_gram_q_buf = df_ac_gram_q_buf[df_ac_gram_q_buf[self.score_clm_name].isin(self.ref_scores)]

    def pmi_distribution(self, gram = r'bigram'):
        #The 'Bigram/Trigram' is index (not a column) when self.df_ac_pmi_sum_t_bigram/trigram_q is created
        #so the index should be reset if it's directly used as an input of ac_bi_trigram_pmi_distribution()
        if gram == r'bigram':
            df_ac_pmi_gram = self.df_ac_pmi_sum_t_bigram_q[self.df_ac_pmi_sum_t_bigram_q['Bigram_sum'] >= self.pmi_frq_min]
            self.df_ac_pmi_dist_bigram_q = gpmd.ac_bi_trigram_pmi_distribution(self.df_ac_gram_q,
                            self.num_clm_in_q + 1, df_ac_pmi_gram.reset_index(), gram, self.decimal_places)
        else:
            df_ac_pmi_gram = self.df_ac_pmi_sum_t_trigram_q[self.df_ac_pmi_sum_t_trigram_q['Trigram_sum'] >= self.pmi_frq_min]
            self.df_ac_pmi_dist_trigram_q = gpmd.ac_bi_trigram_pmi_distribution(self.df_ac_gram_q,
                            self.num_clm_in_q + 1, df_ac_pmi_gram.reset_index(), gram, self.decimal_places)
        
if __name__ == "__main__":
    ref_scores = [1, 2, 3]
    pmipd = odi_pmi(r'EN-QA-Serialized-Def-ELVA.PILOT.PRE-TEST.csv', r'../data/')
    pmipd.bi_trigram_pmi(r'EN-Lemma-Question-Def-PRE.csv', r'EN-Bigram-Question-Def-PRE.csv',
                         ref_scores, r'Pre_Col_Name', r'Definition', r'bigram', 2)
    pmipd.df_ac_pmi_sum_t_bigram_q.to_csv(r'../data/' + 'PMI-Sum-T-Bigram-Def-PRE.csv',
                                          encoding= 'latin1')
    pmipd.df_ac_pmi_dist_bigram_q.to_csv(r'../data/' + 'PMI-Distribution-Bigram-Def-PRE.csv',
                                          encoding= 'latin1')
    pmipd.bi_trigram_pmi(r'EN-Lemma-Question-Def-PRE.csv', r'EN-Trigram-Question-Def-PRE.csv',
                         ref_scores, r'Pre_Col_Name', r'Definition', r'trigram', 2)
    pmipd.df_ac_pmi_sum_t_trigram_q.to_csv(r'../data/' + 'PMI-Sum-T-Trigram-Def-PRE.csv',
                                          encoding= 'latin1')
    pmipd.df_ac_pmi_dist_trigram_q.to_csv(r'../data/' + 'PMI-Distribution-Trigram-Def-PRE.csv',
                                          encoding= 'latin1')

    '''
    pmips = odi_pmi(r'EN-QA-Serialized-Sen-ELVA.PILOT.PRE-TEST.csv', r'../data/')
    pmips.bi_trigram_pmi(r'EN-Lemma-Question-Sen-PRE.csv', r'EN-Bigram-Question-Sen-PRE.csv',
                         ref_scores, r'Pre_Col_Name', r'Sentence', r'bigram', 2)
    pmips.df_ac_pmi_sum_t_bigram_q.to_csv(r'../data/' + 'PMI-Sum-T-Bigram-Sen-PRE.csv',
                                          encoding= 'latin1')
    pmips.df_ac_pmi_dist_bigram_q.to_csv(r'../data/' + 'PMI-Distribution-Bigram-Sen-PRE.csv',
                                          encoding= 'latin1')
    pmips.bi_trigram_pmi(r'EN-Lemma-Question-Sen-PRE.csv', r'EN-Trigram-Question-Sen-PRE.csv',
                         ref_scores, r'Pre_Col_Name', r'Sentence', r'trigram', 2)
    pmips.df_ac_pmi_sum_t_trigram_q.to_csv(r'../data/' + 'PMI-Sum-T-Trigram-Sen-PRE.csv',
                                          encoding= 'latin1')
    pmips.df_ac_pmi_dist_trigram_q.to_csv(r'../data/' + 'PMI-Distribution-Trigram-Sen-PRE.csv',
                                          encoding= 'latin1')
    '''
