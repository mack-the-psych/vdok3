#!/usr/bin/env python

import pandas as pd
import basic_nlp as fexb
import ac_bi_trigram as btgm

'''
Put a path file like "plimac-custom.pth" into any of your sys.path directories
(e.g. C:/ProgramData/Anaconda3/Lib/site-packages).

# plimac-custom.pth ###############################

# .pth file for the PLIMAC extension

C:/Users/macks/Documents/Research/ContentTextAnalysis/plimac/3.00/Lib
C:/Users/macks/Documents/Research/ContentTextAnalysis/plimac/3.00/Tools

###################################################
'''

class bi_trigram(fexb.fex_basic_nlp):
    def nlp_run(self, gram = r'bigram'):
        df_ac_gram = btgm.ac_bi_trigram(self.df_ac_in, self.cntnt_clm, gram)
        if gram == r'bigram':
            self.df_ac_bigram = df_ac_gram.copy()
        else:
            self.df_ac_trigram = df_ac_gram.copy()

if __name__ == "__main__":
    btgqd = bi_trigram(r'EN-QA-Serialized-Def-ELVA.PILOT.PRE-TEST.csv', r'../data/')
    btgqd.nlp_run(r'bigram')
    btgqd.df_ac_bigram.to_csv(r'../data/' + r'EN-Bigram-Question-Def-PRE.csv', encoding= 'latin1')
    btgqd.nlp_run(r'trigram')
    btgqd.df_ac_trigram.to_csv(r'../data/' + r'EN-Trigram-Question-Def-PRE.csv', encoding= 'latin1')

    '''
    btgqs = bi_trigram(r'EN-QA-Serialized-Sen-ELVA.PILOT.PRE-TEST.csv', r'../data/')
    btgqs.nlp_run(r'bigram')
    btgqs.df_ac_bigram.to_csv(r'../data/' + r'EN-Bigram-Question-Sen-PRE.csv', encoding= 'latin1')
    btgqs.nlp_run(r'trigram')
    btgqs.df_ac_trigram.to_csv(r'../data/' + r'EN-Trigram-Question-Sen-PRE.csv', encoding= 'latin1')
    '''
