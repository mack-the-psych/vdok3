#!/usr/bin/env python

import pandas as pd
import ac_pos_tagger as pstg
import ac_lemmatizer as lmtz
import ac_synset as syns
import ac_hypernyms as hype
import ac_hyponyms as hypo

'''
Put a path file like "plimac-custom.pth" into any of your sys.path directories
(e.g. C:/ProgramData/Anaconda3/Lib/site-packages).

# plimac-custom.pth ###############################

# .pth file for the PLIMAC extension

C:/Users/macks/Documents/Research/ContentTextAnalysis/plimac/3.00/Lib
C:/Users/macks/Documents/Research/ContentTextAnalysis/plimac/3.00/Tools

###################################################
'''

class fex_basic_nlp:
    def __init__(self, csv_file_in, data_dir=r'./', cntnt_clm = 'Content'):
        self.data_dir = data_dir
        self.df_ac_in = pd.read_csv(self.data_dir + csv_file_in, encoding= 'latin1')
        self.num_clm_in = len(self.df_ac_in.columns)
        self.cntnt_clm = cntnt_clm
        self.df_ac_lemma = None

    def nlp_run(self, pipeline=['pos', 'lemma', 'synset', 'hype', 'hypo']):
        if 'pos' in pipeline:
            self.df_ac_pos = pstg.ac_pos_tagger(self.df_ac_in, self.cntnt_clm)
            
        if 'lemma' in pipeline:
            self.df_ac_lemma = lmtz.ac_lemmatizer(self.df_ac_in, self.cntnt_clm)
        if 'synset' in pipeline:
            if self.df_ac_lemma is  None:
                self.df_ac_lemma = lmtz.ac_lemmatizer(self.df_ac_in, self.cntnt_clm)
            self.df_ac_synset = syns.ac_synset(self.df_ac_lemma.iloc[:,0: self.num_clm_in + 1],
                                               'Cntnt_Lemma')
            self.df_ac_synset = self.df_ac_synset.drop(['Cntnt_Lemma'], axis=1)

        if 'hype' in pipeline:
            if self.df_ac_lemma is  None:
                self.df_ac_lemma = lmtz.ac_lemmatizer(self.df_ac_in, self.cntnt_clm)
            self.df_ac_hypernyms = hype.ac_hypernyms(self.df_ac_lemma.iloc[:,0: self.num_clm_in + 1],
                                                     'Cntnt_Lemma')
            self.df_ac_hypernyms = self.df_ac_hypernyms.drop(['Cntnt_Lemma'], axis=1)

        if 'hypo' in pipeline:
            if self.df_ac_lemma is  None:
                self.df_ac_lemma = lmtz.ac_lemmatizer(self.df_ac_in, self.cntnt_clm)
            self.df_ac_hyponyms = hypo.ac_hyponyms(self.df_ac_lemma.iloc[:,0: self.num_clm_in + 1],
                                                   'Cntnt_Lemma')
            self.df_ac_hyponyms = self.df_ac_hyponyms.drop(['Cntnt_Lemma'], axis=1)

if __name__ == "__main__":
    pipeline=['pos', 'lemma', 'synset', 'hype', 'hypo']

    bnlqd = fex_basic_nlp(r'EN-QA-Serialized-Def-ELVA.PILOT.PRE-TEST.csv', r'../data/')
    bnlqd.nlp_run(pipeline[0])
    bnlqd.df_ac_pos.to_csv(r'../data/' + r'EN-POS-Question-Def-PRE.csv', encoding= 'latin1')
    
    bnlqd = fex_basic_nlp(r'EN-QA-Serialized-Def-ELVA.PILOT.PRE-TEST.csv', r'../data/')
    bnlqd.nlp_run(pipeline[1])
    bnlqd.df_ac_lemma.to_csv(r'../data/' + r'EN-Lemma-Question-Def-PRE.csv', encoding= 'latin1')
    bnlqd.nlp_run(pipeline[2])
    bnlqd.df_ac_synset.to_csv(r'../data/' + r'EN-Synset-Question-Def-PRE.csv', encoding= 'latin1')

    bnlqd = fex_basic_nlp(r'EN-QA-Serialized-Def-ELVA.PILOT.PRE-TEST.csv', r'../data/')
    bnlqd.nlp_run(pipeline[3])
    bnlqd.df_ac_hypernyms.to_csv(r'../data/' + r'EN-Hypernyms-Question-Def-PRE.csv', encoding= 'latin1')

    bnlqd = fex_basic_nlp(r'EN-QA-Serialized-Def-ELVA.PILOT.PRE-TEST.csv', r'../data/')
    bnlqd.nlp_run(pipeline[4])
    bnlqd.df_ac_hyponyms.to_csv(r'../data/' + r'EN-Hyponyms-Question-Def-PRE.csv', encoding= 'latin1')

    bnlpd = fex_basic_nlp(r'Questin_ID_Definition.csv', r'../data/', r'Definition')
    bnlpd.nlp_run(pipeline[0])
    bnlpd.df_ac_pos.to_csv(r'../data/' + r'EN-POS-Passage-Def-PRE.csv', encoding= 'latin1')

    bnlpd = fex_basic_nlp(r'Questin_ID_Definition.csv', r'../data/', r'Definition')
    bnlpd.nlp_run(pipeline[1])
    bnlpd.df_ac_lemma.to_csv(r'../data/' + r'EN-Lemma-Passage-Def-PRE.csv', encoding= 'latin1')

    '''
    bnlqs = fex_basic_nlp(r'EN-QA-Serialized-Sen-ELVA.PILOT.PRE-TEST.csv', r'../data/')
    bnlqs.nlp_run()
    bnlqs.df_ac_pos.to_csv(r'../data/' + r'EN-POS-Question-Sen-PRE.csv', encoding= 'latin1')
    bnlqs.df_ac_lemma.to_csv(r'../data/' + r'EN-Lemma-Question-Sen-PRE.csv', encoding= 'latin1')
    bnlqs.df_ac_synset.to_csv(r'../data/' + r'EN-Synset-Question-Sen-PRE.csv', encoding= 'latin1')
    bnlqs.df_ac_hypernyms.to_csv(r'../data/' + r'EN-Hypernyms-Question-Sen-PRE.csv', encoding= 'latin1')
    bnlqs.df_ac_hyponyms.to_csv(r'../data/' + r'EN-Hyponyms-Question-Sen-PRE.csv', encoding= 'latin1')

    bnlps = fex_basic_nlp(r'Questin_ID_Sentence_Example.csv', r'../data/', r'Example')
    bnlps.nlp_run()
    bnlps.df_ac_pos.to_csv(r'../data/' + r'EN-POS-Passage-Sen-PRE.csv', encoding= 'latin1')
    bnlps.df_ac_lemma.to_csv(r'../data/' + r'EN-Lemma-Passage-Sen-PRE.csv', encoding= 'latin1')
    '''
