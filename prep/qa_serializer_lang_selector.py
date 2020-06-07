#!/usr/bin/env python

import pandas as pd
import ac_column_serializer as clsr

'''
Put a path file like "plimac-custom.pth" into any of your sys.path directories
(e.g. C:/ProgramData/Anaconda3/Lib/site-packages).

# plimac-custom.pth ###############################

# .pth file for the PLIMAC extension

C:/Users/macks/Documents/Research/ContentTextAnalysis/plimac/3.00/Lib
C:/Users/macks/Documents/Research/ContentTextAnalysis/plimac/3.00/Tools

###################################################
'''

class qa_serializer_lang_selector:
    def __init__(self, data_dir=r'./'):
        self.data_dir = data_dir

    def serialize_record(self, csv_file_in, key_word = 'Definition'):
        new_question_clm_name = key_word + r'-Question'
        new_answer_clm_name = key_word + r'-Answer'
        self.df_response_serialized = clsr.ac_column_serializer(self.data_dir + csv_file_in,
                                'Student_Question_Index', [new_question_clm_name, new_answer_clm_name])

    def select_lang(self, langs, key_word = 'Definition'):
        lang_clm = key_word + r'-Language'
        return self.df_response_serialized[self.df_response_serialized[lang_clm].isin(langs)]

if __name__ == "__main__":
    ql = qa_serializer_lang_selector(r'../data/')

    ql.serialize_record(r'Serialized-Def-ELVA.PILOT.PRE-TEST.csv', r'Definition')
    df_es = ql.select_lang([2, 3], r'Definition')
    df_es.to_csv(r'../data/' + r'ES-QA-Serialized-Def-ELVA.PILOT.PRE-TEST.csv', encoding= 'latin1')
    df_en = ql.select_lang([0, 1], r'Definition')
    df_en.to_csv(r'../data/' + r'EN-QA-Serialized-Def-ELVA.PILOT.PRE-TEST.csv', encoding= 'latin1')

    '''
    ql.serialize_record(r'Serialized-Def-ELVA.PILOT.POST-TEST.csv', r'Definition')
    df_es = ql.select_lang([2, 3], r'Definition')
    df_es.to_csv(r'../data/' + r'ES-QA-Serialized-Def-ELVA.PILOT.POST-TEST.csv', encoding= 'latin1')
    df_en = ql.select_lang([0, 1], r'Definition')
    df_en.to_csv(r'../data/' + r'EN-QA-Serialized-Def-ELVA.PILOT.POST-TEST.csv', encoding= 'latin1')

    ql.serialize_record(r'Serialized-Sen-ELVA.PILOT.PRE-TEST.csv', r'Sentence')
    df_es = ql.select_lang([2, 3], r'Sentence')
    df_es.to_csv(r'../data/' + r'ES-QA-Serialized-Sen-ELVA.PILOT.PRE-TEST.csv', encoding= 'latin1')
    df_en = ql.select_lang([0, 1], r'Sentence')
    df_en.to_csv(r'../data/' + r'EN-QA-Serialized-Sen-ELVA.PILOT.PRE-TEST.csv', encoding= 'latin1')

    ql.serialize_record(r'Serialized-Sen-ELVA.PILOT.POST-TEST.csv', r'Sentence')
    df_es = ql.select_lang([2, 3], r'Sentence')
    df_es.to_csv(r'../data/' + r'ES-QA-Serialized-Sen-ELVA.PILOT.POST-TEST.csv', encoding= 'latin1')
    df_en = ql.select_lang([0, 1], r'Sentence')
    df_en.to_csv(r'../data/' + r'EN-QA-Serialized-Sen-ELVA.PILOT.POST-TEST.csv', encoding= 'latin1')
    '''
