#!/usr/bin/env python

import pandas as pd

def refmt_serialized_sen_for_word_embd(csv_file_sen, csv_file_sen_exmpl, data_dir=r'./'):
    df_ac_sen = pd.read_csv(data_dir + csv_file_sen, encoding= 'latin1')
    df_ac_sen = df_ac_sen.set_index('Student_Question_Index')
    df_ac_sen_exmpl = pd.read_csv(data_dir + csv_file_sen_exmpl, encoding= 'latin1')
    df_ac_merge = pd.merge(df_ac_sen, df_ac_sen_exmpl.loc[:,['Question_ID', 'Example']],
                           on='Question_ID', how='left')
    df_ac_merge.index = df_ac_sen.index
    df_ac_merge.index.name = df_ac_sen.index.name
    df_ac_merge = df_ac_merge.drop(['Definition'], axis=1)
    return df_ac_merge

if __name__ == "__main__":
    data_dir = r'../data/'

    df_ac_merge_pre = refmt_serialized_sen_for_word_embd(r'Serialized-Sen-ELVA.PILOT.PRE-TEST.csv',
                                             r'Questin_ID_Sentence_Example.csv', data_dir)
    df_ac_merge_pre.to_csv(data_dir + r'Refmt_Serialized-Sen-ELVA.PILOT.PRE-TEST.csv',
                          encoding= 'latin1')

    df_ac_merge_post = refmt_serialized_sen_for_word_embd(r'Serialized-Sen-ELVA.PILOT.POST-TEST.csv',
                                             r'Questin_ID_Sentence_Example.csv', data_dir)
    df_ac_merge_post.to_csv(data_dir + r'Refmt_Serialized-Sen-ELVA.PILOT.POST-TEST.csv',
                          encoding= 'latin1')
