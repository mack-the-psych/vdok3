#!/usr/bin/env python

import pandas as pd

def bas_join_student_question_index(csv_file_src, csv_file_trg, data_dir=r'./'):
    df_ac_src = pd.read_csv(data_dir + csv_file_src, encoding= 'latin1')
    df_ac_src = df_ac_src.set_index(r'AC_Doc_ID')
    df_ac_trg = pd.read_csv(data_dir + csv_file_trg, encoding= 'latin1')
    df_ac_trg = df_ac_trg.set_index(r'AC_Doc_ID')

    df_ac_trg[r'Student_Question_Index'] = df_ac_src[r'Student_Question_Index']
    return df_ac_trg

if __name__ == "__main__":
    data_dir = r'../data/'

    df_ac_merge_trg = bas_join_student_question_index(r'Key-Stem-Passage-Aggregate_plim-Def-PRE.csv',
                                             r'RF_Classified-Prediction-Def-PRE-All.csv', data_dir)
    df_ac_merge_trg.to_csv(data_dir + r'Refmt_RF_Classified-Prediction-Def-PRE-All.csv',
                          encoding= 'latin1')

    df_ac_merge_trg = bas_join_student_question_index(r'Key-Stem-Passage-Aggregate_plim-Sen-PRE.csv',
                                             r'RF-Classified-Prediction-Sen-PRE-All.csv', data_dir)
    df_ac_merge_trg.to_csv(data_dir + r'Refmt_RF-Classified-Prediction-Sen-PRE-All.csv',
                          encoding= 'latin1')

    df_ac_merge_trg = bas_join_student_question_index(r'Key-Stem-Passage-Aggregate_plim-Def-PRE-POST.csv',
                                             r'RF-Classified-Prediction-Def-PRE-POST-All.csv', data_dir)
    df_ac_merge_trg.to_csv(data_dir + r'Refmt_RF-Classified-Prediction-Def-PRE-POST-All.csv',
                          encoding= 'latin1')

    df_ac_merge_trg = bas_join_student_question_index(r'Key-Stem-Passage-Aggregate_plim-Sen-PRE-POST.csv',
                                             r'RF-Classified-Prediction-Sen-PRE-POST-All.csv', data_dir)
    df_ac_merge_trg.to_csv(data_dir + r'Refmt_RF-Classified-Prediction-Sen-PRE-POST-All.csv',
                          encoding= 'latin1')

    data_dir = r'../data/wo_DK_NR/'

    df_ac_merge_trg = bas_join_student_question_index(r'Key-Stem-Passage-Aggregate_plim-Def-PRE_wo_DK_NR.csv',
                                             r'RF_wo_DK_NR_Classified-Prediction-Def-PRE-All.csv', data_dir)
    df_ac_merge_trg.to_csv(data_dir + r'Refmt_RF_wo_DK_NR_Classified-Prediction-Def-PRE-All.csv',
                          encoding= 'latin1')

    df_ac_merge_trg = bas_join_student_question_index(r'Key-Stem-Passage-Aggregate_plim-Sen-PRE_wo_DK_NR.csv',
                                             r'RF_wo_DK_NR-Classified-Prediction-Sen-PRE-All.csv', data_dir)
    df_ac_merge_trg.to_csv(data_dir + r'Refmt_RF_wo_DK_NR-Classified-Prediction-Sen-PRE-All.csv',
                          encoding= 'latin1')

    df_ac_merge_trg = bas_join_student_question_index(r'Key-Stem-Passage-Aggregate_plim-Def-PRE-POST_wo_DK_NR.csv',
                                             r'RF-Classified-Prediction-Def-PRE-POST_wo_DK_NR-All.csv', data_dir)
    df_ac_merge_trg.to_csv(data_dir + r'Refmt_RF-Classified-Prediction-Def-PRE-POST_wo_DK_NR-All.csv',
                          encoding= 'latin1')

    df_ac_merge_trg = bas_join_student_question_index(r'Key-Stem-Passage-Aggregate_plim-Sen-PRE-POST_wo_DK_NR.csv',
                                             r'RF-Classified-Prediction-Sen-PRE-POST_wo_DK_NR-All.csv', data_dir)
    df_ac_merge_trg.to_csv(data_dir + r'Refmt_RF-Classified-Prediction-Sen-PRE-POST_wo_DK_NR-All.csv',
                          encoding= 'latin1')
