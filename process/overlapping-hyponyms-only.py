#!/usr/bin/env python

from  overlapping import  odi_overlapping

stop_words_d = [r'a', r'be', r'to', r'and', r'or']
stop_words_hy = [r'be']

ovlpd = odi_overlapping(r'EN-QA-Serialized-Def-ELVA.PILOT.PRE-TEST.csv',
                        r'Questin_ID_Definition.csv', r'../data/', stop_words_d)
ovlpd.count_overlapping(r'EN-Lemma-Question-Def-PRE.csv', r'Student_Question_Index',
                     r'Pre_Col_Name', r'Question_ID', r'Question_ID_Sec',
                     r'EN-Lemma-Passage-Def-PRE.csv', r'Question_ID', r'Question_ID_Sec')
ovlpd.count_overlapping_hyponyms(r'EN-Hyponyms-Question-Def-PRE.csv', stop_words_hy)
ovlpd.df_ac_overlapping_hypo_lemma.to_csv(r'../data/' + 'Overlapping-Hyponyms-Def-PRE.csv',
                                          encoding= 'latin1')
