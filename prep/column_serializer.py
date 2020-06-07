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

class column_serializer:
    '''
    qid_csv_file_in is assumed to have the columns 'Question_ID' and 'Question'
    '''
    
    def __init__(self, data_dir=r'./', qid_csv_file_in = r'Questin_ID_Definition.csv'):
        self.data_dir = data_dir
        self.qid_csv_file_in = qid_csv_file_in

    def serialize_record(self, csv_file_in, key_word = r'Definition'):
        self.df_response_serialized = clsr.ac_column_serializer(self.data_dir + csv_file_in,
                    'Student_Index', self.columms_to_be_serialized(csv_file_in, key_word))
        new_question_clm_name = key_word + r'-Question'
        new_answer_clm_name = key_word + r'-Answer'
        self.df_response_serialized = self.df_response_serialized.rename(
            columns={r'Pre_Col_Name' : new_question_clm_name, r'Content' : new_answer_clm_name})

        score_columns = self.columms_to_be_serialized(csv_file_in, r'Score')
        self.df_score_serialized = clsr.ac_column_serializer(self.data_dir + csv_file_in,
                    r'Student_Index', score_columns)
        new_score_clm_name = key_word + r'-Score'
        self.df_score_serialized = self.df_score_serialized.rename(
            columns={r'Content' : new_score_clm_name})

        lang_columns = self.columms_to_be_serialized(csv_file_in, r'Language')
        self.df_lang_serialized = clsr.ac_column_serializer(self.data_dir + csv_file_in,
                    'Student_Index', lang_columns)
        new_lang_clm_name = key_word + r'-Language'
        self.df_lang_serialized = self.df_lang_serialized.rename(
            columns={r'Content' : new_lang_clm_name})

        self.df_all_serialized = self.df_response_serialized.copy()
        self.df_all_serialized[new_score_clm_name] = self.df_score_serialized[new_score_clm_name]
        self.df_all_serialized[new_lang_clm_name] = self.df_lang_serialized[new_lang_clm_name]
        self.df_all_serialized = self.df_all_serialized.drop(score_columns, axis=1)
        self.df_all_serialized = self.df_all_serialized.drop(lang_columns, axis=1)
        
        self.df_all_serialized = self.remove_word_from_clm_values(self.df_all_serialized, r'-' + key_word, new_question_clm_name)
        self.df_all_serialized = self.add_questin_id(self.df_all_serialized, self.qid_csv_file_in, new_question_clm_name)
        self.df_all_serialized = self.add_clm_values_to_index(self.df_all_serialized, new_question_clm_name)        
    def columms_to_be_serialized(self, csv_file_in, key_word):
        df_file_in = pd.read_csv(self.data_dir + csv_file_in, encoding= 'latin1')
        in_columns = df_file_in.columns
        ser_columns = []
        for x in in_columns:
            if key_word in x:
                ser_columns = ser_columns + [x]
        return ser_columns

    def remove_word_from_clm_values(self, df_to_be_mod, rm_word, column_to_be_mod):
        mod_data = df_to_be_mod[column_to_be_mod]
        new_data = []
        for x in mod_data:
            new_data = new_data + [x.replace(rm_word, r"")]
        df_new_data = pd.DataFrame({column_to_be_mod : new_data}, index = df_to_be_mod.index)
        df_to_be_mod[column_to_be_mod] = df_new_data[column_to_be_mod]
        return df_to_be_mod

    def add_questin_id(self, df_to_be_mod, qid_csv_file_in, column_to_be_matched):
        df_file_in = pd.read_csv(self.data_dir + qid_csv_file_in, encoding= 'latin1')
        df_file_in = df_file_in.rename(columns={'Question': column_to_be_matched})
        original_index = df_to_be_mod.index
        df_to_be_mod = pd.merge(df_to_be_mod, df_file_in, on=column_to_be_matched, how='left')
        #df_to_be_mod['Question_ID_Sec'] = df_to_be_mod['Question_ID']
        df_to_be_mod.index = original_index
        return df_to_be_mod
    
    def add_clm_values_to_index(self, df_to_be_mod, column_to_be_added):
        mod_index = df_to_be_mod.index
        add_data = df_to_be_mod[column_to_be_added]
        new_index = []
        for i in range(len(mod_index)):
            new_index = new_index + [str(mod_index[i]) + r'-' + add_data.iloc[i]]
        df_to_be_mod.index = new_index
        df_to_be_mod.index.name = r'Student_Question_Index'
        return df_to_be_mod
    
if __name__ == "__main__":
    cs = column_serializer(r'../data/')
    cs.serialize_record(r'Cleaned-Def-ELVA.PILOT.PRE-TEST (MODIFIED SCORES.Deidentified.11.19.17)12.13.2017.csv', r'Definition')
    cs.df_all_serialized.to_csv(r'../data/' + r'Serialized-Def-ELVA.PILOT.PRE-TEST.csv', encoding= 'latin1')

    '''
    cs.serialize_record(r'Cleaned-Sen-ELVA.PILOT.PRE-TEST (MODIFIED SCORES.Deidentified.11.19.17)12.13.2017.csv', r'Sentence')
    cs.df_all_serialized.to_csv(r'../data/' + r'Serialized-Sen-ELVA.PILOT.PRE-TEST.csv', encoding= 'latin1')
    cs.serialize_record(r'Cleaned-Def-ELVA.PILOT.POST-TEST (MODIFIED SCORES.Deidentifed11.19.17)12.13.2017.csv', r'Definition')
    cs.df_all_serialized.to_csv(r'../data/' + r'Serialized-Def-ELVA.PILOT.POST-TEST.csv', encoding= 'latin1')
    cs.serialize_record(r'Cleaned-Sen-ELVA.PILOT.POST-TEST (MODIFIED SCORES.Deidentifed11.19.17)12.13.2017.csv', r'Sentence')
    cs.df_all_serialized.to_csv(r'../data/' + r'Serialized-Sen-ELVA.PILOT.POST-TEST.csv', encoding= 'latin1')
    '''
