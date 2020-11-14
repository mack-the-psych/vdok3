#!/usr/bin/env python
# Reference: https://github.com/YutaroOgawa/pytorch_advanced

# Put a path file like "plimac-custom.pth" into any of your sys.path directories
# (e.g. C:\Users\macks\Anaconda3\Lib\site-packages).

# vdok-custom.pth ###############################
#
# # .pth file for the VDOK extension
#
# C:\Users\macks\Documents\Research\MELVA-S\vdok3\prep
# C:\Users\macks\Documents\Research\MELVA-S\vdok3\extract
# C:\Users\macks\Documents\Research\MELVA-S\vdok3\process
# C:\Users\macks\Documents\Research\MELVA-S\vdok3\reorganize
# C:\Users\macks\Documents\Research\MELVA-S\vdok3\train
# C:\Users\macks\Documents\Research\MELVA-S\vdok3\train\pytorch_advanced\nlp_sentiment_bert
#
###################################################

# may need to modify the torchtext code of utils.py
# e.g. C:\Users\macks\Anaconda3\Lib\site-packages\torchtext\utils.py
# (see: https://stackoverflow.com/questions/57988897/overflowerror-python-int-too-large-to-convert-to-c-long-torchtext-datasets-text)

import pandas as pd
import numpy as np
import string

import random
import time
from tqdm import tqdm
import torch 
from torch import nn
import torch.optim as optim
import torchtext
from utils.bert import BertTokenizer, load_vocab
from utils.bert import get_config, BertModel, set_learned_params
        
max_length = 128
random_seed = 0

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

import tf_log_regress_classify as lreg

class tmv_torch_bert_classify(lreg.tmv_tf_log_regress_classify):
    def __init__(self, data_dir=r'./'):
        self.data_dir = data_dir
        self.tokenizer_bert = BertTokenizer(
            vocab_file="./pytorch_advanced/nlp_sentiment_bert/vocab/bert-base-uncased-vocab.txt",
            do_lower_case=True)
        self.vocab_bert, self.ids_to_tokens_bert = load_vocab(
            vocab_file="./pytorch_advanced/nlp_sentiment_bert/vocab/bert-base-uncased-vocab.txt")

        config = get_config(file_path="./pytorch_advanced/nlp_sentiment_bert/weights/bert_config.json")
        self.net_bert = BertModel(config)
        self.net_bert = set_learned_params(self.net_bert,
                weights_path="./pytorch_advanced/nlp_sentiment_bert/weights/pytorch_model.bin")
                
    def load_data(self, csv_file_kspa, dependent_var, langs = None, task_word = 'Definition',
                  answer_ex_clm = 'Definition'):
        self.dependent_var = dependent_var
        self.answer_ex_clm = answer_ex_clm
        self.df_response_answer_ex = pd.read_csv(self.data_dir + csv_file_kspa, encoding= 'latin1')
        self.df_response_answer_ex = self.df_response_answer_ex.set_index(r'Student_Question_Index')

        if langs != None:
            lang_clm = task_word + r'-Language'
            self.df_response_answer_ex = \
                self.df_response_answer_ex[self.df_response_answer_ex[lang_clm].isin(langs)]
            
        self.ans_clm = task_word + r'-Answer'
        self.ans_and_ex_clm = task_word + r'-Answer-and-Example'
        
        self.df_response_answer_ex[self.ans_and_ex_clm] = self.df_response_answer_ex[self.answer_ex_clm] \
            + ' ' + self.df_response_answer_ex[self.ans_clm]

        # to move LABEL and TXT columns to the end
        columns = list(self.df_response_answer_ex.columns)
        columns.remove(self.dependent_var)
        columns.remove(self.ans_and_ex_clm)
        columns.append(self.dependent_var)
        columns.append(self.ans_and_ex_clm)
        self.df_ac_modeling_values = self.df_response_answer_ex.reindex(columns=columns)

    def get_tokens(self):        
        def preprocessing_text(text):
            for p in string.punctuation:
                if (p == ".") or (p == ","):
                    continue
                else:
                    text = text.replace(p, " ")

            text = text.replace(".", " . ")
            text = text.replace(",", " , ")
            return text

        def tokenizer_with_preprocessing(text, tokenizer=self.tokenizer_bert.tokenize):
            text = preprocessing_text(text)
            ret = tokenizer(text)  # tokenizer_bert
            return ret

        TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True,
                            lower=True, include_lengths=True, batch_first=True, fix_length=max_length,
                            init_token="[CLS]", eos_token="[SEP]", pad_token='[PAD]', unk_token='[UNK]')
        LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

        fields = [(None, None)] * (len(self.df_response_answer_ex.columns) -1)
        fields.append(('Label', LABEL))
        fields.append(('Text', TEXT))

        train_val_ds = torchtext.data.TabularDataset(path=self.modeling_data_file_name, format='csv',
            fields= fields, skip_header=True)

        TEXT.build_vocab(train_val_ds, min_freq=1)
        TEXT.vocab.stoi = self.vocab_bert

        return train_val_ds
        
    def perform_modeling(self, df_ac_modeling_data, key_word = r'', csv_dump = False,
                         number_class = 3, epochs = 10, batch_size = 32,
                         tmp_csv_name = 'TORCH_RESPONSE_ANSWER_EX_FILE.CSV'):
        self.modeling_data_file_name = self.data_dir + tmp_csv_name
        self.batch_size = batch_size
        df_ac_modeling_data_buf = df_ac_modeling_data.copy()
        
        if self.ans_and_ex_clm not in df_ac_modeling_data_buf.columns:
            df_ac_modeling_data_buf[self.ans_and_ex_clm] = df_ac_modeling_data_buf[self.answer_ex_clm] \
                                    + ' ' + df_ac_modeling_data_buf[self.ans_clm]

            # to move LABEL and TXT columns to the end
            columns = list(df_ac_modeling_data_buf.columns)
            columns.remove(self.dependent_var)
            columns.remove(self.ans_and_ex_clm)
            columns.append(self.dependent_var)
            columns.append(self.ans_and_ex_clm)
            df_ac_modeling_data_buf = df_ac_modeling_data_buf.reindex(columns=columns)

        df_ac_modeling_data_buf.to_csv(self.modeling_data_file_name)
        
        train_val_ds = self.get_tokens()

        train_ds, val_ds = train_val_ds.split(split_ratio=0.8, random_state=random.seed(random_seed))
                
        train_dl = torchtext.data.Iterator(
            train_ds, batch_size=self.batch_size, train=True)

        val_dl = torchtext.data.Iterator(
            val_ds, batch_size=self.batch_size, train=False, sort=False)

        self.dataloaders_dict = {"train": train_dl, "val": val_dl}
        
        batch = next(iter(val_dl))
        print(batch.Text)
        print(batch.Label)
                
        text_minibatch_1 = (batch.Text[0][1]).numpy()
        text = self.tokenizer_bert.convert_ids_to_tokens(text_minibatch_1)
        print(text)

        print('Building model...')
        net = BertForVDOK(self.net_bert)
        net.train()
                
        for name, param in net.named_parameters():
            param.requires_grad = False

        for name, param in net.bert.encoder.layer[-1].named_parameters():
            param.requires_grad = True

        for name, param in net.cls.named_parameters():
            param.requires_grad = True
                
        optimizer = optim.Adam([
            {'params': net.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
            {'params': net.cls.parameters(), 'lr': 5e-5}
        ], betas=(0.9, 0.999))

        self.criterion = nn.CrossEntropyLoss()

        self.net_trained = self.train_model(net, self.dataloaders_dict, self.criterion, optimizer, num_epochs=epochs)

        save_path = './pytorch_advanced/nlp_sentiment_bert/weights/bert_fine_tuning_VDOK_' + key_word + '.pth'
        torch.save(self.net_trained.state_dict(), save_path)
        
    def train_model(self, net, dataloaders_dict, criterion, optimizer, num_epochs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        print('Using device: ', device)
        print('-----start-------')

        net.to(device)

        torch.backends.cudnn.benchmark = True

        batch_size = dataloaders_dict["train"].batch_size

        for epoch in range(num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    net.train()
                else:
                    net.eval()

                epoch_loss = 0.0 
                epoch_corrects = 0
                iteration = 1

                t_epoch_start = time.time()
                t_iter_start = time.time()

                for batch in (dataloaders_dict[phase]):
                    inputs = batch.Text[0].to(device)
                    labels = batch.Label.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = net(inputs, token_type_ids=None, attention_mask=None,
                                      output_all_encoded_layers=False, attention_show_flg=False)

                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                            if (iteration % 10 == 0):
                                t_iter_finish = time.time()
                                duration = t_iter_finish - t_iter_start
                                acc = (torch.sum(preds == labels.data)
                                       ).double()/batch_size
                                print('Iteration {} || Loss: {:.4f} || 10iter: {:.4f} sec. || Accuracy: {}'.format(iteration, loss.item(), duration, acc))
                                t_iter_start = time.time()

                        iteration += 1

                        epoch_loss += loss.item() * batch_size
                        epoch_corrects += torch.sum(preds == labels.data)

                t_epoch_finish = time.time()
                epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                epoch_acc = epoch_corrects.double(
                ) / len(dataloaders_dict[phase].dataset)

                print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs,
                                                                               phase, epoch_loss, epoch_acc))
                t_epoch_start = time.time()

        return net
    
    def perform_prediction(self, df_ac_prediction_data, number_class):
        self.df_ac_predict_target = df_ac_prediction_data.loc[:,[self.dependent_var]]
        df_ac_prediction_data_buf = df_ac_prediction_data.copy()

        if self.ans_and_ex_clm not in df_ac_prediction_data_buf.columns:
            df_ac_prediction_data_buf[self.ans_and_ex_clm] = df_ac_prediction_data_buf[self.answer_ex_clm] \
                                + ' ' + df_ac_prediction_data_buf[self.ans_clm]

            # to move LABEL and TXT columns to the end
            columns = list(df_ac_prediction_data_buf.columns)
            columns.remove(self.dependent_var)
            columns.remove(self.ans_and_ex_clm)
            columns.append(self.dependent_var)
            columns.append(self.ans_and_ex_clm)
            df_ac_prediction_data_buf = df_ac_prediction_data_buf.reindex(columns=columns)

        df_ac_prediction_data_buf.to_csv(self.modeling_data_file_name)
        
        test_ds = self.get_tokens()
        test_dl = torchtext.data.Iterator(test_ds, batch_size=self.batch_size, train=False, sort=False)
                
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net_trained.eval()
        self.net_trained.to(device)

        epoch_corrects = 0
        self.predict_res = []

        for batch in tqdm(test_dl): 
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            inputs = batch.Text[0].to(device)
            labels = batch.Label.to(device)

            with torch.set_grad_enabled(False):
                outputs = self.net_trained(inputs, token_type_ids=None, attention_mask=None,
                                      output_all_encoded_layers=False, attention_show_flg=False)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                epoch_corrects += torch.sum(preds == labels.data)
                self.predict_res += preds.tolist()

        epoch_acc = epoch_corrects.double() / len(test_dl.dataset)

        print('Test Data {} Accuracy: {:.4f}'.format(len(test_dl.dataset), epoch_acc))
        
        self.df_ac_classified = pd.DataFrame(np.array(self.predict_res,
                        dtype=np.int64), df_ac_prediction_data.index,
                        [r'Score_Class'])
        self.df_ac_classified[self.dependent_var] = self.df_ac_predict_target[self.dependent_var]

    def modeling_prediction_evaluation_all(self, key_word = r'', csv_dump = False, number_class = 3,
                                           epochs = 10, batch_size = 32):
        self.df_ac_predict_target_all = pd.DataFrame()
        self.predict_res_all = np.array([], np.int64)
        self.df_ac_classified_all = pd.DataFrame()
                
        for x in range(len(self.random_order_set)):            
            print(r'----------------')
            print(r'RANDOM SET: ', x)
            self.iloc_concat_for_cross_validation(x)
            self.perform_modeling(self.df_ac_modeling_values.iloc[self.concatenated_value_order, :],
                                  key_word, csv_dump, number_class, epochs)
            self.perform_prediction(self.df_ac_modeling_values.iloc[self.random_order_set[x], :], number_class)
            self.evaluate_prediction(key_word)
            if len(self.df_ac_predict_target_all) == 0:
                self.df_ac_predict_target_all = self.df_ac_predict_target.copy()
            else:
                self.df_ac_predict_target_all = self.df_ac_predict_target_all.append(self.df_ac_predict_target)
            self.predict_res_all = np.append(self.predict_res_all, self.predict_res)
            if len(self.df_ac_classified_all) == 0:
                self.df_ac_classified_all = self.df_ac_classified.copy()
                self.df_indices_all = pd.DataFrame(self.se_indices)
            else:
                self.df_ac_classified_all = self.df_ac_classified_all.append(self.df_ac_classified)
                self.df_indices_all = pd.concat([self.df_indices_all, self.se_indices], axis=1)

        self.df_indices_all = self.df_indices_all.T
        print(r'----------------')
        print(r'ALL DATA (Macro Average):')
        print(self.df_indices_all.describe())
        if csv_dump == True:
            self.df_indices_all.describe().to_csv(self.data_dir + r'Classified-Prediction-Indices-Macro-' + key_word + r'.csv', encoding= 'latin1')
        print(r'----------------')
        print(r'ALL DATA (Micro Average):')
        self.evaluate_prediction(key_word, csv_dump = True,
                df_ac_predict_target = self.df_ac_predict_target_all, predict_res = self.predict_res_all)

class BertForVDOK(nn.Module):
    def __init__(self, net_bert):
        super(BertForVDOK, self).__init__()

        self.bert = net_bert

        self.cls = nn.Linear(in_features=768, out_features=number_class)

        nn.init.normal_(self.cls.weight, std=0.02)
        nn.init.normal_(self.cls.bias, 0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                output_all_encoded_layers=False, attention_show_flg=False):
        if attention_show_flg == True:
            encoded_layers, pooled_output, attention_probs = self.bert(
                input_ids, token_type_ids, attention_mask, output_all_encoded_layers, attention_show_flg)
        elif attention_show_flg == False:
            encoded_layers, pooled_output = self.bert(
                input_ids, token_type_ids, attention_mask, output_all_encoded_layers, attention_show_flg)

        vec_0 = encoded_layers[:, 0, :]
        vec_0 = vec_0.view(-1, 768)
        out = self.cls(vec_0)

        if attention_show_flg == True:
            return out, attention_probs
        elif attention_show_flg == False:
            return out
    
if __name__ == "__main__":
    number_data_set = 4
    csv_dump = True
    epochs = 1
    dependent_var = r'Definition-Score'
    task_word = r'Definition'
    number_class = 3
    top_to = 1000

    df_response_answer = pd.read_csv(r'../data/' + r'Serialized-Def-ELVA.PILOT.PRE-TEST.csv', encoding= 'latin1')
    df_response_answer.iloc[:top_to, :].to_csv(r'../data/' + 'Top-to-' + str(top_to) + r'-Serialized-Def-ELVA.PILOT.PRE-TEST.csv', encoding= 'latin1')
    
    bertd = tmv_torch_bert_classify(r'../data/')
    # bertd.load_data(r'Serialized-Def-ELVA.PILOT.PRE-TEST.csv', dependent_var, [0, 1], task_word)
    bertd.load_data('Top-to-' + str(top_to) + r'-Serialized-Def-ELVA.PILOT.PRE-TEST.csv',
                    dependent_var, [0, 1], task_word)
    bertd.iloc_split_for_cross_validation(number_data_set = number_data_set)
    bertd.modeling_prediction_evaluation_all(r'TORCH_BERT-Def-PRE-All', csv_dump, number_class,
                                             epochs=epochs)
    bertd.df_ac_classified_all.to_csv(r'../data/' + 'TORCH_BERT-Classified-Prediction-Def-PRE-All.csv',
                                      encoding= 'latin1')
