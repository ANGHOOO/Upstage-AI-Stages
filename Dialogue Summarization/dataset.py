import pandas as pd
from torch.utils.data import Dataset
import torch


class Preprocess:
    def __init__(self, bos_token, eos_token):
        self.bos_token = bos_token 
        self.eos_token = eos_token

    @staticmethod
    def make_set_as_df(file_path, is_train=True):
        df = pd.read_csv(file_path)
        return df[['fname', 'dialogue', 'summary']] if is_train else df[['fname', 'dialogue']]
        
    def make_input(self, dataset, is_test=False):
        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x : self.bos_token + str(x))
            decoder_output = dataset['summary'].apply(lambda x : str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()


class DatasetForTrain(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, length):
        self.encoder_input = encoder_input 
        self.decoder_input = decoder_input 
        self.labels = labels 
        self.length = length 

    def __getitem__(self, idx):
        item = {key : val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item2 = {key : val[idx].clone().detach() for key, val in self.decoder_input.items()}
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        item.update(item2)
        item['labels'] = self.labels['input_ids'][idx]
        return item 
    
    def __len__(self):
        return self.length


class DatasetForVal(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, length):
        self.encoder_input = encoder_input 
        self.decoder_input = decoder_input
        self.labels = labels 
        self.length = length 

    def __getitem__(self, idx):
        item = {key : val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item2 = {key : val[idx].clone().detach() for key, val in self.decoder_input.items()}
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        item.update(item2)
        item['labels'] = self.labels['input_ids'][idx]
        return item 
    
    def __len__(self):
        return self.length


class DatasetForInference(Dataset):
    def __init__(self, encoder_input, test_id, length):
        self.encoder_input = encoder_input 
        self.test_id = test_id 
        self.length = length 

    def __getitem__(self, idx):
        item = {key : val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item['ID'] = self.test_id[idx]
        return item 
    
    def __len__(self):
        return self.length
