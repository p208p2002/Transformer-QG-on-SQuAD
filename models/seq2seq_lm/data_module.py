from torch.utils.data import DataLoader,Dataset,ConcatDataset
import os
import json
from .tokenizer import get_tokenizer
from .argparser import get_args
import pytorch_lightning as pl
from datasets import load_dataset
from .config import MODEL_CONFIG, MAX_INPUT_LENGTH, MAX_QUESTION_LENGTH, HL_TOKEN
import torch
args = get_args()

class DataModule(pl.LightningDataModule):
    def __init__(self,args = get_args()):
        super().__init__()
        self.batch_size = args.batch_size
        self.train_dataset = DrcdDataset(split_set='train')
        self.dev_dataset = DrcdDataset(split_set='dev')
        self.test_dataset = DrcdDataset(split_set='test',is_test=True)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)

class DatasetUtilsMixin():
     def prepare_input(self,context,label=None):
        tokenizer = self.tokenizer
        pad_token_id = tokenizer.pad_token_id
        input_encodings = tokenizer(context, padding='max_length' if label is not None else False, max_length=MAX_INPUT_LENGTH, truncation=True, add_special_tokens=False)
        
        if label is not None:
            labels = []
            target_encodings = tokenizer(label, padding='max_length', max_length=MAX_QUESTION_LENGTH, truncation=True, add_special_tokens=False)
            for target_encoding_id in target_encodings['input_ids']:
                if target_encoding_id != pad_token_id:
                    labels.append(target_encoding_id)
                else:
                    labels.append(-100)
        else:
            labels = None

        #   
        model_input = {
            'input_ids':input_encodings['input_ids'],
            'attention_mask':input_encodings['attention_mask'],
            'labels': labels
        }
        if label is None: del model_input['labels']

        # convert to tensor
        for key in model_input.keys():
            model_input[key] = torch.LongTensor(model_input[key])

        return model_input

class DrcdDataset(Dataset,DatasetUtilsMixin):
    def __init__(self,split_set:str='train',tokenizer = get_tokenizer(args.base_model),is_test=False):
        """
        Args:
            split_set(str): `train`, `dev` or `test`
            tokenizer(transformers.PreTrainedTokenizer)
        """
        
        if split_set == 'train':
            with open('datasets/drcd/train.json','r',encoding='utf-8') as f_train:
                train_set = json.load(f_train)
                self.data = train_set
        elif split_set == 'dev':
            with open('datasets/drcd/dev.json','r',encoding='utf-8') as f_dev:
                dev_set = json.load(f_dev)
                self.data = dev_set
        elif split_set == 'test':
            with open('datasets/drcd/test.json','r',encoding='utf-8') as f_test:
                test_set = json.load(f_test)
                self.data = test_set
        
        # convert
        self.data = self.data['data']
        new_data = []
        for data in self.data:
            for d in data['paragraphs']:
                context = d['context']
                for qa in d['qas']:
                    new_data.append({
                        'context':context,
                        'answers':qa['answers'],
                        'question':qa['question']
                    })
        
        self.data = new_data
        self.split_set = split_set
        self.is_test = is_test
        self.tokenizer = tokenizer
    
    def __getitem__(self,index):
        data = self.data[index]
        
        answer_text = data['answers'][0]['text']
        answer_len = len(answer_text)
        answer_start = data['answers'][0]['answer_start']
        hl_context = data['context'][:answer_start] + HL_TOKEN + answer_text + HL_TOKEN + data['context'][answer_start + answer_len:]

        if self.is_test == False:
            model_input = self.prepare_input(context=hl_context,label=data['question'] + self.tokenizer.sep_token)
            return model_input['input_ids'],model_input['attention_mask'],model_input['labels'] 
        else:
            model_input = self.prepare_input(context=hl_context)
            return model_input['input_ids'],model_input['attention_mask'],data['question']

    def __len__(self):
        return len(self.data)
        