from torch.utils.data import DataLoader,Dataset,ConcatDataset
import os
import json
from .tokenizer import get_tokenizer
from .argparser import get_args
import pytorch_lightning as pl
from datasets import load_dataset
from .config import MODEL_CONFIG, MAX_INPUT_LENGTH, MAX_QUESTION_LENGTH, MAX_CONTEXT_LENGTH, HL_TOKEN
import torch
args = get_args()

class DataModule(pl.LightningDataModule):
    def __init__(self,args = get_args()):
        super().__init__()
        self.batch_size = 1

        if args.dataset == 'squad':
            self.train_dataset = SquadQGDataset(split_set='train')
            self.dev_dataset = SquadQGDataset(split_set='validation')
            self.test_dataset = SquadQGDataset(split_set='validation',is_test=True)
        elif args.dataset == 'squad-nqg':
            self.train_dataset = SquadNQGDataset(split_set='train')
            self.dev_dataset = SquadNQGDataset(split_set='dev')
            self.test_dataset = SquadNQGDataset(split_set='test',is_test=True)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)

class DatasetUtilsMixin():
    def convert_to_tensor(self,model_input):
        for key in model_input.keys():
            model_input[key] = torch.LongTensor(model_input[key])
        return model_input

    def prepare_input(self,context,label=None):
        tokenizer = self.tokenizer
        pad_token_id = self.tokenizer.pad_token_id

        if label is None:
            model_input = tokenizer(context,max_length=MAX_CONTEXT_LENGTH,truncation=True)
            return self.convert_to_tensor(model_input)

        context_input = tokenizer(context)
        label_input = tokenizer(label)
        
        # limit context length
        model_input = {}
        model_input['input_ids'] = context_input['input_ids'][:MAX_CONTEXT_LENGTH] + label_input['input_ids'][:MAX_QUESTION_LENGTH]
        
        # prepars lables
        model_input['labels'] = model_input['input_ids'][:]
        for i,_ in enumerate(context_input['input_ids'][:MAX_CONTEXT_LENGTH]):
            model_input['labels'][i] = -100 # set the context part to -100 for ignore loss

        # pad or limit to max length
        pad_ids = [pad_token_id]*MAX_INPUT_LENGTH
        pad_labels = [-100]*MAX_INPUT_LENGTH
        model_input['input_ids'] = (model_input['input_ids'] + pad_ids)[:MAX_INPUT_LENGTH] 
        model_input['labels'] = (model_input['labels'] + pad_labels)[:MAX_INPUT_LENGTH]        

        return self.convert_to_tensor(model_input)

class SquadQGDataset(Dataset,DatasetUtilsMixin):
    def __init__(self,split_set:str='train',tokenizer = get_tokenizer(args.base_model),is_test=False):
        """
        Args:
            split_set(str): `train` or `validation`
            tokenizer(transformers.PreTrainedTokenizer)
        """
        dataset = load_dataset("squad")
        self.split_set = split_set
        self.is_test = is_test
        self.data = dataset[split_set]
        self.tokenizer = tokenizer
        
    def __getitem__(self,index):
        data = self.data[index]
        # print(data['context'])
        answer_text = data['answers']['text'][0]
        answer_len = len(answer_text)
        answer_start = data['answers']['answer_start'][0] 
        hl_context = data['context'][:answer_start] + HL_TOKEN + answer_text + HL_TOKEN + data['context'][answer_start + answer_len:]

        if self.is_test == False:
            model_input = self.prepare_input(context=hl_context + self.tokenizer.sep_token,label=data['question'] + self.tokenizer.eos_token)
            return model_input['input_ids'],model_input['labels'] 
        else:
            model_input = self.prepare_input(context=hl_context + self.tokenizer.sep_token)
            return model_input['input_ids'],data['question']
        
    def __len__(self):
        return len(self.data)

class SquadNQGDataset(Dataset,DatasetUtilsMixin):
    def __init__(self,split_set:str='train',tokenizer = get_tokenizer(args.base_model),is_test=False):
        """
        Args:
            split_set(str): `train`, `dev` or `test`
            tokenizer(transformers.PreTrainedTokenizer)
        """
        
        if split_set == 'train':
            with open('datasets/squad-nqg/train.json','r',encoding='utf-8') as f_train:
                train_set = json.load(f_train)
                self.data = train_set
        elif split_set == 'dev':
            with open('datasets/squad-nqg/dev.json','r',encoding='utf-8') as f_dev:
                dev_set = json.load(f_dev)
                self.data = dev_set
        elif split_set == 'test':
            with open('datasets/squad-nqg/test.json','r',encoding='utf-8') as f_test:
                test_set = json.load(f_test)
                self.data = test_set
        
        # convert
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
            model_input = self.prepare_input(context=hl_context,label=data['question'] + self.tokenizer.eos_token)
            return model_input['input_ids'],model_input['labels'] 
        else:
            model_input = self.prepare_input(context=hl_context)
            return model_input['input_ids'],data['question']
        
    def __len__(self):
        return len(self.data)