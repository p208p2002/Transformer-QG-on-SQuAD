from torch.utils.data import DataLoader,Dataset
import os
import json
from .tokenizer import get_tokenizer
from .argparser import get_args
import pytorch_lightning as pl
from datasets import load_dataset
from .config import MODEL_CONFIG, MAX_INPUT_LENGTH, MAX_QUESTION_LENGTH, MAX_CONTEXT_LENGTH, HL_TOKEN
import torch
import copy
import re
from loguru import logger
args = get_args()


class DataModule(pl.LightningDataModule):
    def __init__(self,args = get_args()):
        super().__init__()
        self.batch_size = args.batch_size
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'

        if args.dataset == 'squad':
            self.train_dataset = SquadQGDataset(split_set='train')
            self.dev_dataset = SquadQGDataset(split_set='validation')
            self.test_dataset = SquadQGDataset(split_set='validation',is_test=True)
        elif args.dataset == 'squad-nqg':
            self.train_dataset = SquadNQGDataset(split_set='train')
            self.dev_dataset = SquadNQGDataset(split_set='dev')
            self.test_dataset = SquadNQGDataset(split_set='test',is_test=True)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)

class DatasetUtilsMixin():
    def convert_to_tensor(self,model_input):
        for key in model_input.keys():
            model_input[key] = torch.LongTensor(model_input[key])
        return model_input

    def prepare_input(self,context,label=None,question_mask_position=None):

        if label is None:
            model_input = self.tokenizer(context,max_length=MAX_CONTEXT_LENGTH,truncation=True)
            return self.convert_to_tensor(model_input)

        context_encoding = self.tokenizer(context,return_length=True,max_length=MAX_CONTEXT_LENGTH,truncation=True,add_special_tokens=True)['input_ids']
        
        model_input_encodeing = self.tokenizer(label,return_length=True,max_length=MAX_QUESTION_LENGTH,truncation=True,add_special_tokens=False)['input_ids']
        
        try:
            label_id = model_input_encodeing[question_mask_position]
            model_input_encodeing[question_mask_position] = self.tokenizer.mask_token_id
            model_input_encodeing = model_input_encodeing[:question_mask_position+1]
        except:
            label_id = -100

        label_encoding = [-100]*len(model_input_encodeing)
        if label_id != -100:
            label_encoding[question_mask_position] = label_id
        label_encoding = [-100]*len(context_encoding) + label_encoding

        input_ids = context_encoding+model_input_encodeing
        labels = label_encoding
        # token_type_ids = [0]*len(context_encoding) + [1]*len(model_input_encodeing)

        # pad
        while len(input_ids)<MAX_INPUT_LENGTH:
            input_ids.append(self.tokenizer.pad_token_id)
            labels.append(-100)
            # token_type_ids.append(1)

        return self.convert_to_tensor({
            'input_ids':input_ids,
            # 'token_type_ids':token_type_ids,
            'labels':labels
        })

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

        # compute total data length for Recurrent Masked
        # Recurrent Masked compute one token loss once
        # we have to find out the number of masked token
        if not is_test:
            new_data = []
            for i,data in enumerate(self.data):
                question = data['question']
                question_encodeing = self.tokenizer(question,return_length=True,max_length=MAX_QUESTION_LENGTH,truncation=True,add_special_tokens=False)
                question_length = question_encodeing['length'][0] + 1 # add 1 for [sep] at the end
                for j in range(question_length):
                    data['question_length']= question_length
                    data['question_mask_position']= j
                    new_data.append(copy.deepcopy(data))
                print("loading...%d/%d"%(i,len(self.data)),end='\r')
                # if i == 200: break
                if args.dev and i == 1000: break
            self.data = new_data
        
    def __getitem__(self,index):
        data = self.data[index]
        # print(data['context'])
        answer_text = data['answers']['text'][0]
        answer_len = len(answer_text)
        answer_start = data['answers']['answer_start'][0] 
        hl_context = data['context'][:answer_start] + HL_TOKEN + answer_text + HL_TOKEN + data['context'][answer_start + answer_len:]

        if self.is_test == False:
            question_mask_position = data['question_mask_position']
            model_input = self.prepare_input(
                    context=hl_context,
                    label=data['question'] + self.tokenizer.sep_token,
                    question_mask_position=question_mask_position
                )
            return model_input['input_ids'],model_input['labels']
        else:
            model_input = self.prepare_input(context=hl_context)
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

        # compute total data length for Recurrent Masked
        # Recurrent Masked compute one token loss once
        # we have to find out the number of masked token
        if not is_test:
            new_data = []
            for i,data in enumerate(self.data):
                question = data['question']
                question_encodeing = self.tokenizer(question,return_length=True,max_length=MAX_QUESTION_LENGTH,truncation=True,add_special_tokens=False)
                question_length = question_encodeing['length'][0] + 1 # add 1 for [sep] at the end
                for j in range(question_length):
                    data['question_length']= question_length
                    data['question_mask_position']= j
                    new_data.append(copy.deepcopy(data))
                print("loading...%d/%d"%(i,len(self.data)),end='\r')
                if args.dev and i == 1000: break
            self.data = new_data
        
    def __getitem__(self,index):
        data = self.data[index]
        
        answer_text = data['answers'][0]['text']
        answer_len = len(answer_text)
        answer_start = data['answers'][0]['answer_start']
        hl_context = data['context'][:answer_start] + HL_TOKEN + answer_text + HL_TOKEN + data['context'][answer_start + answer_len:]

        if self.is_test == False:
            question_mask_position = data['question_mask_position']
            model_input = self.prepare_input(
                    context=hl_context,
                    label=data['question'] + self.tokenizer.sep_token,
                    question_mask_position=question_mask_position
                )
            return model_input['input_ids'],model_input['labels']
        else:
            model_input = self.prepare_input(context=hl_context)
            return model_input['input_ids'],data['question']
        
    def __len__(self):
        return len(self.data)