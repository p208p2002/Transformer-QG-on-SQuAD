from torch.utils.data import DataLoader,Dataset,ConcatDataset
import os
import json
from .tokenizer import get_tokenizer
from .argparser import get_args
import pytorch_lightning as pl
from datasets import load_dataset
from .config import MODEL_CONFIG, MAX_INPUT_LENGTH, MAX_QUESTION_LENGTH, MAX_CONTEXT_LENGTH, HL_TOKEN
import torch
import copy
args = get_args()
class DataModule(pl.LightningDataModule):
    def __init__(self,args = get_args()):
        super().__init__()
        self.batch_size = 1
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
        tokenizer = self.tokenizer
        pad_token_id = self.tokenizer.pad_token_id

        if label is None:
            model_input = tokenizer(context,max_length=MAX_CONTEXT_LENGTH,truncation=True)
            return self.convert_to_tensor(model_input)

        context_input = tokenizer(context,add_special_tokens=True,max_length=MAX_CONTEXT_LENGTH,truncation=True)
        label_input = self.tokenizer(label,return_length=True,max_length=MAX_QUESTION_LENGTH,truncation=True,add_special_tokens=False)
        
        # limit context length
        model_input = {}
        model_input['input_ids'] = context_input['input_ids'] + label_input['input_ids'][:question_mask_position+1]
        label_id = model_input['input_ids'].pop(-1)
        model_input['input_ids'].append(self.tokenizer.mask_token_id)

        # print(self.tokenizer.decode([label_id]))

        # prepars lables
        model_input['labels'] = [-100]*len(model_input['input_ids'])
        model_input['labels'][-1] = label_id

        # print(len(model_input['labels']),model_input['labels'])

        # pad or limit to max length
        pad_ids = [pad_token_id]*MAX_INPUT_LENGTH
        pad_labels = [-100]*MAX_INPUT_LENGTH
        model_input['input_ids'] = (model_input['input_ids'] + pad_ids)[:MAX_INPUT_LENGTH] 
        model_input['labels'] = (model_input['labels'] + pad_labels)[:MAX_INPUT_LENGTH]

        # print(model_input)
        # print(tokenizer.decode(model_input['input_ids']))
        # print(tokenizer.decode([label_id]))
        # print('question_mask_position',question_mask_position)

        # for token_id,label_id in zip(model_input['input_ids'],model_input['labels']):
        #     if label_id not in [self.tokenizer.pad_token_id,-100]:
        #         print((self.tokenizer.decode([token_id]),self.tokenizer.decode([label_id])),end="\n")
        # print()

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