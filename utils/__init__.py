import json
import re
import os
from .scorer import Scorer
import torch
import torch.nn as nn

class ModelEvalMixin():
    def write_predict(self, decode_question, ref_question):
        #
        log_dir = os.path.join(self.trainer.default_root_dir,'dev') if self.trainer.log_dir is None else self.trainer.log_dir
        os.makedirs(log_dir,exist_ok=True)

        # write log for my scorer
        with open(os.path.join(log_dir,'predict.jsonl'),'a',encoding='utf-8') as log_f:
            log_f.write(json.dumps({"hyp":decode_question.replace("?","？"),"ref":' '.join(ref_question)},ensure_ascii=False)+"\n")
        
    def evaluate_predict(self,dataset):
        scorer = Scorer()
        log_dir = os.path.join(self.trainer.default_root_dir,'dev') if self.trainer.log_dir is None else self.trainer.log_dir
        
        # load predict and add score
        with open(os.path.join(log_dir,'predict.jsonl'),'r',encoding='utf-8') as log_f:
            lines = log_f.readlines()
            for i,line in enumerate(lines):
                line = json.loads(line)
                hyp = line['hyp']
                ref = line['ref']
                scorer.add(hyp,[ref])
            scorer.compute(save_score_report_path=log_dir)
        
        print("see log_dir:`%s` for full report"%log_dir)
    
    def save_huggingface_model(self):
        log_dir = os.path.join(self.trainer.default_root_dir,'dev') if self.trainer.log_dir is None else self.trainer.log_dir
        log_dir = os.path.join(log_dir,'huggingface_model')
        os.makedirs(log_dir,exist_ok=True)
        self.model.save_pretrained(log_dir)
        self.tokenizer.save_pretrained(log_dir)

class MaskedLMGenerator():
    def __init__(self,model,tokenizer,is_bert=False):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id = self.tokenizer.mask_token_id
        self.eos_token_id = self.tokenizer.sep_token_id
        self.is_bert = is_bert
        
    def generate(self,input_ids,max_length=512):
        current_length = torch.numel(input_ids)
        context_length = current_length
        assert current_length < max_length
        input_ids = input_ids.view(current_length).tolist()
        gen_step_count = max_length - current_length
        input_ids.append(self.mask_token_id)
        for i in range(gen_step_count):
            if self.is_bert:
                logits = self.model(
                    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.model.device),
                    token_type_ids = torch.LongTensor([0]*context_length + [1]*(i+1)).unsqueeze(0).to(self.model.device),
                    return_dict=True
                ).logits
            else:
                logits = self.model(torch.LongTensor(input_ids).unsqueeze(0).to(self.model.device),return_dict=True).logits
            
            last_token_logits = logits[0,-1,:] # shape: [vacab_size]
            decode_id = torch.argmax(last_token_logits,dim=-1).item()
            assert input_ids.pop(-1) == self.mask_token_id
            if decode_id == self.eos_token_id or i == (gen_step_count-1):
                return self.tokenizer.decode(input_ids[context_length:])
            input_ids.append(decode_id)
            input_ids.append(self.mask_token_id)
