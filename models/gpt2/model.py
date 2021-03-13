import pytorch_lightning as pl
from transformers import AutoModelForCausalLM
from .tokenizer import get_tokenizer
from .argparser import get_args
import torch
import re
import os
import json
from .config import MAX_CONTEXT_LENGTH
from utils import Scorer
args = get_args()


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = get_tokenizer(args.base_model)
        self.model = AutoModelForCausalLM.from_pretrained(args.base_model)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.batch_size = args.batch_size # set `batch_size` attr for auto-find batch size in pl

    def forward(self, input_ids,labels=None):
        return self.model(input_ids=input_ids,labels=labels,return_dict=True)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch[0],batch[1])
        loss = outputs['loss']
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch[0],batch[1])
        loss = outputs['loss']
        self.log('dev_loss',loss)
        return loss
    
    def on_test_epoch_start(self):
        self.scorer = Scorer()
        
    def test_step(self, batch, batch_idx):
        input_ids = batch[0]
        ref_question = batch[1][0]
        input_ids_len = input_ids.shape[-1]
        batch_size = input_ids.shape[0]
        assert batch_size == 1

        num_return_sequences = 1
        sample_outputs = self.model.generate(
            input_ids = input_ids,
            max_length=MAX_CONTEXT_LENGTH,
            early_stopping=True,
            temperature=0.85,
            do_sample=True,
            top_p=0.9,
            top_k=10,
            num_beams=3,
            no_repeat_ngram_size=5,
            num_return_sequences=num_return_sequences,
            pad_token_id = self.tokenizer.pad_token_id,
            eos_token_id = self.tokenizer.eos_token_id
        )

        assert len(sample_outputs) == num_return_sequences # 1
        sample_output = sample_outputs[0]        
        decode_question = self.tokenizer.decode(sample_output[input_ids_len:], skip_special_tokens=True)
        score = self.scorer.compute_score(decode_question,[ref_question])

        # log
        log_dir = os.path.join(self.trainer.default_root_dir,'dev') if self.trainer.log_dir is None else self.trainer.log_dir
        os.makedirs(log_dir,exist_ok=True)
        with open(os.path.join(log_dir,'predict.jsonl'),'a',encoding='utf-8') as log_f:
            for k in score.keys(): score[k] = str(score[k])
            log_f.write(json.dumps({"hyp":decode_question,"ref":ref_question,"score":score})+"\n")
    
    def test_epoch_end(self, outputs):
        sum_score = {}
        log_dir = os.path.join(self.trainer.default_root_dir,'dev') if self.trainer.log_dir is None else self.trainer.log_dir
        with open(os.path.join(log_dir,'predict.jsonl'),'r',encoding='utf-8') as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                line = json.loads(line)
                data_socres = line['score']                    
                if i == 0:
                    for socre_key in data_socres.keys():
                        sum_score[socre_key] = 0.0 # init dict
                for socre_key in data_socres.keys():
                    sum_score[socre_key] += float(data_socres[socre_key])
        
        print('total data:',len(lines))
        for key in sum_score.keys():
            print(key,sum_score[key]/len(lines))
                
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=args.lr)