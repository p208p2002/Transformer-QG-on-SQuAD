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
            log_f.write(json.dumps({"hyp":decode_question,"ref":ref_question},ensure_ascii=False)+"\n")

        # write log for nqg scorer
        with open(os.path.join(log_dir,'predict_for_nqg_scorer.txt'),'a',encoding='utf-8') as log_f:
            decode_question = decode_question.lower()
            decode_question = decode_question.replace("?"," ?")
            decode_question = decode_question.replace(","," ,")
            decode_question = decode_question.replace("'s"," 's")
            decode_question = decode_question.replace("..."," ...")
            
            # replace string: "hello" world -> `` hello '' world
            decode_question = re.sub(' "','``',decode_question)
            decode_question = re.sub('"',"''",decode_question)
            decode_question = decode_question.replace('``',' `` ')
            decode_question = decode_question.replace("''"," ''")
            log_f.write(decode_question+"\n")
        
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
        
        # ngq scorer
        assert os.path.isdir('nqg'),'nqg scorer is not detect, please check "README.md" for help'
        nqg_predict_file_path = os.path.join(log_dir,'predict_for_nqg_scorer.txt')
        nqg_predict_score_out_path = os.path.join(log_dir,'nqg_scorer.txt')
        if dataset == 'squad-nqg':
            os.system('python nqg/qgevalcap/eval.py --src nqg/data/processed/src-test.txt --tgt nqg/data/processed/tgt-test.txt --out %s >> %s'%(nqg_predict_file_path,nqg_predict_score_out_path))
        elif dataset == 'squad':
            os.system('python nqg/qgevalcap/eval.py --src nqg/data/processed/src-dev.txt --tgt nqg/data/processed/tgt-dev.txt --out %s >> %s'%(nqg_predict_file_path,nqg_predict_score_out_path))
        
        print("see log_dir:`%s` for full report"%log_dir)
    
    def save_huggingface_model(self):
        log_dir = os.path.join(self.trainer.default_root_dir,'dev') if self.trainer.log_dir is None else self.trainer.log_dir
        log_dir = os.path.join(log_dir,'huggingface_model')
        os.makedirs(log_dir,exist_ok=True)
        self.model.save_pretrained(log_dir)
        self.tokenizer.save_pretrained(log_dir)
