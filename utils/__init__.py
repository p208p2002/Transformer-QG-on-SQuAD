from nlgeval import NLGEval
import transformers
import logging,os
os.environ["TOKENIZERS_PARALLELISM"] = 'true'
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)
import transformers
from bert_score import BERTScorer
import pytorch_lightning as pl
import json

class Scorer():
    def __init__(self):
        print("loading NLGEval...",end="\r")
        self.nlgeval = NLGEval(no_glove=True,no_skipthoughts=True)  # loads the models
        print("loading NLGEval...finish")
        print("loading BERTScorer...",end="\r")
        self.bert_scorer = BERTScorer(lang="en")
        print("loading BERTScorer...finish")

    def compute_score(self,hyp,refs):
        # token scores    
        score = self.nlgeval.compute_individual_metrics(hyp=hyp, ref=refs)
        del score['CIDEr']

        # bert score
        bP, bR, bF1 = self.bert_scorer.score([hyp], [refs])
        score['BertScore'] = bF1.item() if bF1.item() > 0.0 else 0.0

        return score
    
class ModelEvalMixin():
    def test_epoch_end(self, outputs):
        scorer = Scorer()
        log_dir = os.path.join(self.trainer.default_root_dir,'dev') if self.trainer.log_dir is None else self.trainer.log_dir
        
        # load predict and add score
        with open(os.path.join(log_dir,'predict.jsonl'),'r',encoding='utf-8') as log_f,open(os.path.join(log_dir,'predict_with_score.jsonl'),'w',encoding='utf-8') as score_f:
            lines = log_f.readlines()
            for i,line in enumerate(lines):
                line = json.loads(line)
                hyp = line['hyp']
                ref = line['ref']
                score = scorer.compute_score(hyp,[ref])
                for k in score.keys(): score[k] = str(score[k])
                line['score'] = score
                score_f.write(json.dumps(line)+"\n")

        # sum score
        sum_score = {}
        with open(os.path.join(log_dir,'predict_with_score.jsonl'),'r',encoding='utf-8') as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                line = json.loads(line)
                data_socres = line['score']                    
                if i == 0:
                    for socre_key in data_socres.keys():
                        sum_score[socre_key] = 0.0 # init dict
                for socre_key in data_socres.keys():
                    sum_score[socre_key] += float(data_socres[socre_key])
        
        with open(os.path.join(log_dir,'model_score.txt'),'w',encoding='utf-8') as f:
            print('total data:',len(lines))
            for key in sum_score.keys():
                key_score = sum_score[key]/len(lines)
                print(key,key_score)
                f.write("%s: %3.5f\n"%(key,key_score*100))
