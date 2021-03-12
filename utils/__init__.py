from nlgeval import NLGEval
import transformers
import logging,os
os.environ["TOKENIZERS_PARALLELISM"] = 'true'
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)
import transformers
from bert_score import BERTScorer

class Scorer():
    def __init__(self):
        print("loading NLGEval...",end="")
        self.nlgeval = NLGEval(no_glove=True,no_skipthoughts=True)  # loads the models
        print("finish")
        print("loading BERTScorer...",end="")
        self.bert_scorer = BERTScorer(lang="en")
        print("finish")

    def compute_score(self,hyp,refs):
        # token scores    
        score = self.nlgeval.compute_individual_metrics(hyp=hyp, ref=refs)
        
        del score['CIDEr']

        # bert score
        bP, bR, bF1 = self.bert_scorer.score([hyp], [refs])
        score['BertScore'] = bF1.item() if bF1.item() > 0.0 else 0.0


        for k in score.keys(): score[k] = str(score[k])

        return score