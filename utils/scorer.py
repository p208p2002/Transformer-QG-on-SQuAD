from nlgeval import NLGEval
from collections import defaultdict
import os
import stanza

class Scorer():
    def __init__(self,preprocess=True):
        self.preprocess = preprocess
        self.nlgeval = NLGEval(no_glove=True,no_skipthoughts=True,metrics_to_omit=["CIDEr"])
        self.score = defaultdict(lambda : 0.0)
        self.len = 0
        if self.preprocess:
            self.nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True, verbose=False)
    
    def _preprocess(self,raw_sentence):
        result = self.nlp(raw_sentence.replace("\n\n",""))
        tokens = []
        try:
            for token in result.sentences[0].tokens:
                tokens.append(token.text.lower())
                tokenize_sentence = ' '.join(tokens)
        except:
            print('_preprocess fail, return ""\n',raw_sentence,result)
            return ""
        return tokenize_sentence
        
    def add(self,hyp,refs):
        refs = refs[:]
        if self.preprocess:
            hyp = self._preprocess(hyp)
            refs = [self._preprocess(ref) for ref in refs]
        score = self.nlgeval.compute_individual_metrics(hyp=hyp, ref=refs)
        for score_key in score.keys():
            self.score[score_key] += score[score_key]
        self.len += 1

    def compute(self,save_score_report_path=None):
        if save_score_report_path is not None:
            os.makedirs(save_score_report_path,exist_ok=True)
            score_f = open(os.path.join(save_score_report_path,'our_scorer.txt'),'w',encoding='utf-8')
        for score_key in self.score.keys():
            _score = self.score[score_key]/self.len
            if save_score_report_path is not None:
                score_f.write("%s\t%3.5f\n"%(score_key,_score))