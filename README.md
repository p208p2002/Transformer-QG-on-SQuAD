# Transformer QG on SQuAD
The inputs of the model refers to 
```
we integrate C and A into a new C' in the following form.
C' = [c1, c2, ..., [HL], a1, ..., a|A|, [HL], ..., c|C|]
```
> Proposed by [Ying-Hong Chan & Yao-Chung Fan. (2019). A Re-current BERT-based Model for Question Generation.](https://www.aclweb.org/anthology/D19-5821/)

## Data setting
We report two dataset setting as Follow

### SQuAD
- train: 87599	
- validation: 10570
> [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250)

### SQuAD NQG
- train: 75722
- dev: 10570
- test: 11877
> [Learning to Ask: Neural Question Generation for Reading Comprehension](https://arxiv.org/abs/1705.00106)

## Available models
- ALBERT
- BART
- BERT
- GPT2
- RoBERTa
- T5

## Expriments
`NQG Scorer` which using in SQuAD NQG, has a [special pre-porcessing](https://github.com/xinyadu/nqg/issues/21) action before evaluation.
For `Our Scorer`, we first `lower()` and `tokenize()` in glod labels and predcit sentences, then we evaluation.
> see `utils/scorer.py` for implement detail

### SQuAD
#### NQG Scorer
Model                            |Bleu 1|Bleu 2|Bleu 3|Bleu 4|METEOR|ROUGE-L|
---------------------------------|------|------|------|------|------|-------|
BERT-HLSQG (ours)|
RoBERTa-HLSQG|
BART-HLSQG                       |54.67 |39.26 |30.34 |24.15 |25.43 |52.64  |
GPT2-HLSQG|
#### Our Scorer
Model                            |Bleu 1|Bleu 2|Bleu 3|Bleu 4|METEOR|ROUGE-L|
---------------------------------|------|------|------|------|------|-------|
BERT-HLSQG (ours)|
RoBERTa-HLSQG|
BART-HLSQG                       |47.23 |31.44 |22.08 |15.38 |27.38 |51.18  |
GPT2-HLSQG|

### SQuAD NQG
#### NQG Scorer
Model                            |Bleu 1|Bleu 2|Bleu 3|Bleu 4|METEOR|ROUGE-L|
---------------------------------|------|------|------|------|------|-------|
BERT-HLSQG (Chan et al.)         |49.73 |34.60 |26.13 |20.33 |23.88 |48.23  |
BERT-HLSQG (ours)|
RoBERTa-HLSQG|
BART-HLSQG                       |54.12 |38.19 |28.84 |22.35 |24.55 |51.03  |
GPT2-HLSQG                       |49.82 |33.69 |24.71 |18.63 |21.90 |47.60  |

#### Our Scorer
Model     |Bleu 1|Bleu 2|Bleu 3|Bleu 4|METEOR|ROUGE-L|
----------|------|------|------|------|------|-------|
BERT-HLSQG (ours)|
RoBERTa-HLSQG|
BART-HLSQG|44.27 |28.01 |18.72 |12.38 |25.70 |48.23  |
GPT2-HLSQG|


## Install requirements
1. If you don't have pytorch 1.6+ please install or update first
> https://pytorch.org/get-started/locally/

2. Install packages `pip install -r requirements.txt`

3. Setup scorer `python setup_scorer.py`

5. Download dataset `python init_dataset.py`

> It's recommend to run script with my pre-build docker image - [docker-for-ai-dev](https://github.com/p208p2002/docker-for-ai-dev)

## Training
### Seq2Seq LM
```
usage: train_seq2seq_lm.py [-h]
                           [--base_model {facebook/bart-base,facebook/bart-large,t5-small,t5-base,t5-large}]
                           [-d {squad,squad-nqg}] [--epoch EPOCH] [--lr LR]
                           [--dev DEV] [--run_test] [-fc FROM_CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  --base_model {facebook/bart-base,facebook/bart-large,t5-small,t5-base,t5-large}
  -d {squad,squad-nqg}, --dataset {squad,squad-nqg}
  --epoch EPOCH
  --lr LR
  --dev DEV
  --run_test
  -fc FROM_CHECKPOINT, --from_checkpoint FROM_CHECKPOINT
```

### Causal LM
```
usage: train_causal_lm.py [-h] [--base_model {gpt2,gpt2-large}]
                          [-d {squad,squad-nqg}] [--epoch EPOCH] [--lr LR]
                          [--dev DEV] [--run_test] [-fc FROM_CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  --base_model {gpt2,gpt2-large}
  -d {squad,squad-nqg}, --dataset {squad,squad-nqg}
  --epoch EPOCH
  --lr LR
  --dev DEV
  --run_test
  -fc FROM_CHECKPOINT, --from_checkpoint FROM_CHECKPOINT
```

### Masked LM
```
usage: train_masked_lm.py [-h]
                          [--base_model {bert-base-uncased,bert-large-uncased,roberta-base,roberta-large,albert-base-v1,albert-large-v1,albert-base-v2,albert-large-v2}]
                          [-d {squad,squad-nqg}] [--batch_size BATCH_SIZE]
                          [--epoch EPOCH] [--lr LR] [--dev DEV] [--run_test]
                          [-fc FROM_CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  --base_model {bert-base-uncased,bert-large-uncased,roberta-base,roberta-large,albert-base-v1,albert-large-v1,albert-base-v2,albert-large-v2}
  -d {squad,squad-nqg}, --dataset {squad,squad-nqg}
  --batch_size BATCH_SIZE
  --epoch EPOCH
  --lr LR
  --dev DEV
  --run_test
  -fc FROM_CHECKPOINT, --from_checkpoint FROM_CHECKPOINT
```