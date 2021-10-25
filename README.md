# Transformer QG on SQuAD [![Auto-Build](https://github.com/p208p2002/Transformer-QG-on-SQuAD/actions/workflows/main.yml/badge.svg)](https://github.com/p208p2002/Transformer-QG-on-SQuAD/actions/workflows/main.yml)
<span>
<a target="_blank" href="https://github.com/p208p2002/Transformer-QG-on-SQuAD">
<img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white">
</a>

<a target="_blank" href="https://hub.docker.com/repository/docker/p208p2002/transformer-qg-on-squad">
<img src="https://img.shields.io/badge/Docker Hub-2CA5E0?style=for-the-badge&logo=docker&logoColor=white">
</a>

<a target="_blank" href="https://huggingface.co/p208p2002/bart-squad-qg-hl">
<img src="https://img.shields.io/badge/🤗 HF Model Hub-ffea00?style=for-the-badge&logoColor=white">
</a>
</span>

The inputs of the model refers to 
```
we integrate C and A into a new C' in the following form.
C' = [c1, c2, ..., [HL], a1, ..., a|A|, [HL], ..., c|C|]
```
> Proposed by [Ying-Hong Chan & Yao-Chung Fan. (2019). A Re-current BERT-based Model for Question Generation.](https://www.aclweb.org/anthology/D19-5821/)

## Overview
### Features
- Fully pipline from fine-tune to evaluation
- Support most of state of the art models
- Fast deploy as a API server

### Data setting
We report two dataset setting as Follow

#### SQuAD
- train: 87599	
- validation: 10570
> [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250)

#### SQuAD NQG
- train: 75722
- dev: 10570
- test: 11877
> [Learning to Ask: Neural Question Generation for Reading Comprehension](https://arxiv.org/abs/1705.00106)

### Available models
- BART
- GPT2
- T5

### Expriments
We report score with `NQG Scorer` which is using in SQuAD NQG.

If not special explanation, the size of the model defaults to "base".

### SQuAD
Model                            |Bleu 1|Bleu 2|Bleu 3|Bleu 4|METEOR|ROUGE-L|
---------------------------------|------|------|------|------|------|-------|
BART-HLSQG                       |54.67 |39.26 |30.34 |24.15 |25.43 |52.64  |
GPT2-HLSQG                       |49.31 |33.95 |25.41| 19.69 |22.29 |48.82  |
T5-HLSQG                         |54.29 |39.22 |30.43 |24.26 |25.56 |53.11  |

### SQuAD NQG
Model                            |Bleu 1|Bleu 2|Bleu 3|Bleu 4|METEOR|ROUGE-L|
---------------------------------|------|------|------|------|------|-------|
BERT-HLSQG (Chan et al.)         |49.73 |34.60 |26.13 |20.33 |23.88 |48.23  |
BART-HLSQG                       |54.12 |38.19 |28.84 |22.35 |24.55 |51.03  |
GPT2-HLSQG                       |49.82 |33.69 |24.71 |18.63 |21.90 |47.60  |
T5-HLSQG                         |53.13 |37.60 |28.62 |22.38 |24.48 |51.20  |


## Using with Transformers
#### bart-squad-qg-hl
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("p208p2002/bart-squad-qg-hl")

model = AutoModelForSeq2SeqLM.from_pretrained("p208p2002/bart-squad-qg-hl")
```
#### t5-squad-qg-hl
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("p208p2002/t5-squad-qg-hl")

model = AutoModelForSeq2SeqLM.from_pretrained("p208p2002/t5-squad-qg-hl")
```

and more on [HF Model Hub](https://huggingface.co/p208p2002)!

## Run as API server
### Using docker (recommend)
```sh
docker run -it -p 5000:5000 p208p2002/transformer-qg-on-squad:lastest train_seq2seq_lm.py --server --base_model p208p2002/bart-squad-qg-hl
```

### From your own checkpoint
```sh
python train_xxx_lm.py --server --base_model YOUR_BASE_MODEL --from_checkpoint FROM_CHECKPOINT
```
### Request example
```sh
curl --location --request POST 'http://127.0.0.1:5000/' \
--header 'Content-Type: application/x-www-form-urlencoded' \
--data-urlencode 'context=Harry Potter is a series of seven fantasy novels written by [HL] J. K. Rowling. [HL]'
```
```json
{"predict": "Who wrote the books?"}
```

## Training from scratch
### Environment setup
The hole development is based on Ubuntu system

1. If you don't have pytorch 1.6+ please install or update first
> https://pytorch.org/get-started/locally/

2. Install packages `pip install -r requirements.txt`

3. Setup scorer `python setup_scorer.py`

5. Download dataset `python init_dataset.py`

### Seq2Seq LM
```
usage: train_seq2seq_lm.py [-h]
                           [--base_model {facebook/bart-base,facebook/bart-large,t5-small,t5-base,t5-large,p208p2002/bart-squad-qg-hl,p208p2002/bart-squad-nqg-hl,p208p2002/t5-squad-qg-hl,p208p2002/t5-squad-nqg-hl}]
                           [-d {squad,squad-nqg}] [--epoch EPOCH] [--lr LR]
                           [--dev DEV] [--server] [--run_test]
                           [-fc FROM_CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  --base_model {facebook/bart-base,facebook/bart-large,t5-small,t5-base,t5-large,p208p2002/bart-squad-qg-hl,p208p2002/bart-squad-nqg-hl,p208p2002/t5-squad-qg-hl,p208p2002/t5-squad-nqg-hl}
  -d {squad,squad-nqg}, --dataset {squad,squad-nqg}
  --epoch EPOCH
  --lr LR
  --dev DEV
  --server
  --run_test
  -fc FROM_CHECKPOINT, --from_checkpoint FROM_CHECKPOINT
```

### Causal LM
```
usage: train_causal_lm.py [-h]
                          [--base_model {gpt2,gpt2-large,p208p2002/gpt2-squad-qg-hl,p208p2002/gpt2-squad-nqg-hl}]
                          [-d {squad,squad-nqg}] [--epoch EPOCH] [--lr LR]
                          [--dev DEV] [--server] [--run_test]
                          [-fc FROM_CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  --base_model {gpt2,gpt2-large,p208p2002/gpt2-squad-qg-hl,p208p2002/gpt2-squad-nqg-hl}
  -d {squad,squad-nqg}, --dataset {squad,squad-nqg}
  --epoch EPOCH
  --lr LR
  --dev DEV
  --server
  --run_test
  -fc FROM_CHECKPOINT, --from_checkpoint FROM_CHECKPOINT
```
