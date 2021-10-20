# Transformer QG on DRCD
The inputs of the model refers to 
```
we integrate C and A into a new C' in the following form.
C' = [c1, c2, ..., [HL], a1, ..., a|A|, [HL], ..., c|C|]
```
> Proposed by [Ying-Hong Chan & Yao-Chung Fan. (2019). A Re-current BERT-based Model for Question Generation.](https://www.aclweb.org/anthology/D19-5821/)

## Features
- Fully pipline from fine-tune to evaluation
- Support most of state of the art models
- Fast deploy as a API server

## DRCD dataset
[台達閱讀理解資料集 Delta Reading Comprehension Dataset (DRCD)](https://github.com/DRCKnowledgeTeam/DRCD) 屬於通用領域繁體中文機器閱讀理解資料集。 DRCD資料集從2,108篇維基條目中整理出10,014篇段落，並從段落中標註出30,000多個問題。

## Available models
- BART (base on **[uer/bart-base-chinese-cluecorpussmall](https://huggingface.co/uer/bart-base-chinese-cluecorpussmall)**)

## Use in Transformers
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("p208p2002/bart-drcd-qg-hl")

model = AutoModelForSeq2SeqLM.from_pretrained("p208p2002/bart-drcd-qg-hl")
```

## Expriments
Model             |Bleu 1|Bleu 2|Bleu 3|Bleu 4|METEOR|ROUGE-L|
------------------|------|------|------|------|------|-------|
BART-HLSQG        |34.25 |27.70 |22.43 |18.13 |23.58 |36.88  |

## Environment requirements
The hole development is based on Ubuntu system

1. If you don't have pytorch 1.6+ please install or update first
> https://pytorch.org/get-started/locally/

2. Install packages `pip install -r requirements.txt`

3. Setup scorer `python setup_scorer.py`

5. Download dataset `python init_dataset.py`


## Training
### Seq2Seq LM
```
usage: train_seq2seq_lm.py [-h]
                           [--base_model {facebook/bart-base,facebook/bart-large,t5-small,t5-base,t5-large}]
                           [-d {squad,squad-nqg}] [--epoch EPOCH] [--lr LR]
                           [--dev DEV] [--server] [--run_test]
                           [-fc FROM_CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  --base_model {facebook/bart-base,facebook/bart-large,t5-small,t5-base,t5-large}
  -d {squad,squad-nqg}, --dataset {squad,squad-nqg}
  --epoch EPOCH
  --lr LR
  --dev DEV
  --server
  --run_test
  -fc FROM_CHECKPOINT, --from_checkpoint FROM_CHECKPOINT
```

## Deploy
### Start up
```
python train_seq2seq_lm.py --server --base_model YOUR_BASE_MODEL --from_checkpoint FROM_CHECKPOINT
```
### Request example
```
curl --location --request POST 'http://127.0.0.1:5000/' \
--header 'Content-Type: application/x-www-form-urlencoded' \
--data-urlencode 'context=哈利·波特是英國作家[HL]羅琳[HL]撰寫的七部幻想小說系列'
```
```json
{"predict": "誰撰寫哈利·波特?"}
```

