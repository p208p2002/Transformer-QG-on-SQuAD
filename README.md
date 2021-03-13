# SQuAD QG
Transfromer QG models on SQuAD datset

The inputs of the model refers to HLSQG proposed by [Ying-Hong Chan & Yao-Chung Fan. (2019). A Re-current BERT-based Model for Question Generation.](https://www.aclweb.org/anthology/D19-5821/)
```
we integrate C and A into a new C' in the following form.
C' = [c1, c2, ..., [HL], a1, ..., a|A|, [HL], ..., c|C|]
```

## Available Models
- GPT2
- BART

## Install requirements
1. If you don't have pytorch please install first
> https://pytorch.org/get-started/locally/

2. Install packages `pip install -Ur requirements`

3. Setup scorer `python setup_scorer.py`

## How to run scripts
```
python train_xxx.py --help
```