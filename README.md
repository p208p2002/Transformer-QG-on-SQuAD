# SQuAD HLSQG
This project is a method reproduction of HLSQG with Transfomer models

Original HLSQG is based on BERT and proposed by [Ying-Hong Chan & Yao-Chung Fan. (2019). A Re-current BERT-based Model for Question Generation.](https://www.aclweb.org/anthology/D19-5821/)
The inputs of the model refers to 
```
we integrate C and A into a new C' in the following form.
C' = [c1, c2, ..., [HL], a1, ..., a|A|, [HL], ..., c|C|]
```

## Data setting
We report two dataset split setting as Follow

### SQuAD
- train: 87599	
- validation: 10570

### SQuAD 73K
- train: 73240
- dev: 11877
- test: 11877

## Available Models
- GPT2
- BART

## Install requirements
1. If you don't have pytorch please install first
> https://pytorch.org/get-started/locally/

2. Install packages `pip install -Ur requirements`

3. Setup scorer `python setup_scorer.py`

4. The model and SQuAD dataset will automatic download in first time training

## How to run scripts
```
python train_xxx.py --help
```