# Transformer QG on SQuAD
This project is a method reproduction of HLSQG with Transfomer models

Original HLSQG is based on BERT and proposed by [Ying-Hong Chan & Yao-Chung Fan. (2019). A Re-current BERT-based Model for Question Generation.](https://www.aclweb.org/anthology/D19-5821/)
The inputs of the model refers to 
```
we integrate C and A into a new C' in the following form.
C' = [c1, c2, ..., [HL], a1, ..., a|A|, [HL], ..., c|C|]
```

## Data setting
We report two dataset setting as Follow

### SQuAD
- train: 87599	
- validation: 10570

### SQuAD NQG
- train: 75722
- dev: 10570
- test: 11877

## Available models
- GPT2
- BART

## Install requirements
1. If you don't have pytorch please install first
> https://pytorch.org/get-started/locally/

2. Install packages `pip install -r requirements`

3. Setup scorer `python setup_scorer.py`

4. Download dataset `python init_dataset.py`

## Training
```
python train_xxx.py --help

usage: train_xxx.py [-h] [--base_model BASE_MODEL] [-d {squad,squad-nqg}]
                     [--batch_size BATCH_SIZE] [--epoch EPOCH] [--lr LR]
                     [--dev DEV] [--run_test] [-fc FROM_CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  --base_model BASE_MODEL
  -d {squad,squad-nqg}, --dataset {squad,squad-nqg}
  --batch_size BATCH_SIZE
  --epoch EPOCH
  --lr LR
  --dev DEV
  --run_test
  -fc FROM_CHECKPOINT, --from_checkpoint FROM_CHECKPOINT
```