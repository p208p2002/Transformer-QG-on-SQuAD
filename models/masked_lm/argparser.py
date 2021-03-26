import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',default='bert-base-uncased',choices=['bert-base-uncased','bert-large-uncased','roberta-base','roberta-large'],type=str)
    parser.add_argument('-d','--dataset',default='squad',choices=['squad','squad-nqg'],type=str)
    parser.add_argument('--epoch',default=12,type=int)
    parser.add_argument('--lr',type=float,default=5e-6)
    parser.add_argument('--dev',type=int,default=0)
    parser.add_argument('--run_test',action='store_true')
    parser.add_argument('-fc','--from_checkpoint',type=str,default=None)
    args = parser.parse_args()
        
    return args