import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',default='facebook/bart-base',type=str)
    parser.add_argument('-d','--dataset',default='squad',choices=['squad','squad-nqg'],type=str)
    parser.add_argument('--batch_size',default=8,type=int)
    parser.add_argument('--epoch',default=6,type=int)
    parser.add_argument('--lr',type=float,default=5e-5)
    parser.add_argument('--dev',type=int,default=0)
    parser.add_argument('--run_test',action='store_true')
    parser.add_argument('-fc','--from_checkpoint',type=str,default=None)
    args = parser.parse_args()
        
    return args