import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',default='/user_data/zh-seq2seq/.log_seq2seq_lm/lightning_logs/version_4/huggingface_model',choices=['bert-base-chinese'],type=str)
    parser.add_argument('-d','--dataset',default='drcd',choices=['drcd'],type=str)
    parser.add_argument('--batch_size',type=int,default=10)
    parser.add_argument('--epoch',default=10,type=int)
    parser.add_argument('--lr',type=float,default=5e-6)
    parser.add_argument('--dev',type=int,default=0)
    parser.add_argument('--server',action='store_true')
    parser.add_argument('--run_test',action='store_true')
    parser.add_argument('-fc','--from_checkpoint',type=str,default=None)
    args = parser.parse_args()
        
    return args