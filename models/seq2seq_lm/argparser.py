import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',
            default='facebook/bart-base',
            choices=[
                'facebook/bart-base',
                'facebook/bart-large',
                't5-small',
                't5-base',
                't5-large',
                'p208p2002/bart-squad-qg-hl',
                'p208p2002/bart-squad-nqg-hl',
                'p208p2002/t5-squad-qg-hl',
                'p208p2002/t5-squad-nqg-hl'
                ],
            type=str
        )
    parser.add_argument('-d','--dataset',default='squad',choices=['squad','squad-nqg'],type=str)
    parser.add_argument('--epoch',default=10,type=int)
    parser.add_argument('--lr',type=float,default=5e-6)
    parser.add_argument('--dev',type=int,default=0)
    parser.add_argument('--server',action='store_true')
    parser.add_argument('--run_test',action='store_true')
    parser.add_argument('-fc','--from_checkpoint',type=str,default=None)
    args = parser.parse_args()
        
    return args