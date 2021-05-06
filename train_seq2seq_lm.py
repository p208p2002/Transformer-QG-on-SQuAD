import pytorch_lightning as pl
from models.seq2seq_lm import argparser
from models.seq2seq_lm.model import Model
from models.seq2seq_lm.data_module import DataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from models.seq2seq_lm.config import GPUS,ACCELERATOR
from copy import deepcopy
import torch
args = argparser.get_args()

if __name__ == "__main__":
    # load model from_checkpoint or init a new one 
    if args.from_checkpoint is None:
        model = Model()
    else:
        print('load from checkpoint')
        model = Model.load_from_checkpoint(args.from_checkpoint)
    
    # run as a flask api server
    if args.server:
        model.run_server()
        exit()
    
    # trainer config
    trainer = pl.Trainer(
        gpus=GPUS,
        accelerator=ACCELERATOR,
        fast_dev_run=args.dev,
        precision=32,
        default_root_dir='.log_seq2seq_lm',
        max_epochs=args.epoch,
        callbacks=[
            EarlyStopping(monitor='dev_loss',patience=5),
            ModelCheckpoint(monitor='dev_loss',filename='{epoch}-{dev_loss:.2f}',save_last=True),
        ]
    )
    
    # DataModule
    dm = DataModule()
    
    # train
    if args.run_test == False:
        trainer.fit(model,datamodule=dm)

    # # decide which checkpoint to use
    # last_model_path = trainer.checkpoint_callback.last_model_path
    # best_model_path = trainer.checkpoint_callback.best_model_path
    # _use_model_path = last_model_path if best_model_path == "" else best_model_path
    # print('use checkpoint:',_use_model_path)

    # run_test
    trainer.test(
        model=model,
        datamodule=dm,
        # ckpt_path=_use_model_path
    )
        
    
    