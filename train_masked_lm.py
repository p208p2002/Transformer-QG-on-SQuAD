import pytorch_lightning as pl
from models.masked_lm import argparser
from models.masked_lm.model import Model
from models.masked_lm.data_module import DataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from models.masked_lm.config import GPUS,ACCELERATOR


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
        precision=16,
        default_root_dir='.log_masked_lm',
        max_epochs=args.epoch,
        callbacks=[
            EarlyStopping(monitor='dev_loss',patience=5),
            ModelCheckpoint(monitor='dev_loss',filename='{epoch}-step={step}-{dev_loss:.2f}',save_last=True,save_top_k=3)
        ]
    )
    
    # DataModule
    dm = DataModule()
    
    # train
    if args.run_test == False: 
        trainer.fit(model,datamodule=dm)

    # decide which checkpoint to use
    last_model_path = trainer.checkpoint_callback.last_model_path
    best_model_path = trainer.checkpoint_callback.best_model_path
    _use_model_path = last_model_path if best_model_path == "" else best_model_path
    print('use checkpoint:',_use_model_path)

    # run_test
    trainer.test(
        model=model if _use_model_path == "" else None,
        datamodule=dm,
        ckpt_path=_use_model_path
    )
    