import pytorch_lightning as pl
from models.gpt2 import argparser
from models.gpt2.model import Model
from models.gpt2.data_module import DataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from models.gpt2.config import GPUS,ACCELERATOR
args = argparser.get_args()

if __name__ == "__main__":
   
    trainer = pl.Trainer(
        gpus=GPUS,
        accelerator=ACCELERATOR,
        fast_dev_run=args.dev,
        precision=32,
        default_root_dir='.log_gpt2',
        max_epochs=args.epoch,
        callbacks=[
            EarlyStopping(monitor='dev_loss',patience=2),
            ModelCheckpoint(monitor='dev_loss',filename='{epoch}-{dev_loss:.2f}',save_last=True),
        ]
    )

    dm = DataModule()

    if args.from_checkpoint is None:
        model = Model()
    else:
        print('load from checkpoint')
        model = Model.load_from_checkpoint(args.from_checkpoint)

    if args.run_test == False:
        trainer.fit(model,datamodule=dm)

    # select which ckpt to use
    last_model_path = trainer.checkpoint_callback.last_model_path
    best_model_path = trainer.checkpoint_callback.best_model_path
    if args.from_checkpoint is not None: # form checkpoint
        _use_model_path = None
    else:
        _use_model_path = last_model_path if best_model_path == "" else best_model_path
        assert _use_model_path != ""
        print('use checkpoint',_use_model_path)
    trainer.test(model if args.run_test else None,datamodule=dm,ckpt_path=_use_model_path)
    