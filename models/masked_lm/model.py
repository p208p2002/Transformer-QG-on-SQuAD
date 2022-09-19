import pytorch_lightning as pl
from transformers import AutoModelForMaskedLM
from .tokenizer import get_tokenizer
from .argparser import get_args
import torch
import re
from utils import ModelEvalMixin
from utils.decoder import BeamSearchForMaskedLM
from utils.server import ServerMixin 
args = get_args()

class Model(pl.LightningModule,ModelEvalMixin,ServerMixin):
    def __init__(self,args = args):
        super().__init__()
        self.save_hyperparameters(args)
        #
        args = get_args()
        self.tokenizer = get_tokenizer(args.base_model)
        self.model = AutoModelForMaskedLM.from_pretrained(args.base_model)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self._type = 'masked_lm'

    def forward(self, input_ids,labels=None):
        return self.model(input_ids=input_ids,labels=labels,return_dict=True)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch[0],batch[1])
        loss = outputs['loss']
        self.log('train_loss',loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        outputs = self(batch[0],batch[1])
        loss = outputs['loss']
        self.log('dev_loss',loss)
        
    def test_step(self, batch, batch_idx):
        input_ids = batch[0]
        ref_question = batch[1][0]
        # input_ids_len = input_ids.shape[-1]
        batch_size = input_ids.shape[0]
        assert batch_size == 1

        decoder = BeamSearchForMaskedLM(self.model,self.tokenizer,beam_size=3,max_token_length=450,device='cuda')
        decode_question = decoder(input_ids)
        print(decode_question)
        self.write_predict(decode_question,ref_question)

    def test_epoch_end(self,outputs):
        self.evaluate_predict(dataset=args.dataset)
        self.save_huggingface_model()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=args.lr)
        opt.zero_grad()
        return opt