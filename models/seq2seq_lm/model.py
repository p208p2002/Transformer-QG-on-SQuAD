import pytorch_lightning as pl
from transformers import AutoModelForSeq2SeqLM,EncoderDecoderModel
from .tokenizer import get_tokenizer
from .argparser import get_args
import torch
import re
import os
import json
from .config import MAX_INPUT_LENGTH
from utils import ModelEvalMixin
from utils.server import ServerMixin
from utils.scheduler import setup_scheduler,step_scheduler
args = get_args()

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class Model(pl.LightningModule,ModelEvalMixin,ServerMixin):
    def __init__(self,args = args):
        super().__init__()
        self.save_hyperparameters(args)
        #
        args = get_args()
        self.tokenizer = get_tokenizer(args.base_model)
        # self.model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-chinese-hl/huggingface_model', "gpt2-base-chinese-hl/huggingface_model")
        # self.model = EncoderDecoderModel.from_pretrained(args.base_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
        self._type = 'seq2seq_lm'
        # print(self.model.config.decoder)
        # exit()

    def forward(self, input_ids,attention_mask,labels=None):
        # decoder_input_ids = shift_tokens_right(
        #     labels, self.model.encoder.config.pad_token_id, self.model.decoder.config.bos_token_id
        # )
        return self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels,return_dict=True)
    
    # @step_scheduler
    def training_step(self, batch, batch_idx):
        outputs = self(batch[0],batch[1],batch[2])
        loss = outputs['loss']
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch[0],batch[1],batch[2])
        loss = outputs['loss']
        self.log('dev_loss',loss)
        
    def test_step(self, batch, batch_idx):
        input_ids = batch[0]
        attention_mask = batch[1]
        ref_question = batch[2][0]
        input_ids_len = input_ids.shape[-1]
        batch_size = input_ids.shape[0]
        assert batch_size == 1

        num_return_sequences = 1
        sample_outputs = self.model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_length=MAX_INPUT_LENGTH,
            early_stopping=True,
            temperature=0.85,
            do_sample=True,
            top_p=0.9,
            top_k=10,
            num_beams=3,
            no_repeat_ngram_size=5,
            num_return_sequences=num_return_sequences,
            eos_token_id=102
        )

        assert len(sample_outputs) == num_return_sequences # 1
        sample_output = sample_outputs[0]        
        decode_question = self.tokenizer.decode(sample_output, skip_special_tokens=True)
        self.write_predict(decode_question,ref_question)

    def test_epoch_end(self,outputs):
        self.evaluate_predict(dataset=args.dataset)
        self.save_huggingface_model()
    
    # @setup_scheduler
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=args.lr)