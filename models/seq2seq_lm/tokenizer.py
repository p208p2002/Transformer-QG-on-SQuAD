from transformers import BertTokenizerFast
from .config import HL_TOKEN

def get_tokenizer(base_model):
    if 'tokenizer' not in globals():
        global tokenizer
        tokenizer = BertTokenizerFast.from_pretrained(base_model)
        # # add special token if needed
        # if tokenizer.pad_token is None:
        #     print('set pad_token...')
        #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # if tokenizer.sep_token is None:
        #     print('set sep_token...')
        #     tokenizer.add_special_tokens({'sep_token': '[SEP]'})
        # if tokenizer.eos_token is None:
        #     print('set eos_token...')
        #     tokenizer.add_special_tokens({'eos_token': '[EOS]'})
        # # add token here
        # tokenizer.add_tokens([HL_TOKEN],special_tokens=True)

    return tokenizer