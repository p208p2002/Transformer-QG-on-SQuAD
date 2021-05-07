from transformers import AutoModel,AutoModelForCausalLM
from transformers import AutoTokenizer
import os

HL_TOKEN = '[HL]'
def get_tokenizer(base_model):
    if 'tokenizer' not in globals():
        global tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        # add special token if needed
        if tokenizer.pad_token is None:
            print('set pad_token...')
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if tokenizer.sep_token is None:
            print('set sep_token...')
            tokenizer.add_special_tokens({'sep_token': '[SEP]'})
        if tokenizer.eos_token is None:
            print('set eos_token...')
            tokenizer.add_special_tokens({'eos_token': '[EOS]'})
        # add token here
        tokenizer.add_tokens([HL_TOKEN],special_tokens=True)

    return tokenizer

# bert
tokenizer = get_tokenizer('bert-base-chinese')
model = AutoModel.from_pretrained('bert-base-chinese')
model.resize_token_embeddings(len(tokenizer))

log_dir = './bert-base-chinese-hl'
log_dir = os.path.join(log_dir,'huggingface_model')
os.makedirs(log_dir,exist_ok=True)
model.save_pretrained(log_dir)
tokenizer.save_pretrained(log_dir)

# gpt2
tokenizer = get_tokenizer('bert-base-chinese')
model = AutoModelForCausalLM.from_pretrained('ckiplab/gpt2-base-chinese',n_layer=6)
model.resize_token_embeddings(len(tokenizer))

log_dir = './gpt2-base-chinese-hl'
log_dir = os.path.join(log_dir,'huggingface_model')
os.makedirs(log_dir,exist_ok=True)
model.save_pretrained(log_dir)
tokenizer.save_pretrained(log_dir)