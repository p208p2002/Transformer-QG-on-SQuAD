from transformers import AutoModel,AutoModelForCausalLM
from transformers import AutoTokenizer
from models.seq2seq_lm import get_tokenizer
from models.seq2seq_lm.config import HL_TOKEN

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
model = AutoModelForCausalLM.from_pretrained('ckiplab/gpt2-base-chinese',n_layer=4)
model.resize_token_embeddings(len(tokenizer))

log_dir = './gpt2-base-chinese-hl'
log_dir = os.path.join(log_dir,'huggingface_model')
os.makedirs(log_dir,exist_ok=True)
model.save_pretrained(log_dir)
tokenizer.save_pretrained(log_dir)