from transformers import AutoConfig
from .argparser import get_args

_args = get_args()

#
GPUS = -1
ACCELERATOR = 'dp'

#
HL_TOKEN = '[HL]'
MODEL_CONFIG = AutoConfig.from_pretrained(_args.base_model)
MAX_INPUT_LENGTH = MODEL_CONFIG.n_positions # 1024
MAX_QUESTION_LENGTH = 100
MAX_CONTEXT_LENGTH = MAX_INPUT_LENGTH - MAX_QUESTION_LENGTH