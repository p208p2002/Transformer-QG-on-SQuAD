from transformers import AutoConfig
from .argparser import get_args

_args = get_args()

#
GPUS = -1
ACCELERATOR = 'dp'

#
HL_TOKEN = '[HL]'
MODEL_CONFIG = AutoConfig.from_pretrained(_args.base_model)
MAX_QUESTION_LENGTH = 32
MAX_CONTEXT_LENGTH = 480
MAX_INPUT_LENGTH = MAX_CONTEXT_LENGTH + MAX_QUESTION_LENGTH # max:1024