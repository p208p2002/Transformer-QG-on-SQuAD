from transformers import AutoConfig
from .argparser import get_args

_args = get_args()

#
GPUS = -1
ACCELERATOR = 'dp'

#
HL_TOKEN = '[HL]'
MODEL_CONFIG = AutoConfig.from_pretrained(_args.base_model)
MAX_INPUT_LENGTH = 512 # max 1024
MAX_QUESTION_LENGTH = 32