# optimizer scheduler from tranining: https://huggingface.co/transformers/main_classes/optimizer_schedules.html
from transformers import get_cosine_schedule_with_warmup
from loguru import logger

def setup_scheduler(func):
    def wrapper(*args,**kwargs):
        self = args[0]
        opt = func(*args,**kwargs)
        # setup_scheduler
        train_dataloader_size = len(self.train_dataloader())
        num_warmup_steps = int(train_dataloader_size*0.5)
        num_training_steps = self.trainer.max_epochs*train_dataloader_size
        self.scheduler = get_cosine_schedule_with_warmup(
                opt,
                num_warmup_steps = num_warmup_steps, # use fisrt 0.5 epoch for warmup
                num_training_steps = num_training_steps # total training step
            )

        logger.info(f'optim scheduler is enable, num_warmup_steps:{num_warmup_steps} num_training_steps:{num_training_steps}')
        return opt
    return wrapper

def step_scheduler(func):
    def wrapper(*args,**kwargs):
        self = args[0]
        scheduler = getattr(self,'scheduler',None)
        assert scheduler is not None, 'scheduler not set'
        out = func(*args,**kwargs)
        # step_scheduler and log
        scheduler.step()
        self.log('lr',self.scheduler.get_last_lr()[0],prog_bar=True)
        return out
    return wrapper
