

GPT_CONFIG = {'vocab_size':32000, 'n_position':256, 'n_embd':768, 'n_layer':12, 'n_head': 12,}

''' trainig config '''
USE_WANDB = True 
LOAD_CKPT = False

LR = 2e-5
WD = 0
BS = 16 
EPOCHS = 10
