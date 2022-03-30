

''' model '''
MAX_SEQ_LEN = 256

''' train'''

IS_LOAD = False 
LOAD_PATH = 'ckpts/gpt_best.h5'
BS = 64 
EPOCHS = 50 
LR = 1e-4

''' deploy '''
TKNZR_PATH = 'tknzrs/daily_tknzr'
DEPLOY_PATH = 'ckpts/for_serve'