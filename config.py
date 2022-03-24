

''' model '''
MAX_SEQ_LEN = 256

''' train'''

IS_LOAD = True
LOAD_PATH = 'ckpts/gpt_best.h5'
BS = 64 
EPOCHS = 4 
LR = 5e-5

''' deploy '''
TKNZR_PATH = 'tknzrs/daily_tknzr'
DEPLOY_PATH = 'ckpts/for_serve'