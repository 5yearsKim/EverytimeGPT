

''' model '''
MAX_SEQ_LEN = 256

GPT_SMALL_CONFIG = {
    'vocab_size': 32000,
    'n_embd': 512,
    'n_layer': 8,
    'n_head': 8,
    'bos_token_id': 3,
    'eos_token_id': 3,
    'pad_token_id': 0,

}

GPT_BASE_CONFIG = {
    'vocab_size': 32000,
    'n_embd': 768,
    'n_layer': 12,
    'n_head': 12,
    'bos_token_id': 3,
    'eos_token_id': 3,
    'pad_token_id': 0,
}


BERT_XXSMALL_CONFIG = {
    'vocab_size': 32000,
    'hidden_size': 512,
    'num_hidden_layers': 1,
    'num_attention_heads': 8
}

BERT_SMALL_CONFIG = {
    'vocab_size': 32000,
    'hidden_size': 512,
    'num_hidden_layers': 8,
    'num_attention_heads': 8
}

''' train'''
IS_LOAD = False 
WITH_SOP = True 
LOAD_PATH = 'ckpts/best.h5'
BS = 128
EPOCHS = 50
LR = 1e-4

''' deploy '''
TKNZR_PATH = 'tknzrs/daily_tknzr'
DEPLOY_PATH = 'ckpts/for_serve'