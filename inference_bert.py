from transformers import BertTokenizerFast, TFBertLMHeadModel
from glob import glob
from config import *
import tensorflow as tf


test_mlm = [
  '오늘 [MASK] 는 [MASK]이다.',
  '설명은 필요[MASK]',
  '쿠팡..[MSEP]그에게 [MASK]할 때가 왔는가&안 그래도[MSEP]매일 문자[MASK] 와요'
]


def inference():
    tokenizer = BertTokenizerFast.from_pretrained('tknzrs/daily_tknzr')

    model = TFBertLMHeadModel.from_pretrained('ckpts/bert/news_epoch')
    # model = TFGPT2LMHeadModel.from_pretrained('distilbert-base-uncased')

    inputs = tokenizer(test_mlm, return_tensors='tf', padding=True) 
    mlm_logits = model(inputs).logits
    # print(mlm_logits)
    print(tf.math.reduce_sum(mlm_logits))

    top_tokens = tf.math.top_k(mlm_logits, 1).indices
    top_tokens = tf.reshape(top_tokens, (len(test_mlm), -1))

    for i, input_ids in enumerate(inputs.input_ids):
        input_ids = input_ids.numpy()
        for j, char in enumerate(input_ids):
            if char == 4:
                input_ids[j] = top_tokens[i, j]
        decoded = tokenizer.decode(input_ids)
        print(decoded)


