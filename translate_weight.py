import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, GPT2Config, TFPreTrainedModel, BertTokenizerFast, BertConfig, TFBertLMHeadModel

from config import BERT_SMALL_CONFIG

def convert_gpt(ckpt_from='ckpts/gpt.h5', ckpt_to='ckpts/gpt'):
    config = GPT2Config(vocab_size=32000, n_embd=768, n_layer=12, n_head=12)
    input_ids = tf.keras.layers.Input(shape=(256,), dtype='int32')
    gpt = TFGPT2LMHeadModel(config)
    out = gpt(input_ids).logits
    model = tf.keras.Model(inputs=input_ids, outputs=out)

    model.load_weights(ckpt_from)

    gpt.save_pretrained(ckpt_to)

def convert_bert(ckpt_from, ckpt_to):
    config = BertConfig(**BERT_SMALL_CONFIG)
    input_ids = tf.keras.layers.Input(shape=(256,), dtype='int32')
    bert = TFBertLMHeadModel(config)
    out = bert(input_ids).logits
    model = tf.keras.Model(inputs=input_ids, outputs=out)

    model.load_weights('best.h5')
    bert.save_pretrained('ckpts/bert/news_epoch')

if __name__ == '__main__':
    convert_bert('', '')