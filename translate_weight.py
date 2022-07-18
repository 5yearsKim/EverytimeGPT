import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Config, BertConfig, TFBertForPreTraining, TFEncoderDecoderModel

from config import BERT_XXSMALL_CONFIG, GPT_SMALL_CONFIG

def convert_gpt(ckpt_from='ckpts/gpt/gpt_small1.h5', ckpt_to='ckpts/gpt/gpt_small'):
    config = GPT2Config(**GPT_SMALL_CONFIG)
    input_ids = tf.keras.layers.Input(shape=(256,), dtype='int32')
    gpt = TFGPT2LMHeadModel(config)
    out = gpt(input_ids).logits
    model = tf.keras.Model(inputs=input_ids, outputs=out)

    model.load_weights(ckpt_from)

    gpt.save_pretrained(ckpt_to)

def convert_bert(ckpt_from, ckpt_to, with_sop=False):
    config = BertConfig(**BERT_XXSMALL_CONFIG)
    input_ids = tf.keras.layers.Input(shape=(256,), dtype='int32')
    bert = TFBertForPreTraining(config)
    bout = bert(input_ids)
    outputs =  [bout.prediction_logits, bout.seq_relationship_logits[:, 0]] if with_sop else bout.prediction_logits
    model = tf.keras.Model(inputs=input_ids, outputs=outputs)

    model.load_weights(ckpt_from)
    bert.save_pretrained(ckpt_to)

def convert_transformer(ckpt_from, ckpt_to):
    transformer = TFEncoderDecoderModel.from_encoder_decoder_pretrained('ckpts/bert/xxsmall_bert', 'ckpts/gpt/gpt_small')

    context_ids = tf.keras.layers.Input(shape=(256,), dtype='int32')
    decoder_ids = tf.keras.layers.Input(shape=(256,), dtype='int32')
    
    tout = transformer(input_ids=context_ids, decoder_input_ids=decoder_ids)
    outputs =  tout.logits
    model = tf.keras.Model(inputs=[context_ids, decoder_ids], outputs=outputs)

    model.load_weights(ckpt_from)
    transformer.save_pretrained(ckpt_to)



if __name__ == '__main__':
    # convert_bert('ckpts/bert/bert_xxsmall.h5', 'ckpts/bert/xxsmall_bert')
    convert_gpt('ckpts/gpt/context_gpt_small1.h5', 'ckpts/gpt/context_gpt_small')
    # convert_transformer('ckpts/transformer/context2.h5', 'ckpts/transformer/context')