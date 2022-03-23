import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, GPT2Config, TFPreTrainedModel, BertTokenizerFast

config = GPT2Config(vocab_size=32000, n_embd=768, n_layer=12, n_head=12)
input_ids = tf.keras.layers.Input(shape=(256,), dtype='int32')
gpt = TFGPT2LMHeadModel(config)
out = gpt(input_ids).logits
model = tf.keras.Model(inputs=input_ids, outputs=out)

model.load_weights('ckpts/best2.h5')

gpt.save_pretrained('ckpts/converted')