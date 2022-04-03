from transformers import BertConfig, BertTokenizerFast, TFEncoderDecoderModel 
from transformers import BertConfig, GPT2Config, EncoderDecoderConfig
from config import *
from distribute.utils import setup_strategy
import tensorflow as tf
from dataloader.load_data import read_mlm_tfrecord
from dataloader.tfrecord_utils import load_from_gcs
from trainer.criterion import masked_cce
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from glob import glob

tokenizer = BertTokenizerFast.from_pretrained(TKNZR_PATH)
strategy, num_replica = setup_strategy()
batch_size = (BS // num_replica) * num_replica
if num_replica == 1:
    batch_size = 1

def create_model(max_len=256):
    config_enc = BertConfig(vocab_size=32000, num_hidden_layers=1)
    config_dec = GPT2Config(vocab_size=32000, n_layer=1)
    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_enc, config_dec)
    transformer = TFEncoderDecoderModel(config=config)

    context_ids = tf.keras.layers.Input(shape=(max_len,), dtype='int32')
    decoder_ids = tf.keras.layers.Input(shape=(max_len,), dtype='int32')
    
    tout = transformer(input_ids=context_ids, decoder_input_ids=decoder_ids)
    outputs =  tout.logits
    model = tf.keras.Model(inputs=[context_ids, decoder_ids], outputs=outputs)

    if IS_LOAD:
        load_path = LOAD_PATH 
        model.load_weights(load_path)
        print(f'loaded from {load_path}!!')

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return model

with strategy.scope():
    model = create_model()

inp_a = tf.zeros((3, 256), dtype=tf.int32)
inp_b = tf.zeros((3, 256), dtype=tf.int32)
out = model((inp_a, inp_b))
print(out)