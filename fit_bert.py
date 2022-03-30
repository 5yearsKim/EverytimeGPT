from transformers import BertConfig, BertTokenizerFast, TFBertLMHeadModel
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
    config = BertConfig(vocab_size=32000, n_embd=512, n_layer=8, n_head=8)
    input_ids = tf.keras.layers.Input(shape=(max_len,), dtype='int32')
    bert = TFBertLMHeadModel(config)
    out = bert(input_ids).logits
    model = tf.keras.Model(inputs=input_ids, outputs=out)
    if IS_LOAD:
        load_path = LOAD_PATH 
        model.load_weights(load_path)
        print(f'loaded from {load_path}!!')
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer,
        loss=masked_cce,
    )
    return model

with strategy.scope():
    model = create_model()

# train_from = glob('data/sample/*.tfrecord', recursive=True)
train_from = load_from_gcs('nlp-pololo', prefix='mlm_tfrecord/everytime/')
# train_from = train_from[:1]
print(train_from)

dset = read_mlm_tfrecord(train_from)\
    .padded_batch(batch_size,\
        padded_shapes=(MAX_SEQ_LEN, MAX_SEQ_LEN),\
        padding_values=(tf.constant(0, dtype=tf.int64), tf.constant(-100, dtype=tf.int64)),\
        drop_remainder=True)

skip_point = 1000 
train_set, val_set = dset.skip(skip_point), dset.take(skip_point)
print('splitting train/val set..')

train_set = strategy.experimental_distribute_dataset(train_set.repeat())
val_set = strategy.experimental_distribute_dataset(val_set)

train_steps = 2000000 // BS + 1
val_steps = skip_point 


callbacks = [
    # EarlyStopping(monitor='val_loss', patience=10), 
    # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=_learning_rate * 0.05),
    ModelCheckpoint(f"ckpts/best.h5",
        monitor='val_loss', 
        save_best_only=True,
        save_weights_only=True,
        mode='auto')
    # WandbCallback(),
    ]

for data in train_set:
    print(data.shape)
    break
# print(f'train batch size={batch_size}, lr={LR}')
# model.fit(train_set,
#     epochs=EPOCHS,
#     steps_per_epoch=train_steps,
#     callbacks=callbacks,
#     validation_data=val_set,
#     validation_steps=val_steps
# )