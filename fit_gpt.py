import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, GPT2Config, TFPreTrainedModel, BertTokenizerFast
from trainer import Trainer
from trainer.criterion import sparse_categorical_crossentropy 
from dataloader.load_gpt_data import read_tfrecord
from glob import glob
from config import *
from google.cloud import storage
import os

run_type = 'cpu'

if run_type == 'cpu':
    strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
elif run_type == 'tpu':
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
num_replica = strategy.num_replicas_in_sync
batch_size = (BS // num_replica) * num_replica

def create_model(max_len=256):
    config = GPT2Config(vocab_size=32000, n_embd=768, n_layer=12, n_head=12)
    input_ids = tf.keras.layers.Input(shape=(max_len,), dtype='int32')
    gpt = TFGPT2LMHeadModel(config)
    out = gpt(input_ids).logits
    model = tf.keras.Model(inputs=input_ids, outputs=out)
    # model = TFGPT2LMHeadModel.from_pretrained('ckpts/epoch0_best')
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer,
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss=sparse_categorical_crossentropy,
    )
    return model

with strategy.scope():
    model = create_model()

train_from = glob('data/everytime/*.tfrecord', recursive=True)


# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/onion/private/languagemodel-tpu-key.json"

# storage_client = storage.Client()
# blobs = storage_client.list_blobs('nlp-pololo')

# train_from = []
# bucket_name = 'nlp-pololo'
# for blob in blobs:
#     name = blob.name
#     gsutil = 'gs://' + bucket_name + '/' + name 
#     train_from.append(gsutil)
# print(train_from)

train_from = train_from[:1]
print(train_from)

dset = read_tfrecord(train_from).padded_batch(batch_size, padded_shapes=(MAX_SEQ_LEN, MAX_SEQ_LEN),\
    padding_values=tf.constant(0, dtype=tf.int64), drop_remainder=True)

skip_point = 1000 
train_set, val_set = dset.skip(skip_point), dset.take(skip_point)
print('splitting train/val set..')

if run_type == 'tpu':
    train_set = strategy.experimental_distribute_dataset(train_set)
    val_set = strategy.experimental_distribute_dataset(val_set)

steps_per_epoch = 500000 // BS + 1

model.fit(train_set,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_set,
    )