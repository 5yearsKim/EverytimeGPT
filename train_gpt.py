import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, GPT2Config, TFPreTrainedModel, BertTokenizerFast
from trainer import Trainer
from trainer.criterion import sparse_categorical_crossentropy 
from dataloader.load_gpt_data import read_tfrecord
from glob import glob
from config import *
from google.cloud import storage
import os

run_type = 'tpu'

if run_type == 'cpu':
    strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
elif run_type == 'tpu':
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
num_replica = strategy.num_replicas_in_sync
batch_size = (BS // num_replica) * num_replica


with strategy.scope():
    config = GPT2Config(n_embd=48, n_layer=1)
    model = TFGPT2LMHeadModel(config)
    # model = TFGPT2LMHeadModel.from_pretrained('ckpts/epoch0_best')
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    criterion = sparse_categorical_crossentropy 

# train_from = glob('gs://nlp-pololo/everytime/*.tfrecord', recursive=True)


# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/onion/private/languagemodel-tpu-key.json"

storage_client = storage.Client()
# Note: Client.list_blobs requires at least package version 1.17.0.
blobs = storage_client.list_blobs('nlp-pololo')

train_from = []
bucket_name = 'nlp-pololo'
for blob in blobs:
    name = blob.name
    gsutil = 'gs://' + bucket_name + '/' + name 
    train_from.append(gsutil)
print(train_from)

dset = read_tfrecord(train_from).padded_batch(batch_size, padded_shapes=(MAX_SEQ_LEN, MAX_SEQ_LEN),\
    padding_values=tf.constant(0, dtype=tf.int64), drop_remainder=True)

skip_point = 1000 
train_set, val_set = dset.skip(skip_point), dset.take(skip_point)
print('splitting train/val set..')

dist_train_set = strategy.experimental_distribute_dataset(train_set)
dist_val_set = strategy.experimental_distribute_dataset(val_set)

trainer = Trainer(model, optimizer, criterion, train_set, val_set, strategy=strategy)
trainer.train(EPOCHS)
