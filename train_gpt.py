import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, GPT2Config, TFPreTrainedModel, BertTokenizerFast
from trainer import Trainer
from trainer.criterion import sparse_categorical_crossentropy 
from dataloader.load_gpt_data import read_tfrecord
from glob import glob
from config import *

run_type = 'cpu'

if run_type == 'cpu':
    strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
elif run_type == 'tpu':
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
num_replica = strategy.num_replicas_in_sync
batch_size = (BS // num_replica) * num_replica


with strategy.scope():
    # config = GPT2Config(n_embd=48, n_layer=1)
    # model = TFGPT2LMHeadModel(config)
    model = TFGPT2LMHeadModel.from_pretrained('ckpts/epoch0_best')
    optimizer = tf.keras.optimizers.Adam()
    criterion = sparse_categorical_crossentropy 

train_from = glob('data/**/*.tfrecord', recursive=True)
dset = read_tfrecord(train_from).padded_batch(batch_size, padded_shapes=256, padding_values=tf.constant(0, dtype=tf.int64))

skip_point = 16
train_set, val_set = dset.skip(skip_point), dset.take(skip_point)


dist_train_set = strategy.experimental_distribute_dataset(train_set)
dist_val_set = strategy.experimental_distribute_dataset(val_set)


trainer = Trainer(model, optimizer, criterion, train_set, val_set, strategy=strategy)
trainer.train(2)
