import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, GPT2Config, TFPreTrainedModel, BertTokenizerFast
from trainer import Trainer
from trainer.criterion import sparse_categorical_crossentropy 
from dataloader.load_gpt_data import read_tfrecord
from glob import glob
from config import *
from distribute.utils import setup_strategy

strategy, num_replica = setup_strategy()

batch_size = (BS // num_replica) * num_replica

with strategy.scope():
    config = GPT2Config(vocab_size=32000, n_embd=768, n_layer=12, n_head=12)
    model = TFGPT2LMHeadModel(config)
    # model = TFGPT2LMHeadModel.from_pretrained('ckpts/epoch0_best')
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    criterion = sparse_categorical_crossentropy 

train_from = glob('data/everytime/*.tfrecord', recursive=True)

train_from = train_from[:1]

dset = read_tfrecord(train_from).padded_batch(batch_size, padded_shapes=(MAX_SEQ_LEN, MAX_SEQ_LEN),\
    padding_values=tf.constant(0, dtype=tf.int64), drop_remainder=True)

skip_point = 1000 
train_set, val_set = dset.skip(skip_point), dset.take(skip_point)
print('splitting train/val set..')

dist_train_set = strategy.experimental_distribute_dataset(train_set)
dist_val_set = strategy.experimental_distribute_dataset(val_set)

trainer = Trainer(model, optimizer, criterion, train_set, val_set, strategy=strategy)
trainer.train(EPOCHS)
