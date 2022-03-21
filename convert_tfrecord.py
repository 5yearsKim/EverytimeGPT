import tensorflow as tf
from transformers import BertTokenizerFast
from dataloader.tfrecord_utils import _int64_feature
import os

file_from = 'data/sample.txt'
file_to = 'data/sample.tfrecord'

tokenizer = BertTokenizerFast.from_pretrained('./tknzrs/daily_tknzr')


def tokenize_line(line):
    input_ids = tokenizer.encode(line)
    return input_ids

def serialize_example(input_ids):
    feature = {
        'input_ids' : _int64_feature(input_ids),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecord(files_from, out_dir):
    file_item_len = 10000
    file_idx = 1
    os.makedirs(os.path.dirname(out_dir), exist_ok=True) 
    writer = tf.io.TFRecordWriter('sample.tfrecord')
    for fpath in files_from:
        with open(fpath, 'r') as fr:
            for i, line in enumerate(fr):
                if i % file_item_len == 0:
                    writer.close()
                    writer = tf.io.TFRecordWriter(os.path.join(out_dir, f'record_{file_idx}.tfrecord'))
                    file_idx += 1
                input_ids = tokenize_line(line)
                example = serialize_example(input_ids)
                writer.write(example)
    writer.close()


if __name__ == "__main__":
    write_tfrecord(['data/sample.txt'], 'data/sample/')
    # read_tfrecord('data/sample.tfrecord')