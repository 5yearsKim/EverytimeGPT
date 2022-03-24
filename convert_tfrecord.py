import tensorflow as tf
from transformers import BertTokenizerFast
from dataloader.tfrecord_utils import _int64_feature
import os

file_from = 'data/sample.txt'
file_to = 'data/sample.tfrecord'

tokenizer = BertTokenizerFast.from_pretrained('./tknzrs/daily_tknzr')


def tokenize_line(line, max_len=256):
    holder = []
    input_ids = tokenizer.encode(line)
    if len(input_ids) > max_len:
        bos, eos = input_ids[0], input_ids[-1]
        while len(input_ids) > max_len - 4:
            holder.append(input_ids[:max_len - 4] + [eos])
            input_ids = [bos] + input_ids[max_len //4 * 3:]
    holder.append(input_ids)

    return holder 

def serialize_example(input_ids):
    feature = {
        'input_ids' : _int64_feature(input_ids),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecord(files_from, out_dir):
    file_item_len = 500000
    file_idx = 1
    os.makedirs(os.path.dirname(out_dir), exist_ok=True) 
    writer = tf.io.TFRecordWriter('sample.tfrecord')
    for fpath in files_from:
        with open(fpath, 'r') as fr:
            for i, line in enumerate(fr):
                if i % file_item_len == 0:
                    print(i // 10000, '만')
                    writer.close()
                    writer = tf.io.TFRecordWriter(os.path.join(out_dir, f'record_{file_idx}.tfrecord'))
                    file_idx += 1
                input_ids_list = tokenize_line(line)
                for input_ids in input_ids_list:
                    example = serialize_example(input_ids)
                    writer.write(example)
    writer.close()


if __name__ == "__main__":
    write_tfrecord(['data/everytime_keword.txt'], 'data/everytime_keword/')
    # read_tfrecord('data/sample.tfrecord')