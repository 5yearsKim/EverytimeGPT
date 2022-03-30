import tensorflow as tf
from transformers import BertTokenizerFast
from dataloader.tfrecord_utils import  serialize_ids, masked_lm_predictions
import os
from tqdm import tqdm

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

def replace_sep(sent):
    return sent.replace('#|#', '[MSEP]').replace('#&#', '[CSEP]')

def write_tfrecord(files_from, out_dir, seed=0):
    file_item_len = 200_000
    file_idx = 1
    os.makedirs(os.path.dirname(out_dir), exist_ok=True) 
    writer = tf.io.TFRecordWriter('sample.tfrecord')
    for fpath in files_from:
        with open(fpath, 'r') as fr:
            for i, line in enumerate(tqdm(fr)):
                if i % file_item_len == 0:
                    print(i // 10000, '만')
                    writer.close()
                    writer = tf.io.TFRecordWriter(os.path.join(out_dir, f'seed_{seed}_record_{file_idx}.tfrecord'))
                    file_idx += 1
                line = replace_sep(line)
                input_ids_list = tokenize_line(line)
                for input_ids in input_ids_list:
                    mout = masked_lm_predictions(input_ids, 0.3)
                    example = serialize_ids(masked_input_ids = mout['masked_input_ids'], masked_label=mout['masked_label'])
                    writer.write(example)
    writer.close()


if __name__ == "__main__":
    for i in range(1):
        write_tfrecord(['data/bert/mlm_data/news.txt'], 'data/bert/news/', seed=i)
    # write_tfrecord(['data/sample.txt'], 'data/sample')