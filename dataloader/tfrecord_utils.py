import tensorflow as tf
import os
from google.cloud import storage
import random

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def serialize_ids(**kwargs):
    feature = {}
    for k, v in kwargs.items():
        if isinstance(v, float):
            feature[k] = _float_feature(v)
        elif isinstance(list(v)[0], int):
            feature[k] = _int64_feature(v) 
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example_proto.SerializeToString()

def load_from_gcs(bucket_name, prefix=None, sort_key=None):
    if os.path.exists('/home/onion/private'):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/onion/private/languagemodel-tpu-key.json"
    prefix_list = list(prefix)
    storage_client = storage.Client()
    for prix in prefix_list:
        blobs = storage_client.list_blobs(bucket_name, prefix=prix)

        train_from = []
        for blob in blobs:
            name = blob.name
            gsutil = 'gs://' + bucket_name + '/' + name 
            train_from.append(gsutil)
    train_from = sorted(train_from, key=sort_key)
    return train_from

def masked_lm_predictions(input_ids, masked_lm_prob,
                            vocab_size=32000, mask_token_id=4, except_token_ids = [0, 1, 2, 3, 4, 5, 6, 7], with_sop=False, csep_token_id=5, sep_token_id=3):
    original_input_ids = input_ids.copy()

    if with_sop:
        bos, eos = input_ids[0], input_ids[-1]
        csep_locs = [index for index, element in enumerate(input_ids) if element == csep_token_id]
        if len(csep_locs) == 0:
            sop_label = 0.5
        else:
            csep_loc = random.choice(csep_locs)
            if random.random() < 0.5:
                sop_label = 0.
                input_ids[csep_loc] = sep_token_id
            else:
                bos, eos = input_ids[0], input_ids[-1]
                sent_a, sent_b = input_ids[1:csep_loc], input_ids[csep_loc + 1:-1]
                sop_label = 1.
                input_ids = [bos] + sent_b + [sep_token_id] + sent_a + [eos]


    masked_input_ids = input_ids.copy()
    masked_label = [-100 for _ in range(len(input_ids))]

    for i, tkn_id in enumerate(input_ids):
        if tkn_id in except_token_ids:
            continue
        if random.random() > masked_lm_prob:
            continue

        if random.random() < 0.8:
            masked_token = mask_token_id
        elif random.random() < 0.5:
            masked_token = random.randint(0, vocab_size)
        else:
            masked_token = tkn_id

        masked_label[i] = tkn_id
        masked_input_ids[i] = masked_token

    to_return = {
        'input_ids': original_input_ids,
        'masked_input_ids': masked_input_ids,
        'masked_label': masked_label,
    }
    if with_sop:
        to_return['sop_label'] = sop_label

    return to_return



if __name__ == '__main__':
    data = [0,  11, 66, 66, 5, 66, 88, 99, 9, 9, 1, 0]
    masked = masked_lm_predictions(data, 0.3, with_sop=True)
    print(masked)