import tensorflow as tf
import os
from google.cloud import storage
import random

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))

def serialize_ids(**kwargs):
    feature = {k: _int64_feature(v) for k, v in kwargs.items()}
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example_proto.SerializeToString()

def load_from_gcs(bucket_name, prefix=None):
    if os.path.exists('/home/onion/private'):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/onion/private/languagemodel-tpu-key.json"

    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    train_from = []
    for blob in blobs:
        name = blob.name
        gsutil = 'gs://' + bucket_name + '/' + name 
        train_from.append(gsutil)
    return train_from

def masked_lm_predictions(input_ids, masked_lm_prob,
                            vocab_size=32000, mask_token_id=4, except_token_ids = [0, 1, 2, 3, 4, 5, 6, 7]):
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

    return {
        'input_ids': input_ids,
        'masked_input_ids': masked_input_ids,
        'masked_label': masked_label,
    }

if __name__ == '__main__':
    data = [0, 5, 11, 66, 66, 66, 88, 99, 9, 9, 1, 0]
    masked = masked_lm_predictions(data, 0.3)
    print(masked)