import tensorflow as tf

def make_gpt_label(feature):
    input_ids = feature['input_ids']
    return input_ids[:-1], input_ids[1:]

def read_gpt_tfrecord(files_from):
    feature_description = {
        'input_ids': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
    }
    def _parse_func(example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    raw_dataset = tf.data.TFRecordDataset(files_from)
    dataset = raw_dataset.map(_parse_func).map(make_gpt_label)
    return dataset

def make_mlm_label(feature, with_sop=False):
    if with_sop:
        return feature['masked_input_ids'], (feature['masked_label'], tf.reshape(feature['sop_label'], (-1,)))
    return feature['masked_input_ids'], feature['masked_label']

def read_mlm_tfrecord(files_from, with_sop=False):
    feature_description = {
        'masked_input_ids': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
        'masked_label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=-100),
    }
    if with_sop:
        feature_description['sop_label'] = tf.io.FixedLenSequenceFeature([1], tf.float32, allow_missing=True)
    def _parse_func(example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    dataset = tf.data.TFRecordDataset(files_from).map(_parse_func).map(lambda x: make_mlm_label(x, with_sop=with_sop))
    return dataset


def make_ctx_label(feature):
    context_ids = feature['context_ids']
    answer_ids = feature['answer_ids']
    return (context_ids, answer_ids[:-1]), answer_ids[1:]

def read_ctx_tfrecord(files_from):
    feature_description = {
        'context_ids': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
        'answer_ids': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
    }
    def _parse_func(example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    raw_dataset = tf.data.TFRecordDataset(files_from)
    dataset = raw_dataset.map(_parse_func).map(make_ctx_label)
    return dataset




if __name__== "__main__":

    # import os
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/onion/private/languagemodel-tpu-key.json"
    # dset = read_gpt_tfrecord(['gs://nlp-pololo/everytime_keword/record_1.tfrecord' ])
    # for item in dset:
    #     print(dset)

    dset = read_mlm_tfrecord(['data/sample/seed_0_record_1.tfrecord'], with_sop=True)
    for data in dset:
        print(data[1][1])
        break