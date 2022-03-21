import tensorflow as tf

def make_label(feature):
    input_ids = feature['input_ids']
    return input_ids[:-1], input_ids[1:]

def read_tfrecord(files_from):
    feature_description = {
        'input_ids': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
    }
    def _parse_func(example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    raw_dataset = tf.data.TFRecordDataset(files_from)
    dataset = raw_dataset.map(_parse_func).map(make_label)
    return dataset