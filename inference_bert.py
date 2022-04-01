from transformers import BertTokenizerFast, TFBertForPreTraining
from glob import glob
from config import *
import tensorflow as tf
from dataloader.load_data import read_mlm_tfrecord


test_mlm = [
    '대한민국의 수도는 [MASK]이다. 대한민국의 [MASK]는 문재인이다.',
    '이번에 발생했[MASK] [MASK]는 [MASK]일 것으로 예상됩니다.',
    '피부가 [MASK] 이면 탄력이 없다고 [MASK]한다.',
    '그에게 기도할 [MASK]가 왔는가 아닌가는 성실한 [MASK]이다.'
]



def inference():

    model = TFBertForPreTraining.from_pretrained('ckpts/bert/news_bert')

    tokenizer = BertTokenizerFast.from_pretrained('tknzrs/daily_tknzr')
    inputs = tokenizer(test_mlm, return_tensors='tf', padding=True) 

    # import os
    # if os.path.exists('/home/onion/private'):
    #     os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/onion/private/languagemodel-tpu-key.json"
    # dset = read_mlm_tfrecord(['gs://nlp-pololo/mlm_tfrecord/news/seed_0_record_1.tfrecord']).padded_batch(4,
    #         padded_shapes=(256, 256),\
    #         padding_values=(tf.constant(0, dtype=tf.int64), tf.constant(-100, dtype=tf.int64)),\
    #         drop_remainder=True)
    # for data in dset:
    #     inputs, label = data
    #     break
    mlm_logits = model(inputs).prediction_logits
    # print(mlm_logits)
    print(tf.math.reduce_sum(mlm_logits))

    top_tokens = tf.math.top_k(mlm_logits, 1).indices
    top_tokens = tf.reshape(top_tokens, (len(test_mlm), -1))

    for i, input_ids in enumerate(inputs.input_ids):
    # for i, input_ids in enumerate(inputs):

        masked = tokenizer.decode(input_ids)
        print(masked.replace('[PAD]', ''))

        input_ids = input_ids.numpy()
        for j, char in enumerate(input_ids):
            if char == 4:
            # if True:
                input_ids[j] = top_tokens[i, j]
        decoded = tokenizer.decode(input_ids)
        print(decoded.replace('[PAD]', ''))
        print('-----------------')

if __name__ == '__main__':
    inference()

