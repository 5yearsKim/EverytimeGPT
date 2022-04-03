from transformers import BertTokenizerFast, TFBertForPreTraining
from glob import glob
from config import *
import tensorflow as tf
from dataloader.load_data import read_mlm_tfrecord


test_mlm = [
    '미국의 대통령은 [MASK]이다.',
    '얼굴이 [MASK]하면 산이 무너지고 땅이 갈라진다.',
    '대한민국의 대통령인 [MASK]은 국가의 [MASK]를 위해 헌신하고 있다.',
    '그에게 기도할 [MASK]가 왔는가 아닌가는 내가 [MASK] 할 [MASK]이다.',
    '처음에는 4면 단행으로 발행하였으나, [MASK]의 자본에다가 농민들이 많이 봐서 그런지 1978년 8면, 1983년 12면, 1993년에 16면으로 증면됐다. 이후로 [MASK] 16면 체제.'
    '[MASK] 배송을 우편에 [MASK] 대부분 의존하는 [MASK] 때문에 16면 체제를 오래도록 [MASK]했지만, 2016년 20~24면의 지면개편을 단행했다. 2021년 현재는 16~28면 발행. 특집[MASK] 32면도 나온다'
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

