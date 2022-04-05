from transformers import BertTokenizerFast, TFGPT2LMHeadModel, GPT2Config
from transformers import TFEncoderDecoderModel
import tensorflow as tf
from config import *

tokenizer = BertTokenizerFast.from_pretrained(TKNZR_PATH)

def inference_gpt():
    print('tokenizer loaded')

    # model = TFGPT2LMHeadModel.from_pretrained(DEPLOY_PATH, pad_token_id=0, eos_token_id=3 )
    model = TFGPT2LMHeadModel.from_pretrained('ckpts/gpt/gpt_small', pad_token_id=0, eos_token_id=3 )

    text_start = '사실 여자친구가 '

    inputs = tokenizer(text_start, return_tensors="tf")

    input_ids = inputs.input_ids[:, :-1]
    sample_outputs = generate_topk(model, input_ids, max_len=80) 
    # sample_outputs = generate_beam(model, input_ids)

    decoded = decoding(sample_outputs)
    for sent in decoded:
        print(sent)

def inference_transformer():
    model = TFEncoderDecoderModel.from_pretrained('ckpts/transformer/context')
    context = '여자친구랑 헤어졌어.'
    inputs = tokenizer(context, return_tensors="tf")
    input_ids = inputs.input_ids

    sample_outputs = generate_topk(model, input_ids, max_len=80) 
    # sample_outputs = generate_beam(model, input_ids) 

    decoded = decoding(sample_outputs)
    for sent in decoded:
        print(sent)


def decoding(ids_list):
    def make_pretty(sent):
        sent = sent.replace('[PAD]', '').replace('\n', '').strip()
        return sent
    decoded = tokenizer.batch_decode(ids_list)
    decoded = list(map(make_pretty, decoded))
    return decoded
    # return tokenizer.convert_ids_to_tokens(ids[0])

def generate_beam(model, input_ids, num_beams=1, max_len=40):
    sample_outputs = model.generate(
        input_ids,
        max_length=max_len, 
        num_beams=num_beams,
        num_return_sequences=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=1,
        # bad_token_ids = msep, csep
    )
    return sample_outputs.numpy()

def generate_topk(model, input_ids, k=5, max_len=40, num_sent=3, temperature=0.8, no_repeat_size=1):
    sample_outputs = model.generate(
        input_ids,
        max_length=max_len,
        do_sample=True,
        top_k=k,
        num_return_sequences=num_sent,
        temperature=temperature,
        early_stopping=True,
        # no_repeat_ngram_size=no_repeat_size,
        no_repeat_ngram_size=no_repeat_size,
        repetition_penalty=1.15,
    )
    return sample_outputs.numpy()

if __name__ == '__main__':
    inference_transformer()
    # inference_gpt()