from transformers import GPT2Model, GPT2LMHeadModel, BertTokenizerFast 

tokenizer = BertTokenizerFast.from_pretrained('./tknzrs/daily_tknzr')

# tokenizer config
pad = tokenizer.convert_tokens_to_ids('[PAD]')
sep = tokenizer.convert_tokens_to_ids('[SEP]')
unk = tokenizer.convert_tokens_to_ids('[UNK]')
csep = tokenizer.convert_tokens_to_ids('[CSEP]')
msep = tokenizer.convert_tokens_to_ids('[MSEP]')
print(sep, pad)

# config = GPT2Config(vocab_size=32000, n_position=256, n_embd=512, n_layer=2, n_head=8)
load_from = 'ckpts/val_best_epoch_1'
model = GPT2LMHeadModel.from_pretrained(load_from, pad_token_id=pad, eos_token_id=sep)

inputs = tokenizer("어제 교수님이 ", return_tensors="pt")
outputs = model(**inputs)
# print(outputs.last_hidden_state.shape)

print('decoding start..')

def decoding(ids_list):
    decoded = tokenizer.batch_decode(ids_list)
    return decoded
    # return tokenizer.convert_ids_to_tokens(ids[0])

sample_outputs = model.generate(
    inputs.input_ids[:, :-1],
    max_length=60, 
    num_beams=1,
    num_return_sequences=1,
    early_stopping=True,
    no_repeat_ngram_size=2,
    bad_words_ids=[[unk]],
)

decoded = decoding(sample_outputs.tolist())
print(decoded)
