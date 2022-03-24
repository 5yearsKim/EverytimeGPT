from fastapi import FastAPI
from transformers import BertTokenizerFast, TFGPT2LMHeadModel, GPT2Config
import tensorflow as tf
from pydantic import BaseModel
from config import *
from inference import generate_topk, decoding
app = FastAPI()

tokenizer = BertTokenizerFast.from_pretrained("./tknzrs/daily_tknzr")
model = TFGPT2LMHeadModel.from_pretrained(DEPLOY_PATH, pad_token_id=0, eos_token_id=3)

tf.debugging.set_log_device_placement(True)

@app.get("/")
async def root():
    return {"message": "Hello World"}

class GenerateTpl(BaseModel):
    sent: str
    keywords: list|None = []
    num_sent: int = 3

@app.post('/generate_sample')
async def generate_sample(body: GenerateTpl):
    if body.keywords:
        prefix = '[MSEP]'.join(body.keywords)
        sent = prefix + '[CSEP]' + body.sent
    else:
        sent = body.sent

    inputs = tokenizer(sent, return_tensors="tf")

    input_ids = inputs.input_ids[:, :-1]
    sample_outputs = generate_topk(model, input_ids, max_len=80, k=4) 
    # sample_outputs = generate_beam(model, input_ids) 

    decoded = decoding(sample_outputs)
    return {'generated': decoded}