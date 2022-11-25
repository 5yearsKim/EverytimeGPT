from fastapi import FastAPI
from transformers import BertTokenizerFast, TFGPT2LMHeadModel, GPT2Config
import tensorflow as tf
from pydantic import BaseModel
from config import *
from inference_gpt import generate_topk, decoding
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins=['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



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
    def make_pretty(sent):
        if not sent.endswith('[SEP]'):
            sent = sent + '...'
        sent = sent.replace('[CLS]', '').replace('[SEP]', '')
        sent = sent.split('[CSEP]')[-1]
        return sent

    mode = 'keyword' if body.keywords else 'normal'

    if mode == 'keyword':
        prefix = '[MSEP]'.join(body.keywords)
        sent = prefix + '[CSEP]' + body.sent
        no_repeat_size = 1 # 2
    else:
        sent = body.sent
        no_repeat_size = 1

    inputs = tokenizer(sent, return_tensors="tf")

    input_ids = inputs.input_ids[:, :-1]
    sample_outputs = generate_topk(model, input_ids, max_len=80, k=4, no_repeat_size=no_repeat_size) 
    # sample_outputs = generate_beam(model, input_ids) 

    decoded = decoding(sample_outputs)
    decoded = list(map(make_pretty, decoded))
    return {'generated': decoded}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('deploy:app', host='0.0.0.0', port=8001, reload=False, debug=False, workers=2)