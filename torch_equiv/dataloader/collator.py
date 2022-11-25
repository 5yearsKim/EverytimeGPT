import torch

class Collator:
    def __init__(self, tknzr):
        self.tknzr = tknzr

    def __call__(self, data, max_len=256):
        inputs = self.tknzr(data, return_tensors='pt', padding=True, max_length=max_len, truncation=True)
        labels = inputs['input_ids'].roll(-1, 1)
        labels[:, -1].fill_(-100)
        return inputs, labels