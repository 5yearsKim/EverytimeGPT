
from config import *
from transformers import BertTokenizerFast, GPT2LMHeadModel, GPT2Config
import torch
from torch.utils.data import DataLoader
from dataloader import GenerationData, Collator
from trainer import Trainer
import wandb

if USE_WANDB:
    wandb.init(project='daily-gpt', entity='akaai', name=f'daily-gpt')
    wandb.config = {
        "learning_rate": LR,
        "batch_size": BS,
    }

trainset = GenerationData(files_from=['data/for_lm/train_lm.txt'])
valset = GenerationData(files_from=['data/for_lm/val_lm.txt'])

tknzr = BertTokenizerFast.from_pretrained('tknzrs/daily_tknzr')
collator = Collator(tknzr)

train_loader = DataLoader(trainset, batch_size=BS, collate_fn=collator)

val_loader = DataLoader(valset, batch_size=BS, collate_fn=collator)

config = GPT2Config(**GPT_CONFIG)
model = GPT2LMHeadModel(config)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

trainer = Trainer(model, optimizer, criterion, train_loader, val_loader, use_wandb=USE_WANDB)
trainer.train(EPOCHS)