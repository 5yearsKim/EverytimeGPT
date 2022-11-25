from .utils import AverageMeter
import torch
from tqdm import tqdm
import wandb
import os

class Trainer:
    def __init__(self, model, optim, criterion, train_loader, val_loader, val_best_path='ckpts/', use_wandb=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optim = optim
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_best_path = val_best_path
        self.loss_meter = AverageMeter() 
        self.val_best = float('inf') 
        self.use_wandb = use_wandb

    def train(self, epochs):
        for epoch in range(epochs):
            tepoch = tqdm(self.train_loader, unit="batch")
            tepoch.set_description(f"Epoch {epoch}")
            self.model.train()
            self.loss_meter.reset()
            for i, (inputs, y) in enumerate(tepoch):
                self.train_step(inputs, y)
                tepoch.set_postfix(loss=self.loss_meter.avg)
            self.validate(epoch)
    
    def train_step(self, inputs, y):
        self.optim.zero_grad()
        inputs, y = inputs.to(self.device), y.to(self.device)
        logits = self.model(**inputs).logits
        loss = self.criterion(logits.view((-1, logits.shape[-1])), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.optim.step()
        self.loss_meter.update(loss.item())

        
    def validate(self, epoch):
        loss_meter = AverageMeter()
        with torch.no_grad():
            for i, (inputs, y) in enumerate(self.val_loader):
                inputs, y = inputs.to(self.device), y.to(self.device)
                logits = self.model(**inputs).logits
                loss = self.criterion(logits.view((-1, logits.shape[-1])), y.view(-1))
                loss_meter.update(loss.item())

        print(f'val loss: {loss_meter.avg}')
        if self.use_wandb:
            wandb.log({
                "train_loss": self.loss_meter.avg,
                "val_loss": loss_meter.avg,
            })

        if loss_meter.avg < self.val_best:
            self.val_best = loss_meter.avg 
            print('validation best..')
            path = os.path.join(self.val_best_path, f'val_best_epoch_{epoch}')
            self.model.save_pretrained(path)
            print(f'model saved at {path}')

    # def save(self, save_path):
    #     torch.save({
    #         'model_state': self.model.state_dict(),
    #         }, save_path)
    #     print(f'model saved at {save_path}')
    
    # def load(self, load_path):
    #     save_dict = torch.load(load_path)
    #     self.model.load_state_dict(save_dict['model_state'])
    #     print(f'model loaded from {load_path}')