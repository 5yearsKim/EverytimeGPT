import tensorflow as tf
from tqdm.auto import tqdm
import os

class Trainer:
    def __init__(self, model, optim, criterion, train_set, val_set, strategy=None, save_dir='ckpts'):
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.train_set = train_set
        self.val_set = val_set
        self.val_best = float('inf')
        self.save_dir = save_dir
        self.strategy = strategy
        with strategy.scope():
            self.train_loss = tf.keras.metrics.Mean(name='train_loss') 
            self.val_loss = tf.keras.metrics.Mean(name='val_loss')


    def train(self, epochs):
        for epoch in range(epochs):
            print(f'running epoch {epoch}..')
            self.train_loss.reset_states()
            self.val_loss.reset_states()

            tepoch = tqdm(self.train_set, unit = 'batch', position=0, leave=True, ascii=True)
            for x, y in tepoch:
                self.dist_train_step(x, y)
                tepoch.set_postfix(loss=self.train_loss.result())

            print('validating..')
            for x, y in self.val_set:
                self.dist_val_step(x, y)
            
            if self.val_loss.result().numpy() < self.val_best:
                print('best val result! saving..')
                save_path = os.path.join(self.save_dir, f'epoch{epoch}_best')
                self.save(save_path)
                self.val_best = self.val_loss.result().numpy()

            print(f'@{epoch}epoch@ loss : {self.train_loss.result()}, val_loss: {self.val_loss.result()}')
            print('---------------------------------')

    @tf.function
    def dist_train_step(self, x, y):
        # self.strategy.experimanal_run_v2(self.train_step, args=(x, y))
        self.strategy.run(self.train_step, args=(x, y))

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True).logits
            loss = self.criterion(logits, y)
        gradient = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(gradient, self.model.trainable_variables))
        self.train_loss(loss)

    @tf.function
    def dist_val_step(self, x, y):
        # self.strategy.experimental_run_v2(self.val_step, args=(x, y))
        self.strategy.run(self.val_step, args=(x, y))

    @tf.function(experimental_relax_shapes=True)
    def val_step(self, x, y):
        logits = self.model(x).logits
        loss = self.criterion(logits, y)
        self.val_loss(loss)

    def save(self, save_path):
        self.model.save_pretrained(save_path)

