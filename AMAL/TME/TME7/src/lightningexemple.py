import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split,TensorDataset
from pathlib import Path
from datamaestro import prepare_dataset
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import datasets, transforms
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint


# # Paramètres pour le ModelCheckpoint
# CHECKPOINT_PATH = "/runs/lightning_logs/checkpoints"  # Chemin où les checkpoints seront enregistrés
# checkpoint_callback = ModelCheckpoint(
#     dirpath=CHECKPOINT_PATH,
#     filename='{epoch}-{val_loss:.2f}',
#     save_top_k=3,  # Nombre maximal de checkpoints à garder
#     verbose=True,
#     monitor='val_loss',  # Métrique à surveiller pour l'enregistrement
#     mode='min'  # 'min' pour sauvegarder lorsque la métrique surveillée a diminué
# )




BATCH_SIZE = 311
TRAIN_RATIO = 0.8
LOG_PATH = "/tmp/runs/lightning_logs"


class Lit2Layer(pl.LightningModule):
    def __init__(self,dim_in,l,dim_out,learning_rate=1e-3):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(dim_in,l),nn.ReLU(),nn.Linear(l,l),nn.ReLU(), nn.Linear(l,dim_out))
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.name = "exemple-lightning"
        self.valid_outputs = []
        self.training_outputs = []
        self.epoch_losses = []  # To store losses for each epoch
    def forward(self,x):
        """ Définit le comportement forward du module"""
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """ Définit l'optimiseur """
        optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        return optimizer

    def training_step(self,batch,batch_idx):
        """ une étape d'apprentissage
        doit retourner soit un scalaire (la loss),
        soit un dictionnaire qui contient au moins la clé 'loss'"""
        x, y = batch
        yhat= self(x) ## equivalent à self.model(x)
        loss = self.loss(yhat,y)
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        self.log("accuracy",acc/len(x),on_step=False,on_epoch=True)
        self.valid_outputs.append({"loss":loss,"accuracy":acc,"nb":len(x)})
        return logs

    def validation_step(self,batch,batch_idx):
        """ une étape de validation
        doit retourner un dictionnaire"""
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat,y)
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        self.log("val_accuracy", acc/len(x),on_step=False,on_epoch=True)
        self.valid_outputs.append({"loss":loss,"accuracy":acc,"nb":len(x)})
        return logs

    def test_step(self,batch,batch_idx):
        """ une étape de test """
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat,y)
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        return logs

    def log_x_end(self,outputs,phase):
        total_acc = sum([o['accuracy'] for o in outputs])
        total_nb = sum([o['nb'] for o in outputs])
        total_loss = sum([o['loss'] for o in outputs])/len(outputs)
        total_acc = total_acc/total_nb
        self.log_dict({f"loss/{phase}":total_loss,f"acc/{phase}":total_acc})
        #self.logger.experiment.add_scalar(f'loss/{phase}',total_loss,self.current_epoch)
        #self.logger.experiment.add_scalar(f'acc/{phase}',total_acc,self.current_epoch)

    def on_training_epoch_end(self):
        """ hook optionel, si on a besoin de faire quelque chose apres une époque d'apprentissage.
        Par exemple ici calculer des valeurs à logger"""

        # Store the average loss for the current epoch
        current_epoch_loss = sum([o['loss'].item() for o in self.training_outputs]) / len(self.training_outputs)
        self.epoch_losses.append(current_epoch_loss)

        # Log the losses every 5 epochs
        if (self.current_epoch + 1) % 5 == 0:
            for epoch, loss in enumerate(self.epoch_losses, 1):
                self.logger.experiment.add_scalar("epoch_loss", loss, epoch)
            
            # Optionally, you can clear the stored losses after logging
            # self.epoch_losses.clear()

        self.log_x_end(self.training_outputs,'train')
        self.training_outputs.clear()
        # Le logger de tensorboard est accessible directement avec self.logger.experiment.add_XXX
    def on_validation_epoch_end(self):
        """ hook optionel, si on a besoin de faire quelque chose apres une époque de validation."""
        self.log_x_end(self.valid_outputs,'valid')
        self.valid_outputs.clear()

    def on_test_epoch_end(self):
        pass




class LitMnistData(pl.LightningDataModule):

    def __init__(self,batch_size=BATCH_SIZE,train_ratio=TRAIN_RATIO):
        super().__init__()
        self.dim_in = None
        self.dim_out = None
        self.batch_size = batch_size
        self.train_ratio = train_ratio

    def prepare_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])
        # Téléchargement et stockage des ensembles de données
        self.mnist_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Dimensions
            self.dim_in = 28 * 28  # MNIST images are 28x28
            self.dim_out = 10  # 10 classes
            # Division en train et validation
            train_length = int(len(self.mnist_train_dataset) * self.train_ratio)
            self.mnist_train, self.mnist_val = random_split(self.mnist_train_dataset, [train_length, len(self.mnist_train_dataset) - train_length])
        if stage == "test" or stage is None:
            self.mnist_test = self.mnist_test_dataset


    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=3)
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=3)
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=3)



data = LitMnistData()

data.prepare_data()
data.setup(stage="fit")

model = Lit2Layer(data.dim_in,80,data.dim_out,learning_rate=1e-3)

logger = TensorBoardLogger(save_dir=LOG_PATH,name=model.name,version=time.asctime(),default_hp_metric=False)

trainer = pl.Trainer(default_root_dir=LOG_PATH,logger=logger,max_epochs=500)#, callbacks=[checkpoint_callback])
trainer.fit(model,data)
trainer.test(model,data)

# !tensorboard --logdir=/tmp/runs/lightning_logs



def on_after_backward(self):
    for name, param in self.named_parameters():
        if "weight" in name:
            self.logger.experiment.add_histogram(name, param, self.current_epoch)
def on_after_backward(self):
    for name, param in self.named_parameters():
        if param.grad is not None:
            self.logger.experiment.add_histogram(f"{name}_grad", param.grad, self.current_epoch)
def validation_step(self, batch, batch_idx):
    # ... Votre code existant pour calculer la loss et l'accuracy ...
    entropy = calculate_entropy(yhat)  # Utilisez la fonction calculate_entropy définie précédemment
    self.log("val_entropy", entropy, on_step=False, on_epoch=True)
    return logs
