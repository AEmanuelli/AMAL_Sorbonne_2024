from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from pathlib import Path


class MonDataset(Dataset) :
    def __init__(self, images, labels):
            self.images = images
            self.labels = labels
    
    def __len__(self):
        # Retourne la taille du dataset
        return len(self.images)
    
    def __getitem__(self, index):
        # Récupère l'image à l'index spécifié
        img = torch.tensor(self.images[index]) / 255.0  # Normalisation pour avoir les valeurs entre 0 et 1
        label = self.labels[index]
        return img, label
    

class Autoencodeur(nn.Module):
    def __init__(self):
        super().__init__()
        # Définition de l'encodeur
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),  # MNIST images are 28x28 pixels, 128 est un exemple de taille d'espace latent
            nn.ReLU(True)  # Activation ReLU
        )
        # Définition du décodeur
        self.decoder = nn.Sequential(
            nn.Linear(128, 28*28),  # Reconstruire à partir de l'espace latent vers l'image originale
            nn.Sigmoid()  # Activation Sigmoid pour obtenir des valeurs de pixel entre 0 et 1
        )

    def forward(self, x):
        x = self.encoder(x)  # Encode les données d'entrée
        x = self.decoder(x)  # Décode les données pour reconstruire l'entrée
        return x
    

class State:
    def __init__(self, model, optim, device, savepath = ""):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0, 0
        self.savepath = savepath
        self.device = device
        # Check if we have a saved state and load it
        if savepath.is_file():
            with savepath.open("rb") as fp:
                state = torch.load(fp)  # Restart from saved model
                self.model.load_state_dict(state['model_state_dict'])
                self.optim.load_state_dict(state['optimizer_state_dict'])
                self.epoch = state['epoch']
                self.iteration = state['iteration']
        else:
            # Initialize model and optimizer here
            self.model = model.to(device)
            self.optim = optim
