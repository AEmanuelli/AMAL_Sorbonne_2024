import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click

from datamaestro import prepare_dataset
from torchvision import datasets, transforms

def prepare_data(self):
    # Transformations à appliquer sur les images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Téléchargement des ensembles de données
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)



# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05

def store_grad(var):
    """Stores the gradient during backward

    For a tensor x, call `store_grad(x)`
    before `loss.backward`. The gradient will be available
    as `x.grad`

    """
    def hook(grad):
        var.grad = grad
    var.register_hook(hook)
    return var


#  TODO:  Implémenter
# weight decay = True pour la rég L2 
# reg L1 donne plus de sparsité

# def training_step(self, batch, batch_idx):
#         x, y = batch
#         yhat = self(x)
#         loss = self.loss(yhat, y)

#         # Calcul de la régularisation L1
#         l1_lambda = 0.001  # Coefficient de régularisation L1
#         l1_norm = sum(p.abs().sum() for p in self.model.parameters())
#         loss = loss + l1_lambda * l1_norm

#         # ... (le reste du code pour l'étape d'entraînement)
#         return loss



# le dropout est une manière économe d'implémenter du modèle ensembling 
# différence fondamentale en phase de test et en phase de train, c'est ppour ça que tout les modèles de pytorch ont une méthode train et une test. 
# A Chaque batch on génère un nouveua masquage mais pendant un batch 1 seul masqyuage
# modèle convolutionnnels invariants par translation donc focntionnentbien avce images  

# Batchnorm s'occupe du covariateshift en centrant réduisant les données, puis en leur imposant moyenne et variance, permet de ne pas se concentrer sur ces grandeurs 
# là mais de se ocus plutot sur lees moments statistques d'ordre supérieur

# BATCHNORM == pas possible d'envoyer un batch de taille 1.
# entrainement avce de la batchnorm différent du test. la taille du minibatch est un paramètre de régularisation
# histoire de running average
# la régularisation (L1 ?) permet de mettre des plus grands learnings rates 
