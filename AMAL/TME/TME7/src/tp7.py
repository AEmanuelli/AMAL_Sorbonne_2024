import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import datetime
from datamaestro import prepare_dataset


class MyDataset(Dataset):
    def __init__(self, data, label) -> None:
        super().__init__()
        self.data = data.reshape((len(data), len(data[0]) ** 2)) / 255
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
    

# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05

# Téléchargement des données
ds = prepare_dataset("com.lecun.mnist")
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels = ds.test.images.data(), ds.test.labels.data()
# Préparation des jeux de données
full_train_dataset = MyDataset(train_images, train_labels)
train_size = int(TRAIN_RATIO * len(full_train_dataset))
subset_train_dataset, _ = random_split(full_train_dataset, [train_size, len(full_train_dataset) - train_size])

train_loader = DataLoader(subset_train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(MyDataset(test_images, test_labels), batch_size=1000, shuffle=False)

# Tensorboard : rappel, lancer dans une console tensorboard --logdir AMAL/TME/TME7/src/runs/runs
writer = SummaryWriter("AMAL/TME/TME7/src/runs/runs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDataset(Dataset):
    def __init__(self, data, label) -> None:
        super().__init__()
        self.data = data.reshape((len(data), len(data[0]) ** 2)) / 255
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 100)  # Première couche linéaire
        self.layer2 = nn.Linear(100, 100)      # Deuxième couche linéaire
        self.layer3 = nn.Linear(100, 100)      # Troisième couche linéaire
        self.classifier = nn.Linear(100, 10)   # Couche de classification

    def forward(self, x):
        x = x.to(self.layer1.weight.dtype)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.classifier(x)
        return x

def train(model, train_loader, optimizer, criterion, device, epoch, l1_lambda=0.001, l2_lambda=0.01):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculate the standard CrossEntropyLoss
        loss = criterion(output, target)

        # L1 Regularization: Add L1 norm of parameters to the loss
        if l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_norm

        # L2 Regularization: Handled through optimizer's weight decay
        # You don't need to explicitly add L2 regularization term

        # Backward pass
        loss.backward()
        
        # Enregistrement des gradients et des poids
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
            writer.add_histogram(f'{name}.grad', param.grad, epoch)

        # Mise à jour des paramètres
        optimizer.step()

        total_loss += loss.item()

        # Affichage des informations
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader.dataset)
    writer.add_scalar('Loss/train', avg_loss, epoch)

# Ajouter une fonction pour calculer l'entropie
def calculate_entropy(output):
    p = F.softmax(output, dim=1)
    log_p = torch.log2(p)
    entropy = -torch.sum(p * log_p, dim=1)
    return entropy.mean()

# Fonction pour tester et enregistrer l'entropie
def test(model, test_loader, device, epoch):
    model.eval()
    entropy_list = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            entropy = calculate_entropy(output)
            entropy_list.append(entropy.item())

    avg_entropy = sum(entropy_list) / len(entropy_list)
    writer.add_scalar('Entropy/test', avg_entropy, epoch)
    writer.add_histogram('Entropy/test_hist', torch.tensor(entropy_list), epoch)

# Define and train models for each regularization case

# Case 1: No Regularization
model_no_reg = SimpleNN().to(device)
optimizer_no_reg = torch.optim.Adam(model_no_reg.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(subset_train_dataset, batch_size=64, shuffle=True)
for epoch in range(10):
    train(model_no_reg, train_loader, optimizer_no_reg, criterion, device, epoch)
    test(model_no_reg, test_loader, device, epoch)

# Case 2: L1 Regularization
model_l1_reg = SimpleNN().to(device)
optimizer_l1_reg = torch.optim.Adam(model_l1_reg.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(subset_train_dataset, batch_size=64, shuffle=True)
for epoch in range(10):
    train(model_l1_reg, train_loader, optimizer_l1_reg, criterion, device, epoch, l1_lambda=0.001)

# Case 3: L2 Regularization (handled through optimizer's weight decay)
model_l2_reg = SimpleNN().to(device)
optimizer_l2_reg = torch.optim.Adam(model_l2_reg.parameters(), lr=0.001, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(subset_train_dataset, batch_size=64, shuffle=True)
for epoch in range(10):
    train(model_l2_reg, train_loader, optimizer_l2_reg, criterion, device, epoch)

# Fermeture du writer TensorBoard
writer.close()


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
