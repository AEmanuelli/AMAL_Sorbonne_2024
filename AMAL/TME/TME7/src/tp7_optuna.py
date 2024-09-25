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
import torchvision.transforms as transforms
import numpy as np
from tp7_classes import MyDataset, SimpleNN, AddGaussianNoise, NumpyToTensor
import optuna  # Importer Optuna

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Affichage des informations sur l'environnement
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Python executable: {os.path.abspath(os.sys.executable)}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"CUDA version: {torch.version.cuda}")
logger.info(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Ratios pour les jeux de données
TRAIN_RATIO = 0.05  # Utiliser 5% des données pour l'entraînement
VAL_RATIO = 0.05    # Utiliser 5% des données pour la validation
EPOCHS = 50         # Réduire le nombre d'époques pour accélérer la recherche

# Transformation (pas de transformations gaussiennes pour le moment)
transform = None

# Téléchargement des données
ds = prepare_dataset("com.lecun.mnist")
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels = ds.test.images.data(), ds.test.labels.data()

# Préparation des jeux de données avec les transformations
full_train_dataset = MyDataset(train_images, train_labels, transform=transform)

train_size = int(TRAIN_RATIO * len(full_train_dataset))
val_size = int(VAL_RATIO * len(full_train_dataset))

subset_train_dataset, subset_val_dataset, _ = random_split(
    full_train_dataset, [train_size, val_size, len(full_train_dataset) - train_size - val_size])

train_loader = DataLoader(subset_train_dataset, batch_size=300, shuffle=True)
val_loader = DataLoader(subset_val_dataset, batch_size=300, shuffle=False)
test_loader = DataLoader(MyDataset(test_images, test_labels), batch_size=300, shuffle=False)

print(f"Nombre de batches dans test_loader : {len(test_loader)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, optimizer, criterion, device, l1_lambda=0.0):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Régularisation L1
        if l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_norm

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

def validate(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def test(model, test_loader, device, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# Définition de la fonction objective pour Optuna
def objective(trial):
    # Définition des hyperparamètres à optimiser
    lr = trial.suggest_categorical('lr', [5e-4, 6e-4, 7e-4, 8e-4])
    dropout_rate = trial.suggest_categorical('dropout_rate', [i * 0.005 for i in range(8)])
    l1_lambda = trial.suggest_float('l1_lambda', 0.0, 1e-4)
    l2_lambda = trial.suggest_float('l2_lambda', 0.0, 1e-4)
    use_batchnorm = trial.suggest_categorical('use_batchnorm', [True])

    # Création du modèle avec les hyperparamètres du trial
    model = SimpleNN(dropout_rate=dropout_rate, use_batchnorm=use_batchnorm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device, l1_lambda=l1_lambda)
        test_loss, test_acc = test(model, val_loader, device, criterion)

        # Rapport à Optuna
        trial.report(test_loss, epoch)

        # Arrêt anticipé si nécessaire
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return test_loss  # Ou -val_accuracy si vous souhaitez maximiser l'exactitude

# Création de l'étude et optimisation
study = optuna.create_study(direction='minimize')  # 'maximize' si vous optimisez l'exactitude
study.optimize(objective, n_trials=300)

# Affichage des meilleurs hyperparamètres
print("Best hyperparameters: ", study.best_params)
# {'lr': 0.007830503662215669, 'dropout_rate': 0.09602871652862545, 'l1_lambda': 1.6883058744681504e-06, 'l2_lambda': 0.00036554020142375715, 'use_batchnorm': True}
# {'lr': 0.0064221391487605676, 'dropout_rate': 0.08152420792112025, 'l1_lambda': 3.162196781231575e-05, 'l2_lambda': 0.00041721456690074933, 'use_batchnorm': True}
# {'lr': 0.0008, 'dropout_rate': 0.0, 'l1_lambda': 2.694811477767316e-06, 'l2_lambda': 1.5149881302603684e-05, 'use_batchnorm': True}