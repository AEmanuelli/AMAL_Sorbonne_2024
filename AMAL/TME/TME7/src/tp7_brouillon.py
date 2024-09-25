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
EPOCHS = 20      # Nombre d'époques


# Remplacez la transformation ToTensor() par la transformation NumpyToTensor()



# Transformation (pas de transformations gaussiennes pour le moment, j'arrive pas à les faire fonctionner et puis gourmands en energie)
transform = None
# transforms.Compose([
#     NumpyToTensor(),
#     transforms.RandomRotation(10),  # Rotation aléatoire
#     AddGaussianNoise(0., 0.1),  # Ajout de bruit
# ])



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


def train(model, train_loader, optimizer, criterion, device, epoch, writer, l1_lambda=0.0):
    model.train()  # Mettre le modèle en mode entraînement
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calcul de la perte
        loss = criterion(output, target)

        # Régularisation L1
        if l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_norm

        # Backward pass
        loss.backward()
        
        # Mise à jour des paramètres
        optimizer.step()

        total_loss += loss.item() * data.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    return avg_loss

def validate(model, val_loader, device, epoch, writer, criterion):
    model.eval()  # Mettre le modèle en mode évaluation
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
    writer.add_scalar('Loss/validation', avg_loss, epoch)
    writer.add_scalar('Accuracy/validation', accuracy, epoch)
    print(f"Validation Loss: {avg_loss:.6f}, Validation Accuracy: {accuracy:.6f}")
    return avg_loss

def calculate_entropy(output):
    p = F.softmax(output, dim=1)
    log_p = torch.log2(p + 1e-9)  # Ajouter epsilon pour éviter log(0)
    entropy = -torch.sum(p * log_p, dim=1)
    mean_entropy = torch.mean(entropy)
    # Gestion robuste de NaN
    if torch.isnan(mean_entropy):
        print("Attention : l'entropie moyenne est NaN.")
        mean_entropy = torch.tensor(0.0)  # Défaut à 0 si NaN
    return mean_entropy

def test(model, test_loader, device, epoch, writer, criterion):
    model.eval()  # Mettre le modèle en mode évaluation
    entropy_list = []
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Vérification NaN au niveau des sorties du modèle avant entropie
            if torch.isnan(output).any():
                print(f"Attention : les sorties contiennent des NaN au batch {batch_idx+1}.")
                continue
            
            entropy = calculate_entropy(output)
            if torch.isnan(entropy):
                print(f"Attention : l'entropie est NaN au batch {batch_idx+1}.")
                continue
            
            entropy_list.append(entropy.item())
            
            # Calcul de la perte
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            
            # Calcul de la précision
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    if len(entropy_list) > 0:
        avg_entropy = sum(entropy_list) / len(entropy_list)
        writer.add_scalar('Entropy/test', avg_entropy, epoch)
        # Enregistrement de l'histogramme de l'entropie (environ 20 fois)
        if epoch % max(1, EPOCHS // 20) == 0:
            writer.add_histogram('Entropy/test_hist', torch.tensor(entropy_list), epoch)
    else:
        print("Aucune entropie calculée, `entropy_list` est vide.")
    
    avg_loss = total_loss / total
    accuracy = correct / total
    writer.add_scalar('Loss/test', avg_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)
    print(f"Test Loss: {avg_loss:.6f}, Test Accuracy: {accuracy:.6f}")

def compute_random_model_entropy():
    # Calculer l'entropie d'un modèle aléatoire pour comparaison
    random_outputs = torch.randn(1000, 10)
    entropy = calculate_entropy(random_outputs)
    return entropy.item()

# Calcul de l'entropie du modèle aléatoire
random_model_entropy = compute_random_model_entropy()
print(f"Entropie moyenne d'un modèle aléatoire : {random_model_entropy}")

# Boucle d'entraînement pour chaque cas
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
workingdir = Path(f"/Users/apple/Documents/GitHub/AMAL/AMAL/TME/TME7/src/runs/{timestamp}/")
workingdir.mkdir(parents=True, exist_ok=True)

# Cas 1 : Sans régularisation
print("\n=== Cas 1 : Sans régularisation ===")
writer = SummaryWriter(log_dir=workingdir / "no_regularization")
model_no_reg = SimpleNN(dropout_rate=0.0, use_batchnorm=False).to(device)
optimizer_no_reg = torch.optim.Adam(model_no_reg.parameters(), lr=6e-4)
criterion = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS} - No Regularization")
    avg_train_loss = train(model_no_reg, train_loader, optimizer_no_reg, criterion, device, epoch, writer)
    avg_val_loss = validate(model_no_reg, val_loader, device, epoch, writer, criterion)
    test(model_no_reg, test_loader, device, epoch, writer, criterion)

    # Enregistrement des poids (environ 20 fois)
    if epoch % max(1, EPOCHS // 20) == 0:
        for name, param in model_no_reg.named_parameters():
            if 'weight' in name:
                writer.add_histogram(f"Weights/{name}", param, epoch)
        # Enregistrement des gradients
        for name, grad in model_no_reg.input_grads.items():
            if grad is not None:
                writer.add_histogram(f"Gradients/{name}", grad, epoch)
        # Vider le dictionnaire des gradients pour la prochaine époque
        model_no_reg.input_grads.clear()

    print(f"Train Loss after epoch {epoch+1}: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")
writer.close()



# Cas 2 : Régularisation L1
print("\n=== Cas 2 : Régularisation L1 ===")
writer = SummaryWriter(log_dir=workingdir / "l1_regularization")
model_l1_reg = SimpleNN(dropout_rate=0.0, use_batchnorm=False).to(device)
optimizer_l1_reg = torch.optim.Adam(model_l1_reg.parameters(), lr=6e-4)
criterion = nn.CrossEntropyLoss()
l1_lambda = 03.162196781231575e-05 # Coefficient de régularisation L1
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS} - L1 Regularization")
    avg_train_loss = train(model_l1_reg, train_loader, optimizer_l1_reg, criterion, device, epoch, writer, l1_lambda=l1_lambda)
    avg_val_loss = validate(model_l1_reg, val_loader, device, epoch, writer, criterion)
    test(model_l1_reg, test_loader, device, epoch, writer, criterion)
    if epoch % max(1, EPOCHS // 20) == 0:
        for name, param in model_l1_reg.named_parameters():
            if 'weight' in name:
                writer.add_histogram(f"Weights/{name}", param, epoch)
        for name, grad in model_l1_reg.input_grads.items():
            if grad is not None:
                writer.add_histogram(f"Gradients/{name}", grad, epoch)
        model_l1_reg.input_grads.clear()
    print(f"Train Loss after epoch {epoch+1}: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")
writer.close()

# Cas 3 : Régularisation L2 via weight_decay
print("\n=== Cas 3 : Régularisation L2 via weight_decay ===")
writer = SummaryWriter(log_dir=workingdir / "l2_regularization")
model_l2_reg = SimpleNN(dropout_rate=0.0, use_batchnorm=False).to(device)
l2_lambda = 0.00041721456690074933 # Coefficient de régularisation L2
optimizer_l2_reg = torch.optim.Adam(model_l2_reg.parameters(), lr=6e-4, weight_decay=l2_lambda)
criterion = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS} - L2 Regularization")
    avg_train_loss = train(model_l2_reg, train_loader, optimizer_l2_reg, criterion, device, epoch, writer)
    avg_val_loss = validate(model_l2_reg, val_loader, device, epoch, writer, criterion)
    test(model_l2_reg, test_loader, device, epoch, writer, criterion)
    if epoch % max(1, EPOCHS // 20) == 0:
        for name, param in model_l2_reg.named_parameters():
            if 'weight' in name:
                writer.add_histogram(f"Weights/{name}", param, epoch)
        for name, grad in model_l2_reg.input_grads.items():
            if grad is not None:
                writer.add_histogram(f"Gradients/{name}", grad, epoch)
        model_l2_reg.input_grads.clear()
    print(f"Train Loss after epoch {epoch+1}: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")
writer.close()

# Cas 4 : Avec Dropout
print("\n=== Cas 4 : Avec Dropout ===")
writer = SummaryWriter(log_dir=workingdir / "with_dropout")
model_dropout = SimpleNN(dropout_rate=0.081, use_batchnorm=False).to(device)
optimizer_dropout = torch.optim.Adam(model_dropout.parameters(), lr=6e-4)
criterion = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS} - With Dropout")
    avg_train_loss = train(model_dropout, train_loader, optimizer_dropout, criterion, device, epoch, writer)
    avg_val_loss = validate(model_dropout, val_loader, device, epoch, writer, criterion)
    test(model_dropout, test_loader, device, epoch, writer, criterion)
    if epoch % max(1, EPOCHS // 20) == 0:
        for name, param in model_dropout.named_parameters():
            if 'weight' in name:
                writer.add_histogram(f"Weights/{name}", param, epoch)
        for name, grad in model_dropout.input_grads.items():
            if grad is not None:
                writer.add_histogram(f"Gradients/{name}", grad, epoch)
        model_dropout.input_grads.clear()
    print(f"Train Loss after epoch {epoch+1}: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")
writer.close()

# Cas 5 : Avec LayerNorm
print("\n=== Cas 5 : Avec LayerNorm ===")
writer = SummaryWriter(log_dir=workingdir / "with_layernorm")
model_layernorm = SimpleNN(dropout_rate=0.0, use_batchnorm=False, use_layernorm=True).to(device)
optimizer_layernorm = torch.optim.Adam(model_layernorm.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS} - With LayerNorm")
    avg_train_loss = train(model_layernorm, train_loader, optimizer_layernorm, criterion, device, epoch, writer)
    avg_val_loss = validate(model_layernorm, val_loader, device, epoch, writer, criterion)
    test(model_layernorm, test_loader, device, epoch, writer, criterion)
    if epoch % max(1, EPOCHS // 20) == 0:
        for name, param in model_layernorm.named_parameters():
            if 'weight' in name:
                writer.add_histogram(f"Weights/{name}", param, epoch)
        for name, grad in model_layernorm.input_grads.items():
            if grad is not None:
                writer.add_histogram(f"Gradients/{name}", grad, epoch)
        model_layernorm.input_grads.clear()
    print(f"Train Loss after epoch {epoch+1}: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")
writer.close()

# Cas 6 : Avec Batch Normalization
print("\n=== Cas 6 : Avec Batch Normalization ===")
writer = SummaryWriter(log_dir=workingdir / "with_batchnorm")
model_batchnorm = SimpleNN(dropout_rate=0.0, use_batchnorm=True).to(device)
optimizer_batchnorm = torch.optim.Adam(model_batchnorm.parameters(), lr=6e-4)
criterion = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS} - With Batch Normalization")
    avg_train_loss = train(model_batchnorm, train_loader, optimizer_batchnorm, criterion, device, epoch, writer)
    avg_val_loss = validate(model_batchnorm, val_loader, device, epoch, writer, criterion)
    test(model_batchnorm, test_loader, device, epoch, writer, criterion)
    if epoch % max(1, EPOCHS // 20) == 0:
        for name, param in model_batchnorm.named_parameters():
            if 'weight' in name:
                writer.add_histogram(f"Weights/{name}", param, epoch)
        for name, grad in model_batchnorm.input_grads.items():
            if grad is not None:
                writer.add_histogram(f"Gradients/{name}", grad, epoch)
        model_batchnorm.input_grads.clear()
    print(f"Train Loss after epoch {epoch+1}: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")
writer.close()

# Cas 7 : Avec Dropout et Batch Normalization
print("\n=== Cas 7 : Avec Dropout et Batch Normalization ===")
writer = SummaryWriter(log_dir=workingdir / "with_dropout_and_batchnorm")
model_dropout_bn = SimpleNN(dropout_rate=0.5, use_batchnorm=True).to(device)
optimizer_dropout_bn = torch.optim.Adam(model_dropout_bn.parameters(), lr=6e-4)
criterion = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS} - With Dropout and Batch Normalization")
    avg_train_loss = train(model_dropout_bn, train_loader, optimizer_dropout_bn, criterion, device, epoch, writer)
    avg_val_loss = validate(model_dropout_bn, val_loader, device, epoch, writer, criterion)
    test(model_dropout_bn, test_loader, device, epoch, writer, criterion)
    if epoch % max(1, EPOCHS // 20) == 0:
        for name, param in model_dropout_bn.named_parameters():
            if 'weight' in name:
                writer.add_histogram(f"Weights/{name}", param, epoch)
        for name, grad in model_dropout_bn.input_grads.items():
            if grad is not None:
                writer.add_histogram(f"Gradients/{name}", grad, epoch)
        model_dropout_bn.input_grads.clear()
    print(f"Train Loss after epoch {epoch+1}: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")
writer.close()

# Cas 8 : Avec Dropout, Batch Normalization et Régularisation L1 et L2
print("\n=== Cas 8 : Avec Dropout, Batch Normalization et Régularisation L1 et L2 après recherche des meilleurs params===")

 # {'lr': 0.0064221391487605676, 'dropout_rate': 0.08152420792112025, 'l1_lambda': 3.162196781231575e-05, 'l2_lambda': 0.00041721456690074933, 'use_batchnorm': True}

lr = 6e-4
dropout_rate =  0.08152420792112025
l1_lambda = 03.162196781231575e-05
l2_lambda = 0.00041721456690074933

writer = SummaryWriter(log_dir=workingdir / "with_all_regularizations")
model_all_reg = SimpleNN(dropout_rate=dropout_rate, use_batchnorm=True).to(device)

l1_lambda = l1_lambda  # Coefficient de régularisation L1
l2_lambda = l2_lambda   # Coefficient de régularisation L2
optimizer_all_reg = torch.optim.Adam(model_all_reg.parameters(), lr=lr, weight_decay=l2_lambda)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS} - With All Regularizations")
    avg_train_loss = train(model_all_reg, train_loader, optimizer_all_reg, criterion, device, epoch, writer, l1_lambda=l1_lambda)
    avg_val_loss = validate(model_all_reg, val_loader, device, epoch, writer, criterion)
    test(model_all_reg, test_loader, device, epoch, writer, criterion)
    if epoch % max(1, EPOCHS // 20) == 0:
        for name, param in model_all_reg.named_parameters():
            if 'weight' in name:
                writer.add_histogram(f"Weights/{name}", param, epoch)
        for name, grad in model_all_reg.input_grads.items():
            if grad is not None:
                writer.add_histogram(f"Gradients/{name}", grad, epoch)
        model_all_reg.input_grads.clear()
    print(f"Train Loss after epoch {epoch+1}: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")
writer.close()

