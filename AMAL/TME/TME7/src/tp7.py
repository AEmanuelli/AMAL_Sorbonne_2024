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
EPOCHS = 100


# Téléchargement des données
ds = prepare_dataset("com.lecun.mnist")
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels = ds.test.images.data(), ds.test.labels.data()


# Préparation des jeux de données
full_train_dataset = MyDataset(train_images, train_labels)
train_size = int(TRAIN_RATIO * len(full_train_dataset))
subset_train_dataset, _ = random_split(full_train_dataset, [train_size, len(full_train_dataset) - train_size])

train_loader = DataLoader(subset_train_dataset, batch_size=300, shuffle=True)
test_loader = DataLoader(MyDataset(test_images, test_labels), batch_size=300, shuffle=False)
print(f"Nombre de batches dans test_loader : {len(test_loader)}")

# Tensorboard : rappel, lancer dans une console tensorboard --logdir AMAL/TME/TME7/src/runs/runs
writer = SummaryWriter("AMAL/AMAL/TME/TME7/src/runs/runs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleNN(nn.Module):
    def __init__(self, dropout_rate=0.5, use_batchnorm=False):
        super(SimpleNN, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.layer1 = nn.Linear(28 * 28, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 100)
        self.classifier = nn.Linear(100, EPOCHS)
        self.dropout = nn.Dropout(dropout_rate)

        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm1d(100)
            self.bn2 = nn.BatchNorm1d(100)
            self.bn3 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = x.to(self.layer1.weight.dtype)
        x = self.layer1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.layer2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.layer3(x)
        if self.use_batchnorm:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.classifier(x)
        return x

def train(model, train_loader, optimizer, criterion, device, epoch, l1_lambda=0.0):
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

        # Affichage des informations
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader.dataset)
    writer.add_scalar('Loss/train', avg_loss, epoch)

# Ajouter une fonction pour calculer l'entropie
def calculate_entropy(output):
    p = F.softmax(output, dim=1)
    log_p = torch.log2(p)
    entropy = -torch.sum(p * log_p, dim=1)
    mean_entropy = torch.mean(entropy)
    if torch.isnan(mean_entropy):
        print("Attention : l'entropie moyenne est NaN.")
    return mean_entropy

def test(model, test_loader, device, epoch):
    model.eval()
    entropy_list = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            
            if torch.isnan(output).any():
                print(f"Attention : les sorties contiennent des NaN au batch {batch_idx+1}.")
                continue  # Passer au batch suivant
            
            entropy = calculate_entropy(output)
            if torch.isnan(entropy):
                print(f"Attention : l'entropie est NaN au batch {batch_idx+1}.")
                continue  # Passer au batch suivant
            
            entropy_list.append(entropy.item())

    if len(entropy_list) > 0:
        avg_entropy = sum(entropy_list) / len(entropy_list)
        writer.add_scalar('Entropy/test', avg_entropy, epoch)
        writer.add_histogram('Entropy/test_hist', torch.tensor(entropy_list), epoch)
    else:
        print("Aucune entropie calculée, `entropy_list` est vide.")


# Define and train models for each regularization case


# Cas 1 : Sans régularisation
model_no_reg = SimpleNN(dropout_rate=0.0, use_batchnorm=False).to(device)
optimizer_no_reg = torch.optim.Adam(model_no_reg.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    train(model_no_reg, train_loader, optimizer_no_reg, criterion, device, epoch)
    test(model_no_reg, test_loader, device, epoch)


# Cas 2 : Régularisation L1
model_l1_reg = SimpleNN(dropout_rate=0.0, use_batchnorm=False).to(device)
optimizer_l1_reg = torch.optim.Adam(model_l1_reg.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
l1_lambda = 0.001  # Coefficient de régularisation L1
for epoch in range():
    train(model_l1_reg, train_loader, optimizer_l1_reg, criterion, device, epoch, l1_lambda=l1_lambda)
    test(model_l1_reg, test_loader, device, epoch)

# Cas 3 : Régularisation L2 via weight_decay
model_l2_reg = SimpleNN(dropout_rate=0.0, use_batchnorm=False).to(device)
l2_lambda = 0.01  # Coefficient de régularisation L2
optimizer_l2_reg = torch.optim.Adam(model_l2_reg.parameters(), lr=0.001, weight_decay=l2_lambda)
criterion = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    train(model_l2_reg, train_loader, optimizer_l2_reg, criterion, device, epoch)
    test(model_l2_reg, test_loader, device, epoch)


# Cas 4 : Avec Dropout
model_dropout = SimpleNN(dropout_rate=0.5, use_batchnorm=False).to(device)
optimizer_dropout = torch.optim.Adam(model_dropout.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    train(model_dropout, train_loader, optimizer_dropout, criterion, device, epoch)
    test(model_dropout, test_loader, device, epoch)

# Cas 5 : Avec Batch Normalization
model_batchnorm = SimpleNN(dropout_rate=0.0, use_batchnorm=True).to(device)
optimizer_batchnorm = torch.optim.Adam(model_batchnorm.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    train(model_batchnorm, train_loader, optimizer_batchnorm, criterion, device, epoch)
    test(model_batchnorm, test_loader, device, epoch)


# Cas 6 : Avec Dropout et Batch Normalization
model_dropout_bn = SimpleNN(dropout_rate=0.5, use_batchnorm=True).to(device)
optimizer_dropout_bn = torch.optim.Adam(model_dropout_bn.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    train(model_dropout_bn, train_loader, optimizer_dropout_bn, criterion, device, epoch)
    test(model_dropout_bn, test_loader, device, epoch)

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
