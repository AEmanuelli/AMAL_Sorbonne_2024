import logging
import os
from pathlib import Path
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Dataset
from datamaestro import prepare_dataset
# Removed unused import

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
EPOCHS = 100        # Nombre d'époques
BATCH_SIZE = 300    # Taille de batch

# Transformations pour l'augmentation de données
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.1),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class MyDataset(Dataset):
    def __init__(self, data, label, transform=None) -> None:
        super().__init__()
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        image = self.data[index]
        label = self.label[index]
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32) / 255.0
            image = image.view(-1)
        return image, label

    def __len__(self):
        return len(self.data)

class LitMNISTDataModule(pl.LightningDataModule):
    def __init__(self, train_transforms=None, test_transforms=None):
        super().__init__()
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

    def prepare_data(self):
        # Téléchargement des données
        self.ds = prepare_dataset("com.lecun.mnist")
        self.train_images = self.ds.train.images.data()
        self.train_labels = self.ds.train.labels.data()
        self.test_images = self.ds.test.images.data()
        self.test_labels = self.ds.test.labels.data()

    def setup(self, stage=None):
        # stage is not used, but it's kept for compatibility with LightningDataModule
        # Préparation des datasets
        full_train_dataset = MyDataset(self.train_images, self.train_labels, transform=self.train_transforms)

        train_size = int(TRAIN_RATIO * len(full_train_dataset))
        # Removed unused variable test_size
        test_size = len(full_train_dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, _ = random_split(
            full_train_dataset, [train_size, val_size, len(full_train_dataset) - train_size - val_size])

        self.test_dataset = MyDataset(self.test_images, self.test_labels, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=BATCH_SIZE)

class SimpleNN(pl.LightningModule):
    def __init__(self, dropout_rate=0.5, use_batchnorm=False, l1_lambda=0.0, l2_lambda=0.0):
        super(SimpleNN, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        self.layer1 = nn.Linear(28 * 28, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 100)
        self.classifier = nn.Linear(100, 10)
        self.dropout = nn.Dropout(dropout_rate)

        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm1d(100)
            self.bn2 = nn.BatchNorm1d(100)
            self.bn3 = nn.BatchNorm1d(100)

        # Dictionnaire pour stocker les gradients des entrées de chaque couche
        self.input_grads = {}

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Aplatir les images
        x = x.to(self.layer1.weight.dtype)
        x = self.layer1(x)
        if x.requires_grad:
            x.register_hook(lambda grad: self.input_grads.setdefault('layer1_input', grad))
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.layer2(x)
        if x.requires_grad:
            x.register_hook(lambda grad: self.input_grads.setdefault('layer2_input', grad))
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.layer3(x)
        if x.requires_grad:
            x.register_hook(lambda grad: self.input_grads.setdefault('layer3_input', grad))
        if self.use_batchnorm:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=self.l2_lambda)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)

        # Régularisation L1
        if self.l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in self.parameters())
            loss += self.l1_lambda * l1_norm

        # Log du coût d'entraînement
        self.log('Loss/train', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        total = data.size(0)
        accuracy = correct / total

        # Log du coût de validation et de l'exactitude
        self.log('Loss/validation', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('Accuracy/validation', accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        total = data.size(0)
        accuracy = correct / total

        # Calcul de l'entropie
        entropy = self.calculate_entropy(output)

        # Log du coût de test, de l'exactitude et de l'entropie
        self.log('Loss/test', loss, on_step=False, on_epoch=True)
        self.log('Accuracy/test', accuracy, on_step=False, on_epoch=True)
        self.log('Entropy/test', entropy, on_step=False, on_epoch=True)

    def calculate_entropy(self, output):
        p = F.softmax(output, dim=1)
        log_p = torch.log2(p + 1e-9)
        entropy = -torch.sum(p * log_p, dim=1)
        mean_entropy = torch.mean(entropy)
        if torch.isnan(mean_entropy):
            print("Attention : l'entropie moyenne est NaN.")
            mean_entropy = torch.tensor(0.0)
        return mean_entropy

    def training_epoch_end(self, outputs):
        # Enregistrement des poids et des gradients (environ 20 fois pendant l'entraînement)
        epoch = self.current_epoch
        if epoch % max(1, EPOCHS // 20) == 0:
            for name, param in self.named_parameters():
                if 'weight' in name:
                    self.logger.experiment.add_histogram(f"Weights/{name}", param, epoch)
            for name, grad in self.input_grads.items():
                if grad is not None:
                    self.logger.experiment.add_histogram(f"Gradients/{name}", grad, epoch)
            self.input_grads.clear()

    def test_epoch_end(self, outputs):
        # Enregistrement de l'entropie sur la sortie (histogramme)
        epoch = self.current_epoch
        entropy_values = [self.calculate_entropy(output['output']) for output in outputs]
        if epoch % max(1, EPOCHS // 20) == 0:
            self.logger.experiment.add_histogram('Entropy/test_hist', torch.tensor(entropy_values), epoch)

# Création du module de données
data_module = LitMNISTDataModule(train_transforms=train_transforms, test_transforms=test_transforms)

# Configuration du modèle
model = SimpleNN(
    dropout_rate=0.0,  # Ajuster en fonction du cas
    use_batchnorm=False,  # Ajuster en fonction du cas
    l1_lambda=0.0,  # Régularisation L1
    l2_lambda=0.0   # Régularisation L2
)

# Configuration du logger TensorBoard
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
workingdir = Path(f"./logs/{timestamp}/")
workingdir.mkdir(parents=True, exist_ok=True)
tb_logger = pl.loggers.TensorBoardLogger(save_dir=workingdir, name="no_regularization")

# Configuration du checkpointing
checkpoint_callback = ModelCheckpoint(
    monitor='Loss/validation',
    dirpath=workingdir / "checkpoints",
    filename='model-{epoch:02d}-{Loss/validation:.2f}',
    save_top_k=1,
    mode='min',
)
# Création du Trainer
trainer = Trainer(
    max_epochs=EPOCHS,
    logger=tb_logger,
    callbacks=[checkpoint_callback],
    devices=1 if torch.cuda.is_available() else 1,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu'
)


# Entraînement du modèle
trainer.fit(model, datamodule=data_module)

# Test du modèle
trainer.test(model, datamodule=data_module)
