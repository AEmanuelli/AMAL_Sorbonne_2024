import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data, label, transform=None) -> None:
        super().__init__()
        self.data = data  # Garder les données sous forme d'images
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        image = self.data[index]
        label = self.label[index]
        if self.transform:
            image = self.transform(image)
        else:
            # Normalisation par défaut si aucune transformation n'est fournie
            image = torch.tensor(image, dtype=torch.float32) / 255.0
            image = image.view(-1)  # Aplatir l'image
        return image, label

    def __len__(self):
        return len(self.data)

class NumpyToTensor:
    def __call__(self, pic):
        # Forcer la copie du tableau NumPy si nécessaire
        if isinstance(pic, np.ndarray) and not pic.flags.writeable:
            pic = np.copy(pic)  # Crée une copie modifiable
        
        # Ajout de dimension pour les images en niveaux de gris
        if pic.ndim == 2:  # Image de forme (Hauteur, Largeur)
            pic = np.expand_dims(pic, axis=0)  # Ajoute une dimension (1, Hauteur, Largeur)
        
        return torch.from_numpy(pic).contiguous()

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


class SimpleNN(nn.Module):
    def __init__(self, dropout_rate=0.5, use_batchnorm=False, use_layernorm=False):
        super(SimpleNN, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm
        self.layer1 = nn.Linear(28 * 28, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 100)
        self.classifier = nn.Linear(100, 10)
        self.dropout = nn.Dropout(dropout_rate)

        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm1d(100)
            self.bn2 = nn.BatchNorm1d(100)
            self.bn3 = nn.BatchNorm1d(100)

        if self.use_layernorm:
            self.ln1 = nn.LayerNorm(100)
            self.ln2 = nn.LayerNorm(100)
            self.ln3 = nn.LayerNorm(100)        
            

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