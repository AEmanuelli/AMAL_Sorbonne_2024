
import math
import click
from torch.utils.tensorboard import SummaryWriter
import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time
from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def pad_collate_fn(batch):
    """
    Doit faire du padding
    Ajouter un eos à la fin de la phrase et padder le reste
    """
    # Séparation des données et des étiquettes
    data, labels = zip(*batch)

    # Trouver la longueur maximale dans les données
    maxlen = max(len(item) for item in data)

    # Padding des données
    padded_data = []
    for item in data:
        padded_item = list(item) + [1]  # Ajout de EOS
        padded_item += [0] * (maxlen + 1 - len(padded_item))  # Ajout de padding
        padded_data.append(padded_item)

    # Conversion en tensor PyTorch
    padded_data = torch.tensor(padded_data, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_data, labels


MAX_LENGTH = 500

logging.basicConfig(level=logging.INFO)

class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text()), self.filelabels[ix]
    def get_txt(self,ix):
        s = self.files[ix]
        return s if isinstance(s,str) else s.read_text(), self.filelabels[ix]

def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage (embedding_size = [50,100,200,300])

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText) train
    - DataSet (FolderText) test

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset(
        'edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    words.append("__OOV__")
    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(ds.test.classes, ds.test.path, tokenizer, load=False)

#  TODO: 

@click.command()
@click.option('--test-iterations', default=1000, type=int, help='Number of training iterations (batches) before testing')
@click.option('--epochs', default=50, help='Number of epochs.')
@click.option('--modeltype', required=True, type=int, help="0: base, 1 : Attention1, 2: Attention2")
@click.option('--emb-size', default=100, help='embeddings size')
@click.option('--batch-size', default=20, help='batch size')
def main(epochs,test_iterations,modeltype,emb_size,batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word2id, embeddings, train_data, test_data = get_imdb_data(emb_size)
    id2word = dict((v, k) for k, v in word2id.items())
    PAD = word2id["__OOV__"]
    embeddings = torch.Tensor(embeddings)
    emb_layer = nn.Embedding.from_pretrained(torch.Tensor(embeddings))

    def collate(batch):
        """ Collate function for DataLoader """
        data = [torch.LongTensor(item[0][:MAX_LENGTH]) for item in batch]
        lens = [len(d) for d in data]
        labels = [item[1] for item in batch]
        # Création du masque
        mask = torch.tensor([[float(token_id != 0) for token_id in seq] for seq in padded_data])
        mask = mask.unsqueeze(1)  # Redimensionnement pour correspondre aux attentes de la couche d'attention

        return emb_layer(torch.nn.utils.rnn.pad_sequence(data, batch_first=True,padding_value = PAD)).to(device), torch.LongTensor(labels).to(device), torch.Tensor(lens).to(device), mask


    train_loader = DataLoader(train_data, shuffle=True,
                          batch_size=batch_size, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=batch_size,collate_fn=collate,shuffle=False)
    ##  TODO:

class SelfAttentionLayer(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super().__init__()
        self.query = nn.Linear(emb_size, hidden_size)
        self.key = nn.Linear(emb_size, hidden_size)
        self.value = nn.Linear(emb_size, hidden_size)

    def forward(self, x, mask=None):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(hidden_size)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted = torch.bmm(attention_weights, v)
        return weighted


class AttentionBasedModel(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([SelfAttentionLayer(emb_size, hidden_size) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch_size, seq_length, emb_size]
        for layer in self.layers:
            x = layer(x)

        # Utilisation de la moyenne des représentations pour la classification
        x = torch.mean(x, dim=1)  # [batch_size, hidden_size]
        out = self.fc(x)  # [batch_size, output_size]
        return out


embedding_size = 50 
hidden_size = 128    
output_size = 2      

word2id, embeddings, train_dataset, test_dataset = get_imdb_data(embedding_size)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)


model = AttentionBasedModel(embedding_size, hidden_size, output_size)

learning_rate = 0.001
num_epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Boucle d'entraînement
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Validation après l'entraînement
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print(f'Accuracy of the model on the test: {100 * correct / total} %')

if __name__ == "__main__":
    main()


