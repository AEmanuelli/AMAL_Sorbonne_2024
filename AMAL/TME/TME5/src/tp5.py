from pathlib import Path
import string
import torch
import sys
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  TODO: 
PATH = "AMAL/TME/TME4/data/"
DIM_INPUT = len(id2lettre)
DIM_OUTPUT = len(id2lettre)
EMBEDDING_DIM = 50
BATCH_SIZE = 128
hidden_size = 250
lr = 5e-4
total_epoch = 500
max_len = 60
data_trump = DataLoader(TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=max_len), batch_size= BATCH_SIZE, shuffle=True)

embedding = nn.Embedding(num_embeddings=DIM_INPUT, embedding_dim=EMBEDDING_DIM)
model = RNN(EMBEDDING_DIM, hidden_size, DIM_OUTPUT).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.parameters()) + list(embedding.parameters()))

def train(model, data_loader, criterion, optimizer):
    checkpoint_path = PATH + f"{model_name}.pth.tar"
    start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
    for epoch in tqdm(range(start_epoch, total_epoch)):
        model.train()
        total_loss = 0
        for x, y in tqdm((data_loader)):
            # Réinitialisation de l'état caché pour chaque batch
            x_embedded = embedding(x)
            y_embedded = embedding(y)
            h = torch.zeros(x_embedded.size(0), hidden_size, device=device)
            # Forward pass
            optimizer.zero_grad()
            h = model(x_embedded, h)
            y_hat = model.decode(h)

            # Calcul de la perte
            loss = criterion(y_hat.transpose(1, 2), y)
            total_loss += loss.item()

            # Backward pass et optimisation
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Clipping des gradients
            optimizer.step()
        print(code2string(y[0,:]))
        print(code2string(y_hat[0,:,:].argmax(1)))
        print(generate_text(model, embedding=embedding))
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch + 1}/{total_epoch}, Loss: {avg_loss:.4f}')
        if epoch % 5 == 0 or epoch == total_epoch - 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, filename=PATH + f"{model_name}.pth.tar")


# Train and generate text
model_name = "tp5"
train(model, data_trump, criterion, optimizer)