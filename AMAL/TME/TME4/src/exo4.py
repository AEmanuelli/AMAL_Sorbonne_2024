from pathlib import Path
import string
import unicodedata
import torch
import sys
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

from utils import *

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]


#  TODO: 
PATH = "AMAL/TME/TME4/data/"
DIM_INPUT = len(id2lettre)
DIM_OUTPUT = len(id2lettre)
BATCH_SIZE = 128
hidden_size = 250
lr = 5e-4
total_epoch = 500
max_len = 50
data_trump = DataLoader(TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=max_len), batch_size= BATCH_SIZE, shuffle=True)

model = RNN(DIM_INPUT, hidden_size, DIM_OUTPUT
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Saves checkpoint to disk"""
    torch.save(state, filename)
def load_checkpoint(checkpoint_path, model, optimizer):
    if Path(checkpoint_path).is_file():
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch']
    return 0


# def generate_text(seed, length=100):
#     model.eval()
#     generated = seed
#     input_seq = string2code(seed).unsqueeze(0)
#     input_seq = nn.functional.one_hot(input_seq, num_classes=DIM_INPUT).to(device).float()

#     with torch.no_grad():
#         for i in range(length):  # Use 'i' for iteration count
#             h = torch.zeros(x.size(0), hidden_size, device=device)
#             h = model(input_seq, h)
#             output = h[0, -1] / 1  # Apply temperature
#             probabilities = torch.nn.functional.softmax(output, dim=0)
#             next_char_idx = torch.multinomial(probabilities, 1).item()
#             next_char_idx = max(0, min(next_char_idx, DIM_INPUT - 1))
#             next_char = id2lettre.get(next_char_idx, '')  # Safe retrieval from dictionar

#             generated += next_char
#             next_input = torch.tensor([[next_char_idx]], device=device)
#             next_input = nn.functional.one_hot(next_input, num_classes=DIM_INPUT).float()
#             input_seq = torch.cat([input_seq, next_input], dim=1)

#     return generated

import torch.nn.functional as F

def temperature_sampling(logits, temperature=1.0):
    """
    Apply temperature sampling to logits.
    """
    if temperature <= 0:  # Avoid division by zero
        raise ValueError("Temperature should be greater than 0")

    # Apply temperature
    probs = F.softmax(logits / temperature, dim=-1)

    # Sample from the probability distribution
    next_char_idx = torch.multinomial(probs, num_samples=1)
    return next_char_idx

def generate_text(model, start_string="Trump", generation_length=100, temperature=0.5):
    input_eval = string2code(start_string)  # Convert starting string to tensor
    input_eval = input_eval.unsqueeze(0)  # Add batch dimension

    generated_text = start_string

    model.eval()  # Evaluation mode

    with torch.no_grad():
        h = torch.zeros([1, model.latent_dim], device=input_eval.device)  # Initialize hidden state

        for i in range(generation_length):
            # Update hidden state
            h = model.one_step(nn.functional.one_hot(input_eval[:, -1], num_classes=len(lettre2id)).float(), h)

            # Get logits from the hidden state
            logits = model.decode(h)

            # Apply temperature sampling
            next_char_idx = temperature_sampling(logits, temperature)
            next_char_idx = next_char_idx.squeeze().tolist()  # Convert to list

            # Check if it's a single integer, convert to list if so
            if isinstance(next_char_idx, int):
                next_char_idx = [next_char_idx]

            next_char = code2string(next_char_idx)

            generated_text += next_char

            # Update input for next generation step
            next_char_tensor = torch.tensor([next_char_idx], device=input_eval.device)
            input_eval = torch.cat((input_eval, next_char_tensor), dim=1)

    return generated_text





def generate(model):
    # model.load_state_dict(torch.load(PATH+f"{model_name}.pt"))
    for _ in range(2):
        h = torch.zeros(
            (1, hidden_size), device=device
        )
        generated = [torch.tensor(torch.randint(len(lettre2id), (1,))).to(device)]
        model.eval()
        for i in range(max_len):
            h = model.one_step(nn.functional.one_hot(generated[-1], num_classes=len(lettre2id)).float(), h)
            generated.append(model.decode(h).argmax(1))
        generated = torch.stack(generated[1:])
        print("".join([id2lettre[int(i)] for i in generated.squeeze()]))

def train(model, data_loader, criterion, optimizer):
    checkpoint_path = PATH + f"{model_name}.pth.tar"
    start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
    for epoch in tqdm(range(start_epoch, total_epoch)):
        model.train()
        total_loss = 0

        for x, y in tqdm((data_loader)):
            # Réinitialisation de l'état caché pour chaque batch
            h = torch.zeros(x.size(0), hidden_size, device=device)

            # Préparation des données
            y = nn.functional.one_hot(y, num_classes=DIM_OUTPUT).to(device).float()
            x = nn.functional.one_hot(x, num_classes=DIM_OUTPUT).to(device).float()

            # Forward pass
            optimizer.zero_grad()
            h = model(x, h)
            y_hat = model.decode(h)

            # Calcul de la perte
            loss = criterion(y_hat, y)  # Ajustement pour CrossEntropyLoss
            total_loss += loss.item()

            # Backward pass et optimisation
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Clipping des gradients
            optimizer.step()
        print(code2string(y[0,:,:].argmax(1)))
        print(code2string(y_hat[0,:,:].argmax(1)))
        print(generate_text(model))
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch + 1}/{total_epoch}, Loss: {avg_loss:.4f}')
        if epoch % 5 == 0 or epoch == total_epoch - 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, filename=PATH + f"{model_name}.pth.tar")





#################@Training loop with checkpointing
# savepath = Path("AMAL/TME/TME4/src/trump_3.pch")
# state = State(model, optimizer, device, savepath)
# # Training Loop
# for epoch in tqdm(range(state.epoch, total_epoch)):
#     epoch_loss = 0
#     state.model.train()
#     for x, y in tqdm(data_trump):
#         x = nn.functional.one_hot(x, num_classes=DIM_INPUT).to(device).float()
#         y = y.to(device)
#         state.optim.zero_grad()
#         output = model(x)
#         loss = criterion(output.transpose(1, 2), y)
#         loss.backward()
#         state.optim.step()
#         epoch_loss += loss.item()
#         state.iteration += 1
#     print(f"Epoch {epoch+1}/{total_epoch}, Loss: {epoch_loss / len(data_trump)}")
#     # Save the state at the end of each epoch
#     with savepath.open("wb") as fp:
#         state.epoch = epoch + 1
#         torch.save({
#             'epoch': state.epoch,
#             'iteration': state.iteration,
#             'model_state_dict': state.model.state_dict(),
#             'optimizer_state_dict': state.optim.state_dict()
#         }, fp)




# Train and generate text
model_name = "final_"
train(model, data_trump, criterion, optimizer)