import string
import unicodedata
import torch
import sys
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader

from utils import RNN, device

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
batch_size = 30
DIM_INPUT = len(id2lettre)
DIM_OUTPUT = len(id2lettre)
BATCH_SIZE = 64
HIDDEN_SIZE = 256
lr = 0.001
total_epoch = 33
data_trump = DataLoader(TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=1000), batch_size= batch_size, shuffle=True)

model = RNN(DIM_INPUT, HIDDEN_SIZE, DIM_OUTPUT).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# # Training Loop
# for epoch in tqdm(range(total_epoch)):
#     epoch_loss = 0
#     for x, y in tqdm(data_trump):
#         x = nn.functional.one_hot(x, num_classes=DIM_INPUT).to(device).float()
#         y = y.to(device)
#         optimizer.zero_grad()
#         output = model(x)
#         loss = criterion(output.transpose(1, 2), y)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#     print(f"Epoch {epoch+1}/{total_epoch}, Loss: {epoch_loss / len(data_trump)}")

#################@Training loop with checkpointing
savepath = PATH +"/modeltrump.pch"
state = State(model, optimizer, device, savepath)
# Training Loop
for epoch in tqdm(range(state.epoch, num_epochs)):
    epoch_loss = 0
    state.model.train()
    for x, y in tqdm(data_trump):
        x = nn.functional.one_hot(x, num_classes=DIM_INPUT).to(device).float()
        y = y.to(device)
        state.optim.zero_grad()
        output = model(x)
        loss = criterion(output.transpose(1, 2), y)
        loss.backward()
        state.optim.step()
        epoch_loss += loss.item()
        state.iteration += 1
    print(f"Epoch {epoch+1}/{total_epoch}, Loss: {epoch_loss / len(data_trump)}")
    # Save the state at the end of each epoch
    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save({
            'epoch': state.epoch,
            'iteration': state.iteration,
            'model_state_dict': state.model.state_dict(),
            'optimizer_state_dict': state.optim.state_dict()
        }, fp)

# Text Generation Function
def generate_text(seed, length=100):
    model.eval()
    generated = seed
    input_seq = string2code(seed).unsqueeze(0)
    input_seq = nn.functional.one_hot(input_seq, num_classes=DIM_INPUT).to(device).float()

    with torch.no_grad():
        for _ in range(length):
            output = model(input_seq)
            next_char_idx = output[0, -1].argmax()
            next_char = id2lettre[next_char_idx.item()]
            generated += next_char
            next_input = torch.tensor([[next_char_idx]], device=device)
            next_input = nn.functional.one_hot(next_input, num_classes=DIM_INPUT).float()
            input_seq = torch.cat([input_seq, next_input], dim=1)

    return generated

# Generate some text
print(generate_text("America ", 200))