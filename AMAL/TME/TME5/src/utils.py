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
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))
class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self, input_dim, latent_dim, output_dim, batch_first = None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.f_x = nn.Linear(input_dim, latent_dim)
        self.f_h = nn.Linear(latent_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, output_dim)
        # Initialisation des poids
        self.init_weights()

    def init_weights(self):
        # Initialisation de Xavier pour les couches linéaires
        nn.init.xavier_uniform_(self.f_x.weight)
        nn.init.xavier_uniform_(self.f_h.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.f_h.bias)
        nn.init.zeros_(self.decoder.bias)

    def one_step(self, x, h):
        return torch.tanh(self.f_x(x) + self.f_h(h))

    def forward(self, x, h = 0):
        h = torch.zeros(x.size(0), self.latent_dim).to(x.device)
        h_seq = []

        for t in range(x.size(1)):  # iteration over the sequence dimension
            h = self.one_step(x[:, t, :], h)
            h_seq.append(h)

        h_seq = torch.stack(h_seq, dim=1)
        return h_seq

    def decode(self, h_seq):
        return self.decoder(h_seq)





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

def generate_text(model, embedding, start_string="Trump", generation_length=100, temperature=0.5):
    input_eval = string2code(start_string)  # Convert starting string to tensor
    input_eval = input_eval.unsqueeze(0)  # Add batch dimension

    generated_text = start_string

    model.eval()  # Evaluation mode

    with torch.no_grad():
        h = torch.zeros([1, model.latent_dim], device=input_eval.device)  # Initialize hidden state

        for i in range(generation_length):
            # Update hidden state
            h = model.one_step(embedding(input_eval[:, -1]).float(), h)

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
