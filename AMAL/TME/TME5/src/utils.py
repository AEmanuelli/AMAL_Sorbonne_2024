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

## Token de padding (BLANK)
PAD_IX = 0
## Token de fin de séquence
EOS_IX = 1

LETTRES = string.ascii_letters + string.punctuation + string.digits + " "
id2lettre = dict(zip(range(2, len(LETTRES) + 2), LETTRES))
id2lettre[PAD_IX] = "<PAD>"  ##NULL CHARACTER
id2lettre[EOS_IX] = "<EOS>"
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))


def normalize(s):
    """enlève les accents et les caractères spéciaux"""
    return "".join(c for c in unicodedata.normalize("NFD", s) if c in LETTRES)


def string2code(s):
    """prend une séquence de lettres et renvoie la séquence d'entiers correspondantes"""
    return torch.tensor([lettre2id[c] for c in normalize(s)])


def code2string(t):
    """prend une séquence d'entiers et renvoie la séquence de lettres correspondantes"""
    if type(t) != list:
        t = t.tolist()
    return "".join(id2lettre[i] for i in t)

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
        t = torch.cat([t, torch.zeros(self.MAX_LEN - t.size(0), dtype=torch.long)])
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

# def generate_text(model, embedding, start_string="Trump", generation_length=100, temperature=0.5):
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


class GRUModel(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.gru = nn.GRU(input_dim, latent_dim) #oui j'ai la grosse flemme 
        self.decoder = nn.Linear(latent_dim, output_dim) 

    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        decoded = self.decoder(output)
        return decoded, hidden
    
    def init_hidden(self, batch_size):
        # Initializes hidden state
        device = next(self.gru.parameters()).device
        return torch.zeros(1, batch_size, self.latent_dim).to(device)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, num_layers=1):
        super().__init__()
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, latent_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(latent_dim, output_dim)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        # out = self.fc(out[:, -1, :])
        

        out = out.contiguous().view(-1, self.latent_dim)
        
        # Pass output through fully connected layer
        out = self.fc(out)
        
        # Reshape back to [batch_size, seq_len, output_size]
        out = out.view(x.size(0), x.size(1), -1)
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.latent_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.latent_dim).to(device))

