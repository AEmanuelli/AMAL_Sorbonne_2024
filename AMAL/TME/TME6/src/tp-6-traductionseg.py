#%%
import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List
import random
import time
import re
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# %%
import sentencepiece as spm
import os

FILE = "AMAL/TME/TME6/data/en-fra.txt"  # Replace with your text file path

# Path to your SentencePiece model
model_path = 'AMAL/TME/TME6/src/sentencepieces.model'

# Initialize and load the model
spp = spm.SentencePieceProcessor()
spp.load(model_path)

# Path to your SentencePiece model
model_path = 'AMAL/TME/TME6/src/sentencepieces.model'
# Check if the SentencePiece model exists and train if it doesn't
if not os.path.exists(model_path):
    print("Training SentencePiece model...")
    spm.SentencePieceTrainer.train(
        input=FILE,
        model_prefix="sentencepieces",
        vocab_size=1000,  # Adjust vocab size as needed
        user_defined_symbols=[],  # Add any user-defined symbols if required
    )

# Load the trained SentencePiece model
spp = spm.SentencePieceProcessor(model_file=model_path)
print("SentencePiece model loaded.")


# Example of encoding a string
encoded_string = spp.encode("hey", out_type=str)
print(encoded_string)

# %%
def load_model_if_exists(model_path, device, model_type):
    if model_path.is_file():
        model = torch.load(model_path, map_location=device)
        print(f"Loaded model from {model_path}")
        return model
    else:
        if model_type == 'encoder':
            print("No encoder model found, initializing a new one.")
            return Encoder(SRC_VOCAB_SIZE, EMB_DIM, HID_DIM)
        elif model_type == 'decoder':
            print("No decoder model found, initializing a new one.")
            return Decoder(TRG_VOCAB_SIZE, EMB_DIM, HID_DIM)
        else:
            raise ValueError("Invalid model type specified")

# Specify the device
device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 50
BATCH_SIZE = 16
teacher_forcing_ratio = 1.0
gamma = 0.95  # Facteur de réduction
logging.basicConfig(level=logging.INFO)

FILE = "AMAL/TME/TME6/data/en-fra.txt"

writer = SummaryWriter("/tmp/runs/tag-"+time.asctime())

def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]



class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            orig_encoded = spp.encode(orig, out_type=int)
            dest_encoded = spp.encode(dest, out_type=int)
            self.sentences.append((torch.tensor(orig_encoded + [Vocabulary.EOS]), torch.tensor(dest_encoded + [Vocabulary.EOS])))

    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]



def collate_fn(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len

with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

train_loader = DataLoader(datatrain, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
test_loader = DataLoader(datatest, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)

#  TODO:  Implémenter l'encodeur, le décodeur et la boucle d'apprentissage
#%%
class Encoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.gru(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size):
        super().__init__()
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # Batched input processing
        embedded = self.embedding(input)  # Shape: [batch_size, emb_size]
        embedded = embedded.unsqueeze(0)  # Shape: [1, batch_size, emb_size]
        output, hidden = self.gru(embedded, hidden)
        prediction = self.out(output.squeeze(0))  # Shape: [batch_size, output_size]
        return prediction, hidden

    def generate(self, hidden, lenseq=50, temperature=1.0):  # Add temperature parameter
        outputs = []
        input = torch.tensor(Vocabulary.SOS).unsqueeze(0).to(hidden.device)

        for _ in range(lenseq):
            output, hidden = self.forward(input, hidden)
            
            # Use temperature sampling to select the next token
            top1 = temperature_sampling(output, temperature=temperature)
            print("Generated token ID:", top1.item())  # Debug print
            outputs.append(top1.item())
            input = top1

            if top1.item() == Vocabulary.EOS:
                break

        return outputs

    
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
def get_sentence(vocabulary, indices):
    sentence = []
    for idx in indices:
        word = vocabulary.getword(idx)
        if word is None:
            print(f"Index not found in vocabulary: {idx}")
            sentence.append("<UNK>")
        else:
            sentence.append(word)
    return ' '.join(sentence)




def run_epoch(loader, encoder, decoder, loss_fn, optimizer=None, device=device):
    encoder.to(device)
    decoder.to(device)
    
    if optimizer:
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()

    total_loss = 0
    i = 0
    for x, _, y, _ in tqdm(loader):
        i+=1
        x, y = x.to(device), y.to(device)
        encoder_hidden = encoder(x)

        input = torch.tensor([Vocabulary.SOS] * x.size(1)).to(device)
        outputs = torch.zeros(y.size(0), x.size(1), decoder.output_size).to(device)

        for t in range(y.size(0)):
            output, encoder_hidden = decoder(input, encoder_hidden)
            outputs[t] = output
            use_teacher_forcing = random.random() < teacher_forcing_ratio  # 50% chance of using teacher forcing
            input = y[t] if use_teacher_forcing else output.argmax(1)
            
        loss = loss_fn(outputs.view(-1, decoder.output_size), y.view(-1))
        total_loss += loss.item()
        if i%5==0: 
            # Print a sample prediction
            sample_output = outputs.argmax(2)[:, 0]
            predicted_sentence = ' '.join([vocFra.getword(idx) for idx in sample_output])
            # In your run_epoch function
            target_sentence = get_sentence(vocEng, x[:, 0])
            print(f"\ trad prédite: {predicted_sentence}")
            print(f"phrase: {target_sentence}")
        if optimizer:
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            loss.backward()
            optimizer[0].step()
            optimizer[1].step()

    return total_loss / len(loader)

        



# def train(encoder, decoder, data_loader, encoder_optimizer, decoder_optimizer, criterion, max_length):
    encoder.train()
    decoder.train()
    for src, trg in data_loader:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        hidden = encoder(src)
        input = trg[0]  # SOS token

        loss = 0
        for t in range(1, trg.size(0)):
            output, hidden = decoder(input, hidden)
            loss += criterion(output, trg[t])
            teacher_force = random.random() < 0.5
            input = trg[t] if teacher_force else output.argmax(1)

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        # Ajoutez des mesures pour suivre la perte, etc.

SRC_VOCAB_SIZE = spp.get_piece_size()
TRG_VOCAB_SIZE = spp.get_piece_size() # Taille du vocabulaire cible
EMB_DIM = 128
HID_DIM = 512


lr = 0.0025
lr_encoder = lr
lr_decoder = lr
nb_epoch = 5



# Define the paths for the encoder and decoder models
encoder_path = Path("encoder_{HID_DIM}_{EMB_DIM}.pt")
decoder_path = Path("decoder_{HID_DIM}_{EMB_DIM}.pt")

encoder = load_model_if_exists(encoder_path, device, 'encoder')
decoder = load_model_if_exists(decoder_path, device, 'decoder')

criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD)
encoder_optimizer = optim.Adam(encoder.parameters())
decoder_optimizer = optim.Adam(decoder.parameters())

for epoch in tqdm(range(nb_epoch)):
    mean_train_loss = run_epoch(train_loader, encoder, decoder, criterion, optimizer=(encoder_optimizer, decoder_optimizer), device=device)
    mean_test_loss = run_epoch(test_loader, encoder, decoder, criterion, device=device)
    teacher_forcing_ratio *= gamma 
    torch.save(encoder, f"encoder_{HID_DIM}_{EMB_DIM}.pt")
    torch.save(decoder, f"decoder_{HID_DIM}_{EMB_DIM}.pt")
    print(f"Epoch {epoch}: Train Loss: {mean_train_loss}, Test Loss: {mean_test_loss}")

