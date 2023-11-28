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

import time
import re
from torch.utils.tensorboard import SummaryWriter




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
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]



def collate_fn(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN=100
BATCH_SIZE=100

datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

train_loader = DataLoader(datatrain, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)

#  TODO:  Implémenter l'encodeur, le décodeur et la boucle d'apprentissage

class Encoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.gru(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        prediction = self.softmax(self.out(output[0]))
        return prediction, hidden

    def generate(self, hidden, lenseq=None):
        # Implémentez la génération ici
        pass


def run_epoch(loader, encoder, decoder, loss_fn, optimizer=None, device="cuda"):
    encoder.to(device)
    decoder.to(device)
    if optimizer:
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()

    total_loss = 0
    for x, len_x, y, len_y in loader:
        x, y = x.to(device), y.to(device)

        # Encoder part
        encoder_hidden = encoder(x)
        decoder_outputs, _ = decoder(encoder_hidden, y.size(0), target_tensor=y if optimizer else None)
        loss = loss_fn(decoder_outputs.transpose(1, 2), y)
        total_loss += loss.item()

        # backward if we are training
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

SRC_VOCAB_SIZE = len(vocEng)# Taille du vocabulaire source
TRG_VOCAB_SIZE = len(vocFra) # Taille du vocabulaire cible
EMB_DIM = 64  
HID_DIM = 128  
MAX_LEN = 100
lr = 0.0025
lr_encoder = lr
lr_decoder = lr
nb_epoch = 50



encoder = Encoder(SRC_VOCAB_SIZE, EMB_DIM, HID_DIM)
decoder = Decoder(TRG_VOCAB_SIZE, EMB_DIM, HID_DIM)
criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD)
encoder_optimizer = optim.Adam(encoder.parameters())
decoder_optimizer = optim.Adam(decoder.parameters())

for epoch in tqdm(range(nb_epoch)):
    mean_train_loss = run_epoch(train_loader, encoder, decoder, criterion, optimizer=(encoder_optimizer, decoder_optimizer), device=device)
    mean_test_loss = run_epoch(test_loader, encoder, decoder, criterion, device=device)

    torch.save(encoder, f"encoder_{HID_DIM}_{EMB_DIM}.pt")
    torch.save(decoder, f"decoder_{HID_DIM}_{EMB_DIM}.pt")
    print(f"Epoch {epoch}: Train Loss: {mean_train_loss}, Test Loss: {mean_test_loss}")

