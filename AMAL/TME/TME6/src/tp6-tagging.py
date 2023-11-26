# import itertools
# import logging
# from tqdm import tqdm

# from datamaestro import prepare_dataset
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# import torch
# from typing import List
# import time
# logging.basicConfig(level=logging.INFO)

# ds = prepare_dataset('org.universaldependencies.french.gsd')


# # Format de sortie décrit dans
# # https://pypi.org/project/conllu/

# class Vocabulary:
#     """Permet de gérer un vocabulaire.

#     En test, il est possible qu'un mot ne soit pas dans le
#     vocabulaire : dans ce cas le token "__OOV__" est utilisé.
#     Attention : il faut tenir compte de cela lors de l'apprentissage !

#     Utilisation:

#     - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
#       automatiquement s'il n'est pas connu
#     - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
#     """
#     OOVID = 1
#     PAD = 0

#     def __init__(self, oov: bool):
#         """ oov : autorise ou non les mots OOV """
#         self.oov =  oov
#         self.id2word = [ "PAD"]
#         self.word2id = { "PAD" : Vocabulary.PAD}
#         if oov:
#             self.word2id["__OOV__"] = Vocabulary.OOVID
#             self.id2word.append("__OOV__")

#     def __getitem__(self, word: str):
#         if self.oov:
#             return self.word2id.get(word, Vocabulary.OOVID)
#         return self.word2id[word]

#     def get(self, word: str, adding=True):
#         try:
#             return self.word2id[word]
#         except KeyError:
#             if adding:
#                 wordid = len(self.id2word)
#                 self.word2id[word] = wordid
#                 self.id2word.append(word)
#                 return wordid
#             if self.oov:
#                 return Vocabulary.OOVID
#             raise

#     def __len__(self):
#         return len(self.id2word)

#     def getword(self,idx: int):
#         if idx < len(self):
#             return self.id2word[idx]
#         return None

#     def getwords(self,idx: List[int]):
#         return [self.getword(i) for i in idx]



# class TaggingDataset():
#     def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
#         self.sentences = []

#         for s in data:
#             self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))
#     def __len__(self):
#         return len(self.sentences)
#     def __getitem__(self, ix):
#         return self.sentences[ix]


# def collate_fn(batch):
#     """Collate using pad_sequence"""
#     return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))


# logging.info("Loading datasets...")
# words = Vocabulary(True)
# tags = Vocabulary(False)
# train_data = TaggingDataset(ds.train, words, tags, True)
# dev_data = TaggingDataset(ds.validation, words, tags, True)
# test_data = TaggingDataset(ds.test, words, tags, False)


# logging.info("Vocabulary size: %d", len(words))


# BATCH_SIZE=100

# train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
# dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
# test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)




# #  TODO:  Implémenter le modèle et la boucle d'apprentissage (en utilisant les LSTMs de pytorch)




import itertools, logging, torch, numpy as np, time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from datamaestro import prepare_dataset
from typing import List
from icecream import ic
from torchmetrics import Accuracy as tmAccuracy

class Vocabulary:
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "__OOV__"] if oov else ["PAD"]
        self.word2id = {word: idx for idx, word in enumerate(self.id2word)}

    def __getitem__(self, word: str):
        return self.word2id.get(word, self.OOVID if self.oov else self.word2id[word])

    def get(self, word: str, adding=True):
        return self.word2id.setdefault(word, len(self.id2word) if adding else self.OOVID)

    def __len__(self):
        return len(self.id2word)

    def getword(self, idx: int):
        return self.id2word[idx] if idx < len(self) else None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]

class TaggingDataset:
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = [([words.get(t["form"], adding) for t in s],
                           [tags.get(t["upostag"], adding) for t in s]) for s in data]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, ix):
        return self.sentences[ix]

def collate_fn(batch):
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))

ds = prepare_dataset("org.universaldependencies.french.gsd")
words, tags = Vocabulary(True), Vocabulary(False)
train_data = TaggingDataset(ds.train, words, tags, True)
dev_data = TaggingDataset(ds.validation, words, tags, True)
test_data = TaggingDataset(ds.test, words, tags, False)

BATCH_SIZE, LEN_WORDS, LEN_TAG = 32, len(words), len(tags)
train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)

class Model(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, tag_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size)
        self.f_h = nn.Linear(hidden_size, tag_size)

    def forward(self, x):
        x, _ = self.rnn(self.embedding(x))
        return x

    def decode(self, h):
        return self.f_h(h)

def run_epoch(loader, model, loss_fn, optimizer=None, device="cuda", num_classes=18):
    model.to(device).train() if optimizer else model.eval()
    acc = tmAccuracy(num_classes=num_classes).to(device)
    losses = []
    for input, target in loader:
        input, target = input.to(device), target.to(device)
        output = model.decode(model(input)).transpose(1, 2)
        loss = loss_fn(output, target)
        losses.append(loss.item())
        acc(output.argmax(1), target)
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return np.mean(losses), acc.compute().item()

model = Model(32, 64, LEN_WORDS, LEN_TAG).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
loss_fn, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001)

for epoch in tqdm(range(10)):
    train_loss, train_acc = run_epoch(train_loader, model, loss_fn, optimizer)
    test_loss, test_acc = run_epoch(test_loader, model, loss_fn)
    ic(train_loss, train_acc, test_loss, test_acc)
