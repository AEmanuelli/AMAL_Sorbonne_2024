import logging

from torch.nn.modules.pooling import MaxPool1d
logging.basicConfig(level=logging.INFO)

import heapq
from pathlib import Path
import gzip

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm

from tp8_preprocess import TextDataset

# Utiliser tp8_preprocess pour générer le vocabulaire BPE et
# le jeu de donnée dans un format compact

# --- Configuration

# Taille du vocabulaire
vocab_size = 1000
MAINDIR = Path(__file__).parent

# Chargement du tokenizer

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(f"wp{vocab_size}.model")
ntokens = len(tokenizer)
PATH = "AMAL/TME/TME8/src/"
def loaddata(mode):
    with gzip.open(PATH+f"{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp)


test = loaddata("test")
train = loaddata("train")
TRAIN_BATCHSIZE=500
TEST_BATCHSIZE=500


# --- Chargements des jeux de données train, validation et test

val_size = 1000
train_size = len(train) - val_size
train, val = torch.utils.data.random_split(train, [train_size, val_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
train_iter = torch.utils.data.DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate)
val_iter = torch.utils.data.DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)
test_iter = torch.utils.data.DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)


#  TODO: 

# class Scope pour calculer wi et si par récurrence
# init w,s
# call_ formule de rec 
# _rep affichage

# clazss Model
# convolution = sequential()
# le fully connected pas dans le sequential, sinon pas de stride et de kernel pour l'appel au scope
# scope parcourir la convolution *
# if conv1D[0]
# self.scope((s,k))
    # pooling m.stride
# forward
# conv
# fully-connected 


# class Sample 
# 1 ex forward
# scope 
# retrun idice début indice fin 




# commencer par définir l'embedding

class Model(nn.Module):
    def __init__(self, conv_channels=[20, 20], fc_size=[20, 2], kernel_size=3, stride=1, pooling_kernel=2, pooling_stride = 1):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pooling_kernel, stride = pooling_stride)
        )
        self.fc = nn.Linear(fc_size[0], fc_size[1])

    def forward(self, x):
        return self.fc(self.convolution(x))


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

model = Model()  
input_tensor = torch.randn(12,12,20) #3D batch, length, embedding
output = model(input_tensor)