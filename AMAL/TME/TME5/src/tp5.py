from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader
from utils import *
from generate import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar = PAD_IX):
    """
    Calculate cross-entropy loss, ignoring the padding characters.
    
    :param output: Tensor of shape [length, batch, output_dim] (model output)
    :param target: Tensor of shape [length, batch] (ground truth)
    :param padcar: Integer representing the padding character
    :return: Scalar tensor representing the mean loss
    """
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss = criterion(output, target)

    # Create a mask for non-padding elements
    mask = target != padcar
    masked_loss = loss * mask.type_as(loss)

    # Calculate mean loss only over non-padded elements
    return masked_loss.sum() / mask.sum()




#  TODO: 
PATH = "AMAL/TME/TME4/data/"

def train(model, data_loader, criterion, optimizer):
    checkpoint_path = PATH + f"{model_name}.pth.tar"
    start_epoch = load_checkpoint(checkpoint_path, model, optimizer)

    for epoch in tqdm(range(start_epoch, total_epoch)):
        model.train()
        total_loss = 0

        for x, y in tqdm(data_loader):
            x = x.to(device)
            x_embedded = embedding(x).to(device)

            optimizer.zero_grad()

            if model.__class__.__name__ == "LSTMModel":
                current_batch_size = x.size(0)
                hidden = model.init_hidden(current_batch_size)
                x = x.to(device)
                x_embedded = embedding(x).to(device)
                y_hat, hidden = model(x_embedded, hidden)
                # Detach hidden states from the graph after each batch
                hidden = tuple([state.detach() for state in hidden])
                # Calculate and accumulate loss
                loss = criterion(y_hat.transpose(1, 2), y.to(device))
                total_loss += loss.item()

            elif model.__class__.__name__ == "RNN":
                h = torch.zeros(x_embedded.size(0), hidden_size, device=device)
                h = model(x_embedded, h)
                y_hat = model.decode(h)
                # Calculate and accumulate loss
                loss = criterion(y_hat.transpose(1, 2), y.to(device))
                total_loss += loss.item()
            elif model.__class__.__name__ == "GRUModel":
                current_batch_size = x.size(1)# ça fonctionne mais j'ai pas compris pourquoi 
                hidden = model.init_hidden(current_batch_size)  # Initialize hidden state
                x = x.to(device)
                x_embedded = embedding(x).to(device)
                optimizer.zero_grad()
                y_hat, hidden = model(x_embedded, hidden)  # Forward pass through GRU model
                # Detach hidden states from the graph after each batch
                hidden = hidden.detach()
                # Calculate and accumulate loss
                loss = criterion(y_hat.transpose(1, 2), y.to(device))
                total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch + 1}/{total_epoch}, Loss: {avg_loss:.4f}')
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Clipping des gradients
        print(code2string(y[0,:]))
        print(code2string(y_hat[0,:,:].argmax(1)))
        print(generate(model, embedding=embedding))
        if epoch % 3 == 0 or epoch == total_epoch - 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, filename=PATH + f"{model_name}.pth.tar")


# Train and generate text

DIM_INPUT = len(id2lettre)
DIM_OUTPUT = len(id2lettre)
EMBEDDING_DIM = 50
BATCH_SIZE = 128
hidden_size = 200
lr = 5e-4
total_epoch = 8
max_len = 60
data_trump = DataLoader(TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(), maxlen=max_len), 
                        batch_size=BATCH_SIZE, 
                        shuffle=True
)
embedding = nn.Embedding(num_embeddings=DIM_INPUT, embedding_dim=EMBEDDING_DIM).to(device)



model_name = "rnn"
model = RNN(EMBEDDING_DIM, hidden_size, DIM_OUTPUT).to(device)
optimizer = torch.optim.Adam(list(model.parameters()) + list(embedding.parameters()))
train(model, data_trump, maskedCrossEntropy, optimizer)

model_name = "lstm_"
lstm = LSTMModel(EMBEDDING_DIM, hidden_size, DIM_OUTPUT).to(device)
optimizer_lstm = torch.optim.Adam(list(lstm.parameters()) + list(embedding.parameters()))
train(lstm, data_trump, maskedCrossEntropy, optimizer_lstm)

model_name = "gru1"
gru = GRUModel(EMBEDDING_DIM, hidden_size, DIM_OUTPUT).to(device)
optimizer_gru = torch.optim.Adam(list(gru.parameters()) + list(embedding.parameters()))
train(gru, data_trump, maskedCrossEntropy, optimizer_gru)
