import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchmetrics
from pathlib import Path
# Assuming the RNN class and the SampleMetroDataset class are defined in the utils module
from utils import *



# Parameters
CLASSES = 8  # Number of cities or classes
LENGTH = 30   # Sequence length
DIM_INPUT = 1 # Input dimension (e.g., in/out)
HIDDEN_SIZE = 256 # Hidden layer size for RNN
BATCH_SIZE = 64   # Batch size for training
LEARNING_RATE = 1e-4  # Learning rate for optimizer
LENGTH_TEST = 10
LENGTH_TEST_2 = 60

PATH = "AMAL/TME/TME4/data/"
epochs = 300


# Load the data
matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
accuracy_train = torchmetrics.classification.Accuracy(
    task="multiclass", num_classes=CLASSES
)
accuracy_test = torchmetrics.classification.Accuracy(
    task="multiclass", num_classes=CLASSES
)
# Create datasets and dataloaders
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH_TEST_2, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

# Instantiate the model
model = RNN(input_dim=DIM_INPUT, latent_dim=HIDDEN_SIZE, output_dim=CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)



# Model evaluation and predictions would follow after this, using the data_test DataLoader
# and the trained model.

savepath = Path("AMAL/TME/TME4/src/hzdataset_model_1.pch")
state = State(model, optimizer, device, savepath)

loss_train_per_epoch = []
loss_test_per_epoch = []
# Training Loop
for epoch in tqdm(range(state.epoch, epochs)):
    epoch_loss = 0
    epoch_loss_test = 0
    state.model.train()
    for x, y in tqdm(data_train):
        state.optim.zero_grad()
        x, y = x.to(device), y.to(device)
        h_seq = model(x)
        y_pred = model.decode(h_seq[:, -1, :])  
        loss = criterion(y_pred, y)
        loss.backward()
        state.optim.step()
        epoch_loss += loss.item()
        state.iteration += 1
        accuracy_train(y_pred.argmax(1), y) #on regarde la classe prédite la plus probable
    with torch.no_grad():
        for x, y in data_test:
            x = x.to(device)
            h = model(x)
            y_hat = model.decode(h[:, -1])
            loss = criterion(y_hat, y)
            accuracy_test(y_hat.argmax(1), y)
            epoch_loss_test += loss.sum()
    loss_train_per_epoch.append(epoch_loss / len(data_train))
    loss_test_per_epoch.append(epoch_loss_test / len(data_test))
    print("acc_train:", accuracy_train.compute())
    accuracy_train.reset()
    print("acc_test:", accuracy_test.compute())
    accuracy_test.reset()
    print(f"Epoch {epoch+1}/{epochs}", "Loss_train:", float(loss_train_per_epoch[-1]), "Loss_test:", float(loss_test_per_epoch[-1]))
     
    # Save the state at the end of each epoch
    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save({
            'epoch': state.epoch,
            'iteration': state.iteration,
            'model_state_dict': state.model.state_dict(),
            'optimizer_state_dict': state.optim.state_dict()
        }, fp)