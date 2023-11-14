import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchmetrics

# Assuming the RNN class and the SampleMetroDataset class are defined in the utils module
from utils import RNN, SampleMetroDataset, device

# Parameters
CLASSES = 2  # Number of cities or classes
LENGTH = 20   # Sequence length
DIM_INPUT = 2 # Input dimension (e.g., in/out)
HIDDEN_SIZE = 20  # Hidden layer size for RNN
BATCH_SIZE = 32   # Batch size for training
EPOCHS = 25       # Number of training epochs
LEARNING_RATE = 1e-4  # Learning rate for optimizer

# Load the data
PATH = "AMAL/TME/TME4/data/"
matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
accuracy_train = torchmetrics.classification.Accuracy(
    task="multiclass", num_classes=CLASSES
)
accuracy_test = torchmetrics.classification.Accuracy(
    task="multiclass", num_classes=CLASSES
)
# Create datasets and dataloaders
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

# Instantiate the model
model = RNN(DIM_INPUT, HIDDEN_SIZE, CLASSES).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model = RNN(input_dim=DIM_INPUT, latent_dim=HIDDEN_SIZE, output_dim=CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
epochs = 25
loss_train_per_epoch = []
loss_test_per_epoch = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    epoch_loss_test = 0
    for x, y in data_train:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        h_seq = model(x)
        y_pred = model.decode(h_seq[:, -1, :])  
        # Decode the last output : Dans ce contexte de réseau many-to-one, la supervision ne 
        # se fait qu’en bout de séquence (et non pas à chaque instant de la séquence), lorsque la
        # classe est décodée. L’erreur associée est rétro-propagée sur tous les états traversés
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        accuracy_train(y_pred.argmax(1), y) #on regarde la classe prédite la plus probable
    # Eval
    with torch.no_grad():
        for x, y in data_test:
            x = x.to(device)
            h = torch.zeros(
                (x.size(0), HIDDEN_SIZE), device=device
            )  # (batch_size, hidden_size)
            h = model(x)
            y_hat = model.decode(h[:, -1])

            loss = criterion(y_hat, y)
            accuracy_test(y_hat.argmax(1), y)
            epoch_loss_test += loss.sum()
    loss_train_per_epoch.append(total_loss / len(data_train))
    loss_test_per_epoch.append(epoch_loss_test / len(data_test))
    print("step:", epoch)
    print("Loss_train:", float(loss_train_per_epoch[-1]))
    print("Loss_test:", float(loss_test_per_epoch[-1]))
    print("acc_train:", accuracy_train.compute())
    accuracy_train.reset()
    print("acc_test:", accuracy_test.compute())
    accuracy_test.reset()
    # Add evaluation logic here if needed
    
# Save the model
torch.save(model.state_dict(), 'metro_rnn_model.pth')

# Model evaluation and predictions would follow after this, using the data_test DataLoader
# and the trained model.
