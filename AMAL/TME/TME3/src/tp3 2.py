from icecream import ic
from torch.utils.data import DataLoader
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import datetime

from utils import *

# Téléchargement des données
from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist");
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f'samples', images, 0)


savepath = Path("model.pch")

#  TODO: 

####QUESTION 1

BATCH_SIZE = 20

# Create an instance of MonDataset with the dummy data
train_dataset = MonDataset(train_images, train_labels)

# Create a DataLoader
data_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset= MonDataset(test_images, test_labels)
data_test = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# num_images_to_show = 8
# images, labels = next(iter(DataLoader(dataset, batch_size=num_images_to_show, shuffle=True)))

# # Les images sont actuellement en format (batch_size, H * W), nous devons les redimensionner en (batch_size, 1, H, W)
# images = images.view(-1, 1, 28, 28)

# # Duplique le canal pour RGB en utilisant `repeat`
# images = images.repeat(1, 3, 1, 1)

# # Crée une grille d'images pour TensorBoard
# grid = make_grid(images)

# # Envoie les images dans TensorBoard
# writer.add_image('MNIST Examples', grid, 0)

# # Ferme le SummaryWriter pour libérer des ressources
# writer.close()

####################QUESTION 2


# Prepare the dataset and dataloader
dataset = MonDataset(train_images, train_labels)
data_train = DataLoader(dataset, batch_size=64, shuffle=True)  # Assuming you have a batch size of 64 for training
data_test = DataLoader(dataset, batch_size=64, shuffle=False)  # For evaluation

# Initialize the model, loss criterion, optimizer, and TensorBoard writer
autoencodeur = Autoencodeur()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencodeur.parameters(), lr=1e-3)
writer = SummaryWriter("runs/autoencoder_experiment_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# # Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     autoencodeur.train()  # Set the model to training mode
#     for data in data_train:
#         img, _ = data  # Ignore labels
#         img = img.view(img.size(0), -1)  # Flatten the image
#         output = autoencodeur(img)  # Forward pass
#         loss = criterion(output, img)  # Calculate the loss
#         optimizer.zero_grad()  # Clear existing gradients
#         loss.backward()  # Backpropagate the loss
#         optimizer.step()  # Update weights
#         writer.add_scalar('Loss/Train', loss.item(), epoch)
    
#     # Evaluation loop
#     autoencodeur.eval()  # Set the model to evaluation mode
#     epoch_loss_test = 0.0
#     with torch.no_grad():
#         for data in data_test:
#             img, _ = data  # Ignore labels
#             img = img.view(img.size(0), -1)  # Flatten the image
#             output = autoencodeur(img)  # Forward pass
#             loss = criterion(output, img)  # Calculate the loss
#             epoch_loss_test += loss.item() * img.size(0)  # Accumulate the batch loss
#         writer.add_scalar('Loss/Test', epoch_loss_test / len(data_test.dataset), epoch)

#     print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}')

# Visualize some sample reconstructions after the last epoch
# autoencodeur.eval()  # Set the model to evaluation mode
# with torch.no_grad():
#     for x, _ in data_test:
#         # Ensure that the number of elements to reshape matches the actual number of elements
#         batch_size = x.size(0)  # This should be your actual batch size, not a fixed number like 8
#         images = x.view(batch_size, 1, 28, 28).repeat(1, 3, 1, 1)  # Adjust the reshape to match the batch size
#         images_grid = make_grid(images)
#         writer.add_image(f"samples/original", images_grid, epoch)
        
#         x = x.view(batch_size, -1)  # Flatten the image if it's not already flattened
#         outputs = autoencodeur(x)
#         break  # We just need one batch for visualization
        
#     outputs = outputs.view(batch_size, 1, 28, 28).repeat(1, 3, 1, 1)  # Reshape the output to have 3 channels
#     images_grid = make_grid(outputs)
#     writer.add_image(f"Reconstruction_last_epoch", images_grid, epoch)

# writer.close()  # Close the writer when you're done


#######Question 3

from pathlib import Path
import torch

savepath = Path("model.pch")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare the dataset and dataloader
dataset = MonDataset(train_images, train_labels)
data_train = DataLoader(dataset, batch_size=64, shuffle=True)  # Assuming you have a batch size of 64 for training
data_test = DataLoader(dataset, batch_size=64, shuffle=False)  # For evaluation

# Initialize the model, loss criterion, optimizer, and TensorBoard writer
model = Autoencodeur()
criterion = nn.MSELoss()
optim = torch.optim.Adam(autoencodeur.parameters(), lr=1e-3)

# Create the state instance
state = State(model, optim, device, savepath)

ITERATIONS = 25

for epoch in range(state.epoch, ITERATIONS):
    for x, y in data_train:
        state.optim.zero_grad()
        x = x.to(device)
        y = y.to(device)
        ic(x.size(), y.size())
        xhat = state.model(x)
        loss = criterion(xhat, y)
        loss.backward()
        state.optim.step()
        state.iteration += 1
    
    # Save the state at the end of each epoch
    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save({
            'epoch': state.epoch,
            'iteration': state.iteration,
            'model_state_dict': state.model.state_dict(),
            'optimizer_state_dict': state.optim.state_dict()
        }, fp)