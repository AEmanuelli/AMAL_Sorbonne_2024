from icecream import ic
import numpy as np
from torch.utils.data import DataLoader
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm
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
# writer.add_image(f'samples', images, 0)


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


#Automatically get data dimensions 
first_batch, _ = next(iter(data_train))
input_size = first_batch.view(first_batch.size(0), -1).size(1)




# # Initialize the model, loss criterion, optimizer, and TensorBoard writer
# autoencodeur = Autoencodeur(input_size)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(autoencodeur.parameters(), lr=1e-3)
# writer = SummaryWriter("runs/autoencoder_experiment_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

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

# #Visualize some sample reconstructions after the last epoch
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


# #######Question 3
# Initialize the model, loss criterion, optimizer, and TensorBoard writer
model = Autoencodeur(input_size)
criterion = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)


savepath = Path("AMAL/TME/TME3/model.pch")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create the state instance
state = State(model, optim, device, savepath)

ITERATIONS = 25


for epoch in range(state.epoch, ITERATIONS):
    # Training
    state.model.train()
    train_loss = 0.0
    for x, _ in tqdm(data_train, desc=f"Epoch {epoch + 1}/{ITERATIONS}"):
        state.optim.zero_grad()
        xhat = state.model(x)
        loss = criterion(xhat, x)
        loss.backward()
        state.optim.step()
        state.iteration += 1
        train_loss += loss.item()

    train_loss /= len(data_train)

    # Evaluation
    state.model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for x, _ in data_test:  # Assuming data_test is defined
            xhat = state.model(x)
            loss = criterion(xhat, x)
            eval_loss += loss.item()

    eval_loss /= len(data_test)
    print(f"Epoch {epoch + 1}/{ITERATIONS}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

    # Save the state at the end of each epoch
    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save({
            'epoch': state.epoch,
            'iteration': state.iteration,
            'model_state_dict': state.model.state_dict(),
            'optimizer_state_dict': state.optim.state_dict()
        }, fp)


# ### EMBEDDINGS 
# # Define savepath, model, device, etc.
# # savepath = Path("AMAL/TME/TME3/model.pch")
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # model = Autoencodeur(input_size)  # Ensure input_size is defined
# # model.to(device)

# # # Load saved model state
# # checkpoint = torch.load(savepath, map_location=device)
# # model.load_state_dict(checkpoint['model_state_dict'])

# # Process 3 batches and get embeddings
# embeddings, labels = [], []
# model.eval()
# with torch.no_grad():
#     for i, (images, label) in enumerate(data_train):
#         if i == 3: break
#         images = images.view(images.size(0), -1).to(device)
#         encoded_images = model.encoder(images)
#         embeddings.append(encoded_images.cpu().numpy())
#         labels.append(label.numpy())

# # Calculate mean embeddings per label
# embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
# mean_embeddings = {label: embeddings[labels == label].mean(axis=0) for label in np.unique(labels)}

# # TensorBoard Visualization
# writer = SummaryWriter('runs/embeddings_visualization')
# features = torch.tensor(np.array(list(mean_embeddings.values())))
# class_labels = torch.tensor(list(mean_embeddings.keys()))

# writer.add_embedding(features, metadata=class_labels)
# writer.close()




# # Assumons que data_train est un DataLoader ou un itérable similaire
# x1, label_x1 = None, None
# x2, label_x2 = None, None

# for images, labels in data_train:
#     for i, label in enumerate(labels):
#         if x1 is None:
#             x1, label_x1 = images[i], label
#         elif label != label_x1:
#             x2, label_x2 = images[i], label
#             break
#     if x1 is not None and x2 is not None:
#         break

# # Vérification pour s'assurer que les deux images sont trouvées
# if x1 is None or x2 is None:
#     raise ValueError("Deux images de classes différentes n'ont pas pu être trouvées.")

# # Prétraitement des images si nécessaire (redimensionnement, normalisation, etc.)
# x1 = x1.unsqueeze(0).to(device)  # Ajout d'une dimension batch si nécessaire
# x2 = x2.unsqueeze(0).to(device)


# # Calcul des représentations latentes
# z1, z2 = model.encoder(x1), model.encoder(x2)

# # Interpolation entre z1 et z2
# lambdas = np.linspace(0, 1, num=10)  # 10 valeurs de λ entre 0 et 1
# interpolated_images = []

# for lam in lambdas:
#     z = lam * z1 + (1 - lam) * z2
#     interpolated_img = model.decoder(z).detach()
#     interpolated_images.append(interpolated_img)

# # Utilisation de TensorBoard pour afficher les images
# writer = SummaryWriter('runs/latent_space_interpolation')

# for i, img in enumerate(interpolated_images):
#     # img doit être au format (C, H, W) où C est le nombre de canaux (e.g., 3 pour RGB)
#     img = img.view(1,28,28)
#     writer.add_image(f'Interpolation_{i}', img, global_step=i)

# writer.close()


######## QUESTION 4
# Initialisation du modèle
num_layers = 3
model = HighwayNetwork(input_size, num_layers)

# Définir le critère de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()  # Pour la classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


savepath = Path("AMAL/TME/TME3/HIGHWAY.pch")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create the state instance

state = State(model, optimizer, device, savepath)

# Boucle d'entraînement
num_epochs = 5  # Nombre d'époques
for epoch in range(num_epochs):
    total_loss = 0
    total_loss_test = 0
    state.model.train()
    for images, labels in data_train:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Époque [{epoch+1}/{num_epochs}], Perte_train: {total_loss/len(data_train)}")
    
    # Save the state at the end of each epoch
    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save({
            'epoch': state.epoch,
            'iteration': state.iteration,
            'model_state_dict': state.model.state_dict(),
            'optimizer_state_dict': state.optim.state_dict()
        }, fp)

    state.model.eval()
    for images, labels in data_test:
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss_test += loss.item()
    print(f"Époque [{epoch+1}/{num_epochs}], Perte_test: {total_loss_test/len(data_test)}")
    


import matplotlib.pyplot as plt
# Assuming state.model is your model and first_batch contains your data
first_batch, _ = next(iter(data_train))
state.model.eval()

# Move the batch to the same device as the model
device = next(state.model.parameters()).device
first_batch = first_batch.to(device)

with torch.no_grad():
    outputs = state.model(first_batch)
    # Assuming a classification task, get the predicted labels
    _, predicted = torch.max(outputs.data, 1)


# Display some images along with predicted labels
plt.figure(figsize=(10, 10))
for i in range(min(len(first_batch), 9)):  # Display first 9 images
    plt.subplot(3, 3, i + 1)
    plt.imshow(first_batch[i].reshape(28, 28), cmap='gray')  # Adjust shape for your data
    plt.title(f'Predicted: {predicted[i].item()}')
    plt.axis('off')
plt.show()