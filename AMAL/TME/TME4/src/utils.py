import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station

class ForecastMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data,length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length-1)],self.data[day,(timeslot+1):(timeslot+self.length)]

class State:
    def __init__(self, model, optim, device, savepath = ""):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0, 0
        self.savepath = savepath
        self.device = device
        # Check if we have a saved state and load it
        if savepath.is_file():
            with savepath.open("rb") as fp:
                state = torch.load(fp)  # Restart from saved model
                self.model.load_state_dict(state['model_state_dict'])
                self.optim.load_state_dict(state['optimizer_state_dict'])
                self.epoch = state['epoch']
                self.iteration = state['iteration']
        else:
            # Initialize model and optimizer here
            self.model = model.to(device)
            self.optim = optim
