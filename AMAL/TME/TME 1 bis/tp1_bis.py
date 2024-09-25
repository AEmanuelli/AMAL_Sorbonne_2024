"""je recommendce dès le début, pour cela, j'utilise l'environnement TME_bis, python 3.10, """

import subprocess

# Activer l'environnement conda
# subprocess.call("conda activate TME_bis")
subprocess.call("pip install master_dac", shell=True)

import torch
print(torch.__version__)
from torch.autograd import Function

# Implémentation de la fonction linéaire
class LinearFunction(Function):
    @staticmethod
    def forward(ctx, X, W, b):
        # Calcul de la prédiction linéaire
        Y_hat = X @ W + b
        ctx.save_for_backward(X, W, b)  # Sauvegarde des valeurs pour la passe backward
        return Y_hat

    @staticmethod
    def backward(ctx, grad_output):
        # Récupération des variables sauvegardées lors de la passe forward
        X, W, b = ctx.saved_tensors
        grad_X = grad_W = grad_b = None

        # Calcul du gradient par rapport à chaque entrée
        grad_X = grad_output @ W.t()  # Gradient par rapport à X
        grad_W = X.t() @ grad_output  # Gradient par rapport à W
        grad_b = grad_output.sum(0)   # Gradient par rapport à b

        return grad_X, grad_W, grad_b

# Fonction de coût MSE
class MSEFunction(Function):
    @staticmethod
    def forward(ctx, Y_hat, Y):
        # Calcul de la perte MSE
        loss = (Y_hat - Y) ** 2
        ctx.save_for_backward(Y_hat, Y)  # Sauvegarde des valeurs pour la passe backward
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        Y_hat, Y = ctx.saved_tensors
        grad_Y_hat =grad_output* 2 * (Y_hat - Y)
        grad_Y = -2 *grad_output* (Y_hat - Y)
        return grad_Y_hat, grad_Y  

