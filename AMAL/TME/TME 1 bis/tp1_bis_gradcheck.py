from torch.autograd import gradcheck
import torch
from tp1_bis import LinearFunction, MSEFunction

# Test du gradient de Linear (sur le même modèle que MSE)
n = 10  # Number of exemple (batch size)
d = 5  # Input dimension
p = 1  # Output dimension

# Tester la fonction linéaire
X = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
W = torch.randn(d, p, dtype=torch.float64, requires_grad=True)
b = torch.randn(p, dtype=torch.float64, requires_grad=True)

# Vérification du gradient avec gradcheck
linear = LinearFunction.apply
""" On utilise gradcheck ici pour vérifier que les gradients calculés pour la fonction linéaire sont corrects. 
    gradcheck compare les gradients analytiques (calculés par PyTorch) avec les gradients numériques (calculés par différences finies). 
    Si les deux gradients sont proches, cela signifie que la fonction linéaire est correctement différentiable. 
    Gradcheck renvoie true si c ok, et sinon lève une erreur en précisant pourquoi. """
test = gradcheck(linear, (X, W, b))
print(test)

# Tester la fonction de coût MSE
# Les cibles Y n'ont pas besoin de gradients car elles ne sont pas des paramètres d'apprentissage
Y = torch.randn(n, p, dtype=torch.float64, requires_grad=True)

mse = MSEFunction.apply
# Vérification du gradient avec gradcheck pour la fonction de coût MSE
# Cela permet de s'assurer que les gradients sont correctement calculés pour la fonction de coût
test_mse = gradcheck(mse, (linear(X, W, b), Y))
print(test_mse)
