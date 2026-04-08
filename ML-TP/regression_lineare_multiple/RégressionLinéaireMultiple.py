import numpy as np
import matplotlib.pyplot as plt
import Dataset 


#Nous avons utilisé les 10 variables du dataset.
#Cette méthode donne de meilleurs résultats car elle prend en compte plusieurs facteurs.


# Ajouter biais
X_b = np.c_[np.ones((Dataset.X.shape[0], 1)), Dataset.X]

theta = np.zeros(X_b.shape[1])
lr = 0.01
epochs = 1000
n = len(Dataset.y)

for _ in range(epochs):
    y_pred = X_b.dot(theta)
    gradient = (-2/n) * X_b.T.dot(Dataset.y - y_pred)
    theta -= lr * gradient

print("Theta:", theta)