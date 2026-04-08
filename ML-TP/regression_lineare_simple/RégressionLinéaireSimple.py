import numpy as np
import matplotlib.pyplot as plt
import Dataset


#Nous avons utilisé la variable BMI pour prédire la progression du diabète.
#Deux approches ont été utilisées :

#Implémentation from scratch avec gradient descent
#Utilisation de la bibliothèque sklearn

# Prendre une seule feature (BMI)
X_simple = Dataset.X[:, 2]
y_simple = Dataset.y

# Initialisation
m, b = 0, 0
lr = 0.01
epochs = 1000
n = len(X_simple)

# Gradient Descent
for _ in range(epochs):
    y_pred = m * X_simple + b
    dm = (-2/n) * np.sum(X_simple * (y_simple - y_pred))
    db = (-2/n) * np.sum(y_simple - y_pred)
    m -= lr * dm
    b -= lr * db

print("m =", m, "b =", b)

# Visualisation
plt.scatter(X_simple, y_simple)
plt.plot(X_simple, m*X_simple + b, color='red')
plt.title("Régression Linéaire Simple (From Scratch)")
plt.xlabel("BMI")
plt.ylabel("Progression maladie")
plt.show()