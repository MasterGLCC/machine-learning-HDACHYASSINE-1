from sklearn.datasets import load_diabetes
import numpy as np

# Nous avons utilisé le dataset Diabetes fourni par sklearn.
# Ce dataset contient 442 observations et 10 variables explicatives liées à des caractéristiques médicales.
#On va utiliser un dataset réel intégré dans sklearn :

#Diabetes Dataset

#Utilisé en machine learning réel
#Objectif : prédire la progression du diabète
#10 variables explicatives (âge, BMI, tension, etc.)

data = load_diabetes()
X = data.data
y = data.target

print("Shape X:", X.shape)
print("Shape y:", y.shape)