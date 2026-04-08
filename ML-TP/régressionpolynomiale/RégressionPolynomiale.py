from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import Dataset

#Nous avons transformé les données pour capturer des relations non linéaires.
#Cela permet d'améliorer la précision dans certains cas.

import matplotlib.pyplot as plt

X_simple = Dataset.X[:, 2].reshape(-1, 1)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_simple)

model = LinearRegression()
model.fit(X_poly, Dataset.y)

print("Coefficients:", model.coef_)



plt.scatter(X_simple, Dataset.y)
plt.plot(X_simple, model.predict(X_poly), color='red')
plt.title("Régression Polynomiale")
plt.show()