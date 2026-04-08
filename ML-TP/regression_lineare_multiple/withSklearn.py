from sklearn.linear_model import LinearRegression

import Dataset

model = LinearRegression()
model.fit(Dataset.X, Dataset.y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)