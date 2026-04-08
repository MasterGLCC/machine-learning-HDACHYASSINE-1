from sklearn.linear_model import LinearRegression
import Dataset

X_simple = Dataset.X[:, 2].reshape(-1, 1)

model = LinearRegression()
model.fit(X_simple, Dataset.y)

print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)