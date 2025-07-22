import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.DataFrame({
    'Hours': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2],
    'Scores': [21, 47, 27, 75, 30, 20, 88]
})

X = data[['Hours']]
y = data['Scores']

model = LinearRegression()
model.fit(X, y)

# Prediction
pred = model.predict([[7]])
print(f"Predicted Score for 7 hours: {pred[0]:.2f}")

# Plotting
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.title("Study Hours vs Scores")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()
