import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv("gld_price_data.csv")
X = df[['SPX', 'USO', 'SLV', 'EUR/USD']]
y = df['GLD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GradientBoostingRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, pred):.2f}")
