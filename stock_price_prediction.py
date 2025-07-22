import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Download stock data
df = yf.download("TSLA", start="2020-01-01", end="2023-01-01")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Features & label
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

X = df.drop('Target', axis=1)
y = df['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, predictions):.2f}")
