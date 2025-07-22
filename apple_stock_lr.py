import yfinance as yf
from sklearn.linear_model import LinearRegression
import pandas as pd

data = yf.download("AAPL", start="2022-01-01", end="2023-01-01")
data = data.reset_index()

data['DateOrdinal'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)
X = data[['DateOrdinal']]
y = data['Close']

model = LinearRegression()
model.fit(X, y)

future = pd.to_datetime("2023-07-01").toordinal()
prediction = model.predict([[future]])

print(f"Predicted Price for July 2023: ${prediction[0]:.2f}")
