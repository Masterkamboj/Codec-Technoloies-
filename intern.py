import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Download stock data
stock = yf.download(
    "AAPL",
    start="2020-01-01",
    end="2024-01-01",
    auto_adjust=True
)

# Fix MultiIndex columns if present
if isinstance(stock.columns, pd.MultiIndex):
    stock.columns = stock.columns.get_level_values(0)

# Create target column (next day's closing price)
stock["Prediction"] = stock["Close"].shift(-1)

# Remove last row (contains NaN in Prediction)
data = stock.dropna()

# Features and Labels
X = data[["Close"]].values
y = data["Prediction"].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Test predictions
predictions = model.predict(X_test)

# Accuracy score
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

# Predict next day's stock price
last_close = stock["Close"].iloc[-1]

next_day_price = model.predict([[last_close]])

print(f"Last Closing Price: ${last_close:.2f}")
print(f"Predicted Next Day Price: ${next_day_price[0]:.2f}")

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test[:100], label="Actual Price")
plt.plot(predictions[:100], label="Predicted Price")
plt.title("Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.show()