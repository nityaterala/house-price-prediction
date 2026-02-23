# House Price Prediction using Linear Regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# -----------------------------
# STEP 1: Create Sample Dataset
# -----------------------------
data = {
    "area": [1000, 1500, 1800, 2400, 3000, 3500, 4000],
    "bedrooms": [2, 3, 3, 4, 4, 5, 5],
    "age": [10, 5, 8, 2, 1, 3, 0],
    "price": [200000, 300000, 320000, 450000, 500000, 550000, 600000]
}

df = pd.DataFrame(data)

print("Dataset:\n", df)

# -----------------------------
# STEP 2: Define Features & Target
# -----------------------------
X = df[["area", "bedrooms", "age"]]
y = df["price"]

# -----------------------------
# STEP 3: Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 4: Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# STEP 5: Predictions
# -----------------------------
predictions = model.predict(X_test)

print("\nPredictions:", predictions)
print("Actual:", y_test.values)

# -----------------------------
# STEP 6: Evaluate Model
# -----------------------------
mae = mean_absolute_error(y_test, predictions)
print("\nMean Absolute Error:", mae)

# -----------------------------
# STEP 7: Predict New House
# -----------------------------
new_house = np.array([[2500, 4, 5]])
predicted_price = model.predict(new_house)

print("\nPredicted price for new house:", predicted_price[0])
