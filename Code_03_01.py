# Decision Tree Example using sklearn

import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Training data (features)
X_train = np.array([
    [5, 6],
    [3, 5],
    [8, 4],
    [1, 8],
    [3, 4],
    [7, 6],
    [9, 3],
    [9, 8],
    [3, 8],
    [8, 9],
    [9, 2],
    [5, 7]
])

# Labels
y_train = np.array([
    "GOOD", "BAD", "GOOD", "BAD",
    "GOOD", "BAD", "GOOD", "BAD",
    "GOOD", "BAD", "GOOD", "BAD"
])

# Test data (must be 2D array)
X_test = np.array([[4, 6]])

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Output result
print("Test Input:", X_test)
print("Predicted Output:", y_pred)
