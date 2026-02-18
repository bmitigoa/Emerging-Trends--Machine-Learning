import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Data
X = np.array([
    [5, 6], [3, 5], [8, 4], [1, 8], [3, 4], [7, 6],
    [9, 3], [9, 8], [3, 8], [8, 9], [9, 2], [5, 7]
])

y = np.array([
    "GOOD", "BAD", "GOOD", "BAD", "GOOD", "BAD",
    "GOOD", "BAD", "GOOD", "BAD", "GOOD", "BAD"
])

# Split half for training, half for testing (same as your n/2 logic)
n = len(X)
m = n // 2

X_train, y_train = X[:m], y[:m]
X_test, y_test = X[m:], y[m:]

# Train model
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Print results
print("X_test:\n", X_test)
print("y_pred:\n", y_pred)

# Accuracy (since you imported metrics in your code)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
