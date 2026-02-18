import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix

# Dataset
X = np.array([
    [5, 6], [3, 5], [8, 4], [1, 8],
    [3, 4], [7, 6], [9, 3], [9, 8],
    [3, 8], [8, 9], [9, 2], [5, 7]
])

y = np.array([
    "GOOD", "BAD", "GOOD", "BAD",
    "GOOD", "BAD", "GOOD", "BAD",
    "GOOD", "BAD", "GOOD", "BAD"
])

# Split dataset (half train, half test)
n = len(X)
m = n // 2

X_train, y_train = X[:m], y[:m]
X_test, y_test = X[m:], y[m:]

# Create Decision Tree model
model = DecisionTreeClassifier(random_state=0)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display tree
tree_rules = export_text(model)

# Output
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("\nDecision Tree Rules:\n")
print(tree_rules)
