import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("iris_100.csv")

print(df)

# Features (all columns except last one)
X_train = df.iloc[:, 0:4]

# Target (last column)
y_train = df.iloc[:, 4]

# KNN with K = 5
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_train)

print("Predictions:\n", y_pred)
