import numpy as np
import pandas as pd
import sklearn.tree as dtree
from sklearn.tree import export_text

df = pd.read_csv("Customer1.csv")

print(df)

X_train = df.iloc[:, 1:5]
y_train = df["Risk"]

model = dtree.DecisionTreeClassifier()
model.fit(X_train, y_train)

dt = export_text(model, feature_names=list(df.columns[1:5]))

print("Tree:\n", dt)
