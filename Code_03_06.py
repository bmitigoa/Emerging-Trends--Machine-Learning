import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("iris_100.csv")

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# K = 1
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train, y_train)
pred1 = knn1.predict(X_test)
acc1 = accuracy_score(y_test, pred1)

# K = 5
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, y_train)
pred5 = knn5.predict(X_test)
acc5 = accuracy_score(y_test, pred5)

print("Accuracy K=1:", acc1)
print("Accuracy K=5:", acc5)
