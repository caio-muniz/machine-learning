from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np


def evaluateKNN(k, X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return accuracy_score(y_test, predictions)


def evaluateKNNWithCV(k, X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(model, X_train, y_train)

    return np.mean(scores)


iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=10,
    stratify=y
)


for k in range(1, 11):
    accuracy = evaluateKNN(k, X_train, y_train, X_test, y_test)
    print(f"K = {k}: {accuracy:.2f}")


print("\nCross Validation:")

for k in range(1, 11):
    accuracy = evaluateKNNWithCV(k, X_train, y_train)
    print(f"K = {k}: {accuracy:.2f}")
