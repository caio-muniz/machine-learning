from tensorflow.keras.datasets import fashion_mnist

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from utils.image_processing import preprocessImages
import numpy as np


def evaluateKNN(k, X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
     
    return accuracy

def evaluateKNNWithCV(k, X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(model, X_train, y_train)

    return np.mean(scores)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train_processed = preprocessImages(X_train)
X_test_processed = preprocessImages(X_test)

X_train_small = X_train_processed[:10000] 
y_train_small = y_train[:10000] 
X_test_small = X_test_processed[:2000] 
y_test_small = y_test[:2000]

for k in range(1, 11):
    accuracy = evaluateKNNWithCV(
        k,
        X_train_small,
        y_train_small
    )

    print(f"K = {k}: {accuracy:.4f}")

model = KNeighborsClassifier(n_neighbors=6)

model.fit(X_train_small, y_train_small)

predictions = model.predict(X_test_small)

accuracyFinal = accuracy_score(y_test_small, predictions)

print(f"Final accuracy: {accuracyFinal:.4f}")




