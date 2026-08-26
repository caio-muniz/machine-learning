from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from tensorflow.keras.datasets import fashion_mnist
from utils.image_processing import (
    getPixelInfo,
    resizeImage,
    normalizeImage,
    flattenImages,
    preprocessImages
)
import numpy as np

iris = load_iris()
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


X = iris.data
y = iris.target

#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=10, stratify=y
#)

def evaluateKNN(k, X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
     
    return accuracy

def evaluateKNNWithCV(k, X_train, y_train):
    model = KNeighborsClassifier(n_neighbors = k)

    scores = cross_val_score(model, X_train, y_train)

    mean = np.mean(scores)

    return mean

X_train_processed = preprocessImages(X_train)
X_test_processed = preprocessImages(X_test)

model = KNeighborsClassifier(n_neighbors = 7)
model.fit(X_train_processed[:10000], y_train[:10000])
predict = model.predict(X_test_processed[:10000])
accuracyFinal = accuracy_score(y_test, predict)
print(accuracyFinal)
