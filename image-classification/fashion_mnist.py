from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

def showSampleImages(X, y):
    i = 0
    while i < 5: 
        plt.imshow(X[i], cmap="gray")
        plt.title(y[i])
        plt.show()
        i+=1



#plt.imshow(X_train[4], cmap="gray")
#plt.show()
