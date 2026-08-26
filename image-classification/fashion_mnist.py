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

def flattenImages(X):
    arr = np.array(X)
    flat = arr.reshape(arr.shape[0], -1)
    
    return flat

def normalizeImage(imageArray):
    normalizedArray = imageArray/255

    return normalizedArray
    
def preprocessImages(X):
    flatX = flattenImages(X)
    normalizedX = normalizeImage(flatX)
    
    return normalizedX
    
X_train_processed = preprocessImages(X_train)
X_test_processed = preprocessImages(X_test)

print(X_train_processed.shape)
print(X_test_processed.shape)

print(np.min(X_train_processed))
print(np.max(X_train_processed))

