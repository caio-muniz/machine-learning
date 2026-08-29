from tensorflow.keras.datasets import fashion_mnist

from utils.image_processing import preprocessImages


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train_processed = preprocessImages(X_train)
X_test_processed = preprocessImages(X_test)

print(f"Training data: {X_train_processed.shape}")
print(f"Test data: {X_test_processed.shape}")
print(f"Minimum value: {X_train_processed.min()}")
print(f"Maximum value: {X_train_processed.max()}")
