from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from utils.image_processing import (
    normalizeImage,
    resizeImage
)


image = Image.open("imagem.jpg")

imageArray = np.array(image)

redChannel = imageArray[:, :, 0]
greenChannel = imageArray[:, :, 1]
blueChannel = imageArray[:, :, 2]

grayImage = image.convert("L")

resizedImage = resizeImage(grayImage, 128, 128)
resizedArray = np.array(resizedImage)

normalizedImage = normalizeImage(resizedArray)

print(f"Original image: {imageArray.shape}")
print(f"Grayscale image: {np.array(grayImage).shape}")
print(f"Resized image: {resizedArray.shape}")
print(f"Normalized range: {normalizedImage.min()} - {normalizedImage.max()}")

plt.imshow(grayImage, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()
