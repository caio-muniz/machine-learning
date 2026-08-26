from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image = Image.open("imagem.jpg")

def getPixelInfo(image, x, y):
    result = image.getpixel((x,y))

    return result

def resizeImage(image, width, height):
    resizedImage = image.resize((width, height))

    return resizedImage

def normalizeImage(imageArray):
    normalizedArray = imageArray/255

    return normalizedArray



imageArray = np.array(image)

redChannel = imageArray[:, :, 0]
greenChannel = imageArray[:, :, 1]
blueChannel = imageArray[:, :, 2]

grayImage = image.convert("L")

print(grayImage.mode)
print(np.array(grayImage).shape)

resized = resizeImage(grayImage, 128, 128)
resizedArray = np.array(resized)

normalizedResizedArray = normalizeImage(resizedArray)

plt.imshow(grayImage, cmap="gray")
plt.show()
