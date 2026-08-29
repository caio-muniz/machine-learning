import numpy as np


def getPixelInfo(image, x, y):
    return image.getpixel((x, y))


def resizeImage(image, width, height):
    return image.resize((width, height))


def normalizeImage(imageArray):
    return imageArray / 255


def flattenImages(X):
    array = np.array(X)
    return array.reshape(array.shape[0], -1)


def preprocessImages(X):
    flattened = flattenImages(X)
    return normalizeImage(flattened)
