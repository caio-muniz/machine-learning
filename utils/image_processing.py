import numpy as np

def getPixelInfo(image, x, y):
    result = image.getpixel((x,y))

    return result

def resizeImage(image, width, height):
    resizedImage = image.resize((width, height))

    return resizedImage


def normalizeImage(imageArray):
    normalizedArray = imageArray/255

    return normalizedArray


def flattenImages(X):
    arr = np.array(X)
    flat = arr.reshape(arr.shape[0], -1)
    
    return flat

def preprocessImages(X):
    flatX = flattenImages(X)
    normalizedX = normalizeImage(flatX)
    
    return normalizedX
