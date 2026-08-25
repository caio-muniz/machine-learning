from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image = Image.open("imagem.jpg")

def getPixelInfo(image, x, y):
    result = image.getpixel((x,y))

    return result

imageArray = np.array(image)

redChannel = imageArray[:, :, 0]
greenChannel = imageArray[:, :, 1]
blueChannel = imageArray[:, :, 2]

print(redChannel.shape)
print(greenChannel.shape)
print(blueChannel.shape)

plt.imshow(redChannel, cmap="gray")
plt.show()
