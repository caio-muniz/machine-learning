from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("imagem.jpg")

plt.imshow(image)
plt.show()

print(image.size)
print(image.mode)