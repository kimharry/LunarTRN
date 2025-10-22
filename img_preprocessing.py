# 1600x1600 -> 800x800

from PIL import Image
import os

path = './colmap-251023/images/'
images = os.listdir(path)

for image in images:
    img = Image.open(path + image)
    img = img.resize((800, 800))
    img.save(path + image)
    print(f"Processed {image}")
