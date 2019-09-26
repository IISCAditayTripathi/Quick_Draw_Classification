import pickle
import scipy.misc
import numpy as np
from PIL import Image, ImageDraw, ImageOps

file = pickle.load(open('test_image.pkl','rb'))

key = list(file.keys())[0]
image = file[key]
def convert_to_np_raw(drawing, width=256, height=256):
    img = np.zeros((width, height))
    pil_img = convert_to_PIL(drawing)
    pil_img.thumbnail((width, height), Image.ANTIALIAS)
    pil_img = pil_img.convert('RGB')
    pixels = pil_img.load()

    for i in range(0, width):
        for j in range(0, height):
            img[i,j] = 1- pixels[j,i][0]/255
    return img

def convert_to_PIL(drawing, width=256, height=256):
    pil_img = Image.new('RGB', (width, height), 'white')
    pixels = pil_img.load()
    draw = ImageDraw.Draw(pil_img)
    for x,y in drawing:
        for i in range(1, len(x)):
            draw.line((x[i-1], y[i-1], x[i], y[i]), fill=0)
    return pil_img

image = convert_to_np_raw(image)
scipy.misc.toimage(image).save('outfile.jpg')
