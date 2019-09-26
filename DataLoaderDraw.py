import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import ndjson
import os
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import random
#import cairocffi as cairo
import ast

def parse_ink_array(inkarray):
    stroke_lengths = [len(stroke[0]) for stroke in inkarray]
    total_points = sum(stroke_lengths)
    np_ink = np.zeros((total_points, 3), dtype=np.float32)
    current_t = 0
    if not inkarray:
        print("Empty InkArray")
        return None
    for stroke in inkarray:
        if len(stroke[0]) != len(stroke[1]):
            print("Inconsistent number of x and y coordinates.")
            return None
        for i in [0, 1]:
            np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
        current_t += len(stroke[0])
        np_ink[current_t - 1, 2] = 1
    lower = np.min(np_ink[:, 0:2], axis=0)
    upper = np.max(np_ink[:, 0:2], axis=0)
    scale = upper - lower
    scale[scale == 0] = 1
    np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
    np_ink[1:, 0:2] -= np_ink[0:-1, 0:2]
    np_ink = np_ink[1:, :]
    return np_ink



def vector_to_raster(vector_images, side=28, line_diameter=16, padding=16, bg_color=(0,0,0), fg_color=(1,1,1)):
    original_side = 256
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)

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

class DrawDataset(Dataset):
    def __init__(self, path_array,label2idx):
        self.path_array = path_array
        self.label2idx = label2idx

    def __len__(self):
        return len(self.path_array)

    def __getitem__(self, idx):
        f = self.path_array[idx]
        label = f.split('/')[-2]
        y = self.label2idx[label]
        try:
            f = pickle.load(open(f, 'rb'))
        except:
            f = self.path_array[idx-4]
            label = f.split('/')[-2]
            y = self.label2idx[label]
            f = pickle.load(open(f, 'rb'))
        key = list(f.keys())[0] 
        sample_x = convert_to_np_raw(f[key])
        #print("File opened")
        #key = self.key_array[idx]
        #sample_x = f[self.key2array_idx[key]]['drawing']
        #sample_x = convert_to_np_raw(sample_x)
        #sample_y = self.label2idx[self.key2label[key]]
        #print(sample_x)
        #exit(0)
        sample_x = torch.from_numpy(sample_x)
        sample_x = sample_x.type(torch.FloatTensor)/255.0
        sample_x = torch.cat((sample_x.view(1, sample_x.size(0), sample_x.size(1)), sample_x.view(1, sample_x.size(0), sample_x.size(1)), sample_x.view(1, sample_x.size(0), sample_x.size(1))), dim =0)
        #print(sample_x.size())
        #print(sample_x.size())
        #print(sample_x)
        #sample_x = self.x[idx]
        #sample_y = self.y[idx]

        #sample_x = torch.tensor(sample_x)
        sample_y = torch.tensor(y)

        return sample_x, sample_y

class DrawLoaderDraw(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DrawLoaderDraw, self).__init__(*args, **kwargs)
