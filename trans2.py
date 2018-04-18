# -*- coding: utf-8 -*-
import csv
import os
from PIL import Image
import numpy as np


train_csv = 'train.csv'
val_csv = 'val.csv'
test_csv = 'test.csv'

train_set = 'train'
val_set = 'val'
test_set = 'test'

for save_path, csv_file in [(train_set, train_csv), (val_set, val_csv), (test_set, test_csv)]:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num = 1
    with open(csv_file) as f:
        csvr = csv.reader(f)
        header = next(csvr)
        for i, (label, pixel) in enumerate(csvr):
            pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
            subfolder = os.path.join(save_path, label)
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            im = Image.fromarray(pixel).convert('L')
            image_name = os.path.join(subfolder, '{:05d}.png'.format(i))
            print(image_name)
            im.save(image_name)