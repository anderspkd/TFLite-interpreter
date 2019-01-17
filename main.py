#!/usr/bin/env python

from model import TFLiteModel
import interpreter
import numpy as np
from sys import argv
from PIL import Image

if len(argv) < 3:
    print 'Usage: %s [tflite file] [image]' % (argv[0],)
    exit(0)

# Create a model from a file and load an image from a file. Then we run the
# interpreter.

model = TFLiteModel(argv[1], parse_data=False)
input_shape = model.get_input().shape
img_shape = input_shape[1:-1]
img = Image.open(argv[2])
img = img.resize(img_shape, Image.ANTIALIAS)
data = np.asarray(img, dtype="uint8")
data = data.reshape(input_shape)

interpreter.run_interactive_no_eval(argv[1], 'data')
